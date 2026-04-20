use std::{
    collections::HashMap,
    ffi::OsString,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
};

use anyhow::Context;
use async_trait::async_trait;
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{broadcast, mpsc, Mutex},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::runtime_env;

use super::{
    ActionType, Engine, EngineEvent, EngineThread, ModelInfo, OutputStream, SandboxPolicy,
    ThreadScope, TurnCompletionStatus, TurnInput,
};

#[derive(Debug, Clone)]
pub struct OpenCodeHealthReport {
    pub available: bool,
    pub version: Option<String>,
    pub details: Option<String>,
    pub warnings: Vec<String>,
    pub checks: Vec<String>,
    pub fixes: Vec<String>,
}

pub trait CliStreamParser: Send {
    fn parse_stdout_line(&mut self, line: &str) -> Vec<EngineEvent>;
    fn parse_stderr_line(&mut self, line: &str) -> Vec<EngineEvent>;
}

#[derive(Clone)]
pub struct CliSpawnConfig {
    pub executable: PathBuf,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: HashMap<String, String>,
}

pub struct CliProcessTransport {
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<ChildStdin>>,
    events_tx: broadcast::Sender<EngineEvent>,
}

impl CliProcessTransport {
    pub async fn spawn(
        config: CliSpawnConfig,
        parser: Box<dyn CliStreamParser>,
    ) -> anyhow::Result<Self> {
        let mut command = Command::new(&config.executable);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(cwd) = config.cwd {
            command.current_dir(cwd);
        }

        if !config.env.is_empty() {
            command.envs(config.env);
        }

        if let Some(path) = opencode_augmented_path(&config.executable) {
            command.env("PATH", path);
        }

        let mut child = command
            .spawn()
            .with_context(|| format!("failed to spawn {}", config.executable.display()))?;

        let stdin = child.stdin.take().context("missing cli stdin")?;
        let stdout = child.stdout.take().context("missing cli stdout")?;
        let stderr = child.stderr.take().context("missing cli stderr")?;

        let (events_tx, _) = broadcast::channel(512);
        let parser = Arc::new(Mutex::new(parser));

        {
            let events_tx = events_tx.clone();
            let parser = parser.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut parser = parser.lock().await;
                    for event in parser.parse_stdout_line(&line) {
                        let _ = events_tx.send(event);
                    }
                }
            });
        }

        {
            let events_tx = events_tx.clone();
            let parser = parser.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut parser = parser.lock().await;
                    for event in parser.parse_stderr_line(&line) {
                        let _ = events_tx.send(event);
                    }
                }
            });
        }

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            events_tx,
        })
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EngineEvent> {
        self.events_tx.subscribe()
    }

    pub async fn write_line(&self, line: &str) -> anyhow::Result<()> {
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;
        Ok(())
    }

    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let mut child = self.child.lock().await;
        if child.try_wait()?.is_none() {
            child.kill().await.ok();
            child.wait().await.ok();
        }
        Ok(())
    }
}

#[derive(Default)]
struct OpenCodeThreadLifecycle {
    turns_by_thread: HashMap<String, bool>,
}

impl OpenCodeThreadLifecycle {
    fn register_thread(&mut self, thread_id: &str) {
        self.turns_by_thread
            .entry(thread_id.to_string())
            .or_insert(false);
    }

    fn mark_turn_started(&mut self, thread_id: &str) {
        self.turns_by_thread.insert(thread_id.to_string(), true);
    }

    fn mark_turn_completed(&mut self, thread_id: &str) {
        self.turns_by_thread.insert(thread_id.to_string(), false);
    }

    fn is_turn_active(&self, thread_id: &str) -> bool {
        self.turns_by_thread
            .get(thread_id)
            .copied()
            .unwrap_or(false)
    }
}

#[derive(Default)]
struct OpenCodeEventParser;

impl OpenCodeEventParser {
    fn map_event(value: &Value) -> Vec<EngineEvent> {
        let event = value
            .get("event")
            .and_then(Value::as_str)
            .or_else(|| value.get("type").and_then(Value::as_str));

        match event {
            Some("turn.started") => vec![EngineEvent::TurnStarted {
                client_turn_id: value
                    .get("turn_id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }],
            Some("turn.completed") => vec![EngineEvent::TurnCompleted {
                token_usage: None,
                status: TurnCompletionStatus::Completed,
            }],
            Some("turn.failed") => {
                let mut out = Vec::new();
                if let Some(error) = EngineEvent::normalized_error(
                    value
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("OpenCode turn failed"),
                    false,
                ) {
                    out.push(error);
                }
                out.push(EngineEvent::TurnCompleted {
                    token_usage: None,
                    status: TurnCompletionStatus::Failed,
                });
                out
            }
            Some("message.delta") => EngineEvent::normalized_text_delta(
                value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .into_iter()
            .collect(),
            Some("reasoning.delta") => EngineEvent::normalized_thinking_delta(
                value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .into_iter()
            .collect(),
            Some("tool.started") => {
                let action_id = value
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("tool")
                    .to_string();
                vec![EngineEvent::ActionStarted {
                    action_id: action_id.clone(),
                    engine_action_id: Some(action_id),
                    action_type: ActionType::Other,
                    summary: value
                        .get("tool")
                        .and_then(Value::as_str)
                        .unwrap_or("tool call")
                        .to_string(),
                    details: value.clone(),
                }]
            }
            Some("tool.stdout") => EngineEvent::normalized_action_output_delta(
                value
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("tool")
                    .to_string(),
                OutputStream::Stdout,
                value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .into_iter()
            .collect(),
            Some("tool.completed") => {
                let action_id = value
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("tool")
                    .to_string();
                vec![EngineEvent::ActionCompleted {
                    action_id,
                    result: super::ActionResult {
                        success: value
                            .get("success")
                            .and_then(Value::as_bool)
                            .unwrap_or(true),
                        output: value
                            .get("output")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        error: value
                            .get("error")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        diff: None,
                        duration_ms: value
                            .get("duration_ms")
                            .and_then(Value::as_u64)
                            .unwrap_or_default(),
                    },
                }]
            }
            _ => Vec::new(),
        }
    }
}

impl CliStreamParser for OpenCodeEventParser {
    fn parse_stdout_line(&mut self, line: &str) -> Vec<EngineEvent> {
        match serde_json::from_str::<Value>(line.trim()) {
            Ok(value) => Self::map_event(&value),
            Err(_) => EngineEvent::normalized_text_delta(line)
                .into_iter()
                .collect(),
        }
    }

    fn parse_stderr_line(&mut self, line: &str) -> Vec<EngineEvent> {
        EngineEvent::normalized_error(line, true)
            .into_iter()
            .collect()
    }
}

pub struct OpenCodeEngine {
    lifecycle: Arc<Mutex<OpenCodeThreadLifecycle>>,
}

impl Default for OpenCodeEngine {
    fn default() -> Self {
        Self {
            lifecycle: Arc::new(Mutex::new(OpenCodeThreadLifecycle::default())),
        }
    }
}

impl OpenCodeEngine {
    pub async fn prewarm(&self) -> anyhow::Result<()> {
        Ok(())
    }

    pub async fn health_report(&self) -> OpenCodeHealthReport {
        let executable = runtime_env::resolve_executable("opencode");
        let checks = vec![
            "opencode executable on PATH".to_string(),
            "opencode --version".to_string(),
        ];
        let fixes = vec!["Install OpenCode with `npm install -g opencode-ai`".to_string()];

        let Some(executable_path) = executable else {
            return OpenCodeHealthReport {
                available: false,
                version: None,
                details: Some("`opencode` executable not found in PATH".to_string()),
                warnings: Vec::new(),
                checks,
                fixes,
            };
        };

        let version = tokio::process::Command::new(&executable_path)
            .arg("--version")
            .output()
            .await
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    None
                }
            })
            .filter(|value| !value.is_empty());

        OpenCodeHealthReport {
            available: version.is_some(),
            version,
            details: None,
            warnings: Vec::new(),
            checks,
            fixes,
        }
    }

    pub async fn list_models_runtime(&self) -> Vec<ModelInfo> {
        match self.fetch_runtime_models().await {
            Ok(models) => models,
            Err(_) => self.models(),
        }
    }

    pub async fn runtime_model_fallback(&self) -> Vec<ModelInfo> {
        self.models()
    }

    async fn fetch_runtime_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let executable = runtime_env::resolve_executable("opencode")
            .context("`opencode` executable not found in PATH")?;

        let output = Command::new(&executable)
            .args(["--json", "models"])
            .output()
            .await
            .with_context(|| format!("failed to execute {}", executable.display()))?;

        if !output.status.success() {
            anyhow::bail!(
                "opencode model listing failed with status {}",
                output.status
            );
        }

        parse_opencode_model_list(&output.stdout)
    }
}

fn parse_opencode_model_list(stdout: &[u8]) -> anyhow::Result<Vec<ModelInfo>> {
    let payload: Value = serde_json::from_slice(stdout).context("invalid opencode models json")?;

    let models = if let Some(models) = payload.as_array() {
        models
    } else if let Some(models) = payload.get("data").and_then(Value::as_array) {
        models
    } else if let Some(models) = payload.get("models").and_then(Value::as_array) {
        models
    } else {
        anyhow::bail!("opencode models payload missing model array")
    };

    Ok(models.iter().map(map_opencode_model).collect())
}

fn map_opencode_model(value: &Value) -> ModelInfo {
    let id = value
        .get("id")
        .or_else(|| value.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("opencode-default")
        .to_string();

    let default_reasoning_effort = value
        .get("defaultReasoningEffort")
        .or_else(|| value.get("default_reasoning_effort"))
        .and_then(Value::as_str)
        .unwrap_or("medium")
        .to_string();

    let supported_reasoning_efforts = value
        .get("supportedReasoningEfforts")
        .or_else(|| value.get("supported_reasoning_efforts"))
        .and_then(Value::as_array)
        .map(|options| {
            options
                .iter()
                .filter_map(|option| {
                    if let Some(reasoning_effort) = option.as_str() {
                        return Some(super::ReasoningEffortOption {
                            reasoning_effort: reasoning_effort.to_string(),
                            description: format!("{reasoning_effort} reasoning effort"),
                        });
                    }

                    let reasoning_effort = option
                        .get("reasoningEffort")
                        .or_else(|| option.get("reasoning_effort"))
                        .and_then(Value::as_str)?;
                    let description = option
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or(reasoning_effort);

                    Some(super::ReasoningEffortOption {
                        reasoning_effort: reasoning_effort.to_string(),
                        description: description.to_string(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
        .unwrap_or_else(|| {
            vec![super::ReasoningEffortOption {
                reasoning_effort: default_reasoning_effort.clone(),
                description: format!("{default_reasoning_effort} reasoning effort"),
            }]
        });

    ModelInfo {
        display_name: value
            .get("displayName")
            .or_else(|| value.get("display_name"))
            .and_then(Value::as_str)
            .unwrap_or(&id)
            .to_string(),
        description: value
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        hidden: value
            .get("hidden")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        is_default: value
            .get("isDefault")
            .or_else(|| value.get("is_default"))
            .or_else(|| value.get("default"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        upgrade: None,
        availability_nux: None,
        upgrade_info: None,
        input_modalities: value
            .get("inputModalities")
            .or_else(|| value.get("input_modalities"))
            .and_then(Value::as_array)
            .map(|modalities| {
                modalities
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter(|modalities| !modalities.is_empty())
            .unwrap_or_else(|| vec!["text".to_string()]),
        supports_personality: value
            .get("supportsPersonality")
            .or_else(|| value.get("supports_personality"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        default_reasoning_effort,
        supported_reasoning_efforts,
        id,
    }
}

#[async_trait]
impl Engine for OpenCodeEngine {
    fn id(&self) -> &str {
        "opencode"
    }

    fn name(&self) -> &str {
        "OpenCode"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "opencode-default".to_string(),
            display_name: "OpenCode".to_string(),
            description: "OpenCode default model".to_string(),
            hidden: false,
            is_default: true,
            upgrade: None,
            availability_nux: None,
            upgrade_info: None,
            input_modalities: vec!["text".to_string()],
            supports_personality: false,
            default_reasoning_effort: "medium".to_string(),
            supported_reasoning_efforts: Vec::new(),
        }]
    }

    async fn is_available(&self) -> bool {
        runtime_env::resolve_executable("opencode").is_some()
    }

    async fn start_thread(
        &self,
        _scope: ThreadScope,
        resume_engine_thread_id: Option<&str>,
        _model: &str,
        _sandbox: SandboxPolicy,
    ) -> Result<EngineThread, anyhow::Error> {
        let thread_id = resume_engine_thread_id
            .map(str::to_string)
            .unwrap_or_else(|| format!("opencode-thread-{}", Uuid::new_v4()));

        let mut lifecycle = self.lifecycle.lock().await;
        lifecycle.register_thread(&thread_id);

        Ok(EngineThread {
            engine_thread_id: thread_id,
        })
    }

    async fn send_message(
        &self,
        engine_thread_id: &str,
        input: TurnInput,
        event_tx: mpsc::Sender<EngineEvent>,
        cancellation: CancellationToken,
    ) -> Result<(), anyhow::Error> {
        let executable = runtime_env::resolve_executable("opencode")
            .context("`opencode` executable not found in PATH")?;

        let config = CliSpawnConfig {
            executable,
            args: vec!["--json".to_string(), "chat".to_string(), input.message],
            cwd: None,
            env: HashMap::new(),
        };

        let transport = CliProcessTransport::spawn(config, Box::<OpenCodeEventParser>::default())
            .await
            .context("failed to start opencode transport")?;

        {
            let mut lifecycle = self.lifecycle.lock().await;
            lifecycle.mark_turn_started(engine_thread_id);
        }

        let mut rx = transport.subscribe();
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    transport.shutdown().await.ok();
                    {
                        let mut lifecycle = self.lifecycle.lock().await;
                        lifecycle.mark_turn_completed(engine_thread_id);
                    }
                    let _ = event_tx.send(EngineEvent::TurnCompleted { token_usage: None, status: TurnCompletionStatus::Interrupted }).await;
                    return Ok(());
                }
                incoming = rx.recv() => {
                    match incoming {
                        Ok(event) => {
                            if matches!(event, EngineEvent::TurnCompleted { .. }) {
                                let mut lifecycle = self.lifecycle.lock().await;
                                lifecycle.mark_turn_completed(engine_thread_id);
                            }
                            let _ = event_tx.send(event).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }

        Ok(())
    }

    async fn steer_message(
        &self,
        _engine_thread_id: &str,
        _input: TurnInput,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn respond_to_approval(
        &self,
        _approval_id: &str,
        _response: Value,
        _route: Option<super::ApprovalRequestRoute>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn interrupt(&self, _engine_thread_id: &str) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn archive_thread(&self, _engine_thread_id: &str) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn unarchive_thread(&self, _engine_thread_id: &str) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

fn opencode_augmented_path(executable: &Path) -> Option<OsString> {
    runtime_env::augmented_path_with_prepend([executable.parent()?.to_path_buf()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::{Mutex, OnceLock},
        time::{SystemTime, UNIX_EPOCH},
    };
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    const OPEN_CODE_EVENT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/opencode/events.jsonl");

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct PathEnvGuard {
        original_path: Option<std::ffi::OsString>,
        temp_dir: PathBuf,
    }

    impl Drop for PathEnvGuard {
        fn drop(&mut self) {
            match &self.original_path {
                Some(path) => std::env::set_var("PATH", path),
                None => std::env::remove_var("PATH"),
            }
            let _ = fs::remove_dir_all(&self.temp_dir);
        }
    }

    fn install_stub_opencode(bin_body: &str) -> anyhow::Result<PathEnvGuard> {
        let temp_dir = std::env::temp_dir().join(format!(
            "panes-opencode-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir)?;
        let binary = temp_dir.join("opencode");
        fs::write(&binary, bin_body)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&binary)?.permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&binary, permissions)?;
        }

        let original_path = std::env::var_os("PATH");
        std::env::set_var("PATH", &temp_dir);
        Ok(PathEnvGuard {
            original_path,
            temp_dir,
        })
    }

    #[test]
    fn opencode_parser_maps_fixture_to_engine_events() {
        let mut parser = OpenCodeEventParser;

        let mapped: Vec<EngineEvent> = OPEN_CODE_EVENT_FIXTURE
            .lines()
            .flat_map(|line| parser.parse_stdout_line(line))
            .collect();

        assert!(mapped
            .iter()
            .any(|event| matches!(event, EngineEvent::TurnStarted { .. })));
        assert!(mapped
            .iter()
            .any(|event| matches!(event, EngineEvent::TextDelta { .. })));
        assert!(mapped
            .iter()
            .any(|event| matches!(event, EngineEvent::ActionStarted { .. })));
        assert!(mapped
            .iter()
            .any(|event| matches!(event, EngineEvent::ActionCompleted { .. })));
        assert!(mapped
            .iter()
            .any(|event| matches!(event, EngineEvent::TurnCompleted { .. })));
    }

    #[test]
    fn opencode_thread_lifecycle_follows_turn_fixture() {
        let mut lifecycle = OpenCodeThreadLifecycle::default();
        let thread_id = "thread-1";
        lifecycle.register_thread(thread_id);

        for line in OPEN_CODE_EVENT_FIXTURE.lines() {
            let value: Value = serde_json::from_str(line).expect("valid fixture line");
            let event = value
                .get("event")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if event == "turn.started" {
                lifecycle.mark_turn_started(thread_id);
            }
            if event == "turn.completed" {
                lifecycle.mark_turn_completed(thread_id);
            }
        }

        assert!(!lifecycle.is_turn_active(thread_id));
    }

    #[tokio::test]
    async fn health_report_detects_opencode_binary() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let _path_guard = install_stub_opencode("#!/bin/sh\necho 0.1.0\n").expect("stub binary");

        let engine = OpenCodeEngine::default();
        let report = engine.health_report().await;

        assert!(report.available);
        assert_eq!(report.version.as_deref(), Some("0.1.0"));
    }

    #[tokio::test]
    async fn start_thread_uses_resume_id_when_present() {
        let engine = OpenCodeEngine::default();
        let thread = engine
            .start_thread(
                ThreadScope::Repo {
                    repo_path: "/tmp/repo".to_string(),
                },
                Some("existing-thread"),
                "opencode-default",
                SandboxPolicy {
                    writable_roots: Vec::new(),
                    allow_network: false,
                    approval_policy: None,
                    reasoning_effort: None,
                    sandbox_mode: None,
                    service_tier: None,
                    personality: None,
                    output_schema: None,
                },
            )
            .await
            .expect("start thread");

        assert_eq!(thread.engine_thread_id, "existing-thread");
    }

    #[tokio::test]
    async fn send_message_streams_and_interrupts_on_cancellation() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let _path_guard = install_stub_opencode(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 0.1.0; exit 0; fi\nprintf '{\"event\":\"turn.started\"}\\n'\nsleep 2\nprintf '{\"event\":\"message.delta\",\"delta\":\"late\"}\\n'\n",
        )
        .expect("stub binary");

        let engine = OpenCodeEngine::default();
        let (tx, mut rx) = mpsc::channel(32);
        let cancellation = CancellationToken::new();
        let send_cancel = cancellation.clone();

        let handle = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            send_cancel.cancel();
        });

        engine
            .send_message(
                "thread-cancel",
                TurnInput {
                    message: "hello".to_string(),
                    attachments: Vec::new(),
                    plan_mode: false,
                    input_items: Vec::new(),
                },
                tx,
                cancellation,
            )
            .await
            .expect("send message");
        handle.await.expect("canceller task");

        let mut saw_turn_started = false;
        let mut saw_interrupted_completion = false;
        while let Ok(event) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            let Some(event) = event else { break };
            if matches!(event, EngineEvent::TurnStarted { .. }) {
                saw_turn_started = true;
            }
            if matches!(
                event,
                EngineEvent::TurnCompleted {
                    status: TurnCompletionStatus::Interrupted,
                    ..
                }
            ) {
                saw_interrupted_completion = true;
                break;
            }
        }

        assert!(saw_turn_started);
        assert!(saw_interrupted_completion);
    }

    #[tokio::test]
    async fn approval_and_interrupt_are_noops_that_succeed() {
        let engine = OpenCodeEngine::default();
        engine
            .respond_to_approval("approval-1", serde_json::json!({"decision":"accept"}), None)
            .await
            .expect("approval response");
        engine.interrupt("thread-1").await.expect("interrupt");
    }

    #[tokio::test]
    async fn list_models_runtime_maps_multiple_models_and_flags() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let _path_guard = install_stub_opencode(
            "#!/bin/sh\nif [ \"$1\" = \"--json\" ] && [ \"$2\" = \"models\" ]; then\ncat <<'JSON'\n{\"data\":[{\"id\":\"openai/gpt-5\",\"displayName\":\"GPT-5\",\"description\":\"default\",\"isDefault\":true,\"hidden\":false,\"defaultReasoningEffort\":\"high\",\"supportedReasoningEfforts\":[{\"reasoningEffort\":\"low\",\"description\":\"Fast\"},{\"reasoningEffort\":\"high\",\"description\":\"Deep\"}]},{\"id\":\"openai/gpt-5-mini\",\"displayName\":\"GPT-5 mini\",\"hidden\":true,\"default\":false,\"default_reasoning_effort\":\"medium\",\"supported_reasoning_efforts\":[\"minimal\",\"medium\"]}]}\nJSON\nexit 0\nfi\nexit 1\n",
        )
        .expect("stub binary");

        let engine = OpenCodeEngine::default();
        let models = engine.list_models_runtime().await;

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "openai/gpt-5");
        assert!(models[0].is_default);
        assert!(!models[0].hidden);
        assert_eq!(models[0].default_reasoning_effort, "high");
        assert_eq!(models[0].supported_reasoning_efforts.len(), 2);
        assert_eq!(
            models[0].supported_reasoning_efforts[1].reasoning_effort,
            "high"
        );

        assert_eq!(models[1].id, "openai/gpt-5-mini");
        assert!(!models[1].is_default);
        assert!(models[1].hidden);
        assert_eq!(models[1].default_reasoning_effort, "medium");
        assert_eq!(
            models[1]
                .supported_reasoning_efforts
                .iter()
                .map(|option| option.reasoning_effort.as_str())
                .collect::<Vec<_>>(),
            vec!["minimal", "medium"]
        );
    }

    #[tokio::test]
    async fn list_models_runtime_falls_back_to_static_models_on_failure() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let _path_guard = install_stub_opencode("#!/bin/sh\nexit 1\n").expect("stub binary");

        let engine = OpenCodeEngine::default();
        let models = engine.list_models_runtime().await;

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "opencode-default");
    }
}
