use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{
    ApprovalRequestRoute, Engine, EngineEvent, EngineThread, ModelInfo, SandboxPolicy, ThreadScope,
    TurnInput,
};
use crate::engines::codex::{CodexEngine, CodexHealthReport};

pub struct OpenCodeEngine {
    inner: Arc<CodexEngine>,
}

impl Default for OpenCodeEngine {
    fn default() -> Self {
        Self {
            inner: Arc::new(CodexEngine::default()),
        }
    }
}

impl OpenCodeEngine {
    pub async fn prewarm(&self) -> anyhow::Result<()> {
        self.inner.prewarm().await
    }

    pub async fn health_report(&self) -> CodexHealthReport {
        let mut report = self.inner.health_report().await;
        if let Some(details) = report.details.as_ref() {
            report.details = Some(details.replace("`codex`", "`opencode`"));
        }
        report
    }

    pub async fn list_models_runtime(&self) -> Vec<ModelInfo> {
        self.inner.list_models_runtime().await
    }

    pub async fn runtime_model_fallback(&self) -> Vec<ModelInfo> {
        self.inner.runtime_model_fallback().await
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
        self.inner.models()
    }

    async fn is_available(&self) -> bool {
        self.inner.is_available().await
    }

    async fn start_thread(
        &self,
        scope: ThreadScope,
        resume_engine_thread_id: Option<&str>,
        model: &str,
        sandbox: SandboxPolicy,
    ) -> Result<EngineThread, anyhow::Error> {
        self.inner
            .start_thread(scope, resume_engine_thread_id, model, sandbox)
            .await
    }

    async fn send_message(
        &self,
        engine_thread_id: &str,
        input: TurnInput,
        event_tx: mpsc::Sender<EngineEvent>,
        cancellation: CancellationToken,
    ) -> Result<(), anyhow::Error> {
        self.inner
            .send_message(engine_thread_id, input, event_tx, cancellation)
            .await
    }

    async fn steer_message(
        &self,
        engine_thread_id: &str,
        input: TurnInput,
    ) -> Result<(), anyhow::Error> {
        self.inner.steer_message(engine_thread_id, input).await
    }

    async fn respond_to_approval(
        &self,
        approval_id: &str,
        response: serde_json::Value,
        route: Option<ApprovalRequestRoute>,
    ) -> Result<(), anyhow::Error> {
        self.inner
            .respond_to_approval(approval_id, response, route)
            .await
    }

    async fn interrupt(&self, engine_thread_id: &str) -> Result<(), anyhow::Error> {
        self.inner.interrupt(engine_thread_id).await
    }

    async fn archive_thread(&self, engine_thread_id: &str) -> Result<(), anyhow::Error> {
        self.inner.archive_thread(engine_thread_id).await
    }

    async fn unarchive_thread(&self, engine_thread_id: &str) -> Result<(), anyhow::Error> {
        self.inner.unarchive_thread(engine_thread_id).await
    }
}
