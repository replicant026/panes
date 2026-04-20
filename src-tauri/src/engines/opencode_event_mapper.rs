use std::collections::HashMap;

use serde_json::Value;
use uuid::Uuid;

use super::{
    ActionResult, ActionType, DiffScope, EngineEvent, OutputStream, TokenUsage,
    TurnCompletionStatus, UsageLimitsSnapshot,
};

#[derive(Default)]
pub struct OpenCodeEventMapper {
    engine_action_to_internal: HashMap<String, String>,
    latest_token_usage: Option<TokenUsage>,
}

impl OpenCodeEventMapper {
    pub fn map_event(&mut self, payload: &Value) -> Vec<EngineEvent> {
        let mut out = Vec::new();
        let event_type = extract_any_string(payload, &["type", "event", "name"])
            .map(|value| normalize_event_key(&value))
            .unwrap_or_else(|| "unknown".to_string());

        match event_type.as_str() {
            "turnstarted" | "turnstart" | "responsestarted" => {
                out.push(EngineEvent::TurnStarted {
                    client_turn_id: extract_any_string(
                        payload,
                        &["clientTurnId", "client_turn_id", "turnId", "turn_id", "id"],
                    ),
                });
            }
            "textdelta" | "messagedelta" | "contentdelta" => {
                if let Some(content) =
                    extract_non_empty_string(payload, &["delta", "text", "content"])
                {
                    out.push(EngineEvent::TextDelta { content });
                }
            }
            "actionstarted" | "toolstarted" | "commandstarted" => {
                if let Some(event) = self.map_action_started(payload) {
                    out.push(event);
                }
            }
            "actionoutputdelta" | "tooloutputdelta" | "commandoutputdelta" => {
                if let Some(event) = self.map_action_output_delta(payload) {
                    out.push(event);
                }
            }
            "actionprogressupdated" | "toolprogress" | "progress" => {
                if let Some(event) = self.map_action_progress(payload) {
                    out.push(event);
                }
            }
            "actioncompleted" | "toolcompleted" | "commandcompleted" => {
                if let Some(event) = self.map_action_completed(payload) {
                    out.push(event);
                }
            }
            "approvalrequested" | "approval" => {
                if let Some(event) = self.map_approval_requested(payload) {
                    out.push(event);
                }
            }
            "turndiffupdated" | "diffupdated" => {
                if let Some(diff) = extract_non_empty_string(payload, &["diff"]) {
                    out.push(EngineEvent::DiffUpdated {
                        diff,
                        scope: DiffScope::Turn,
                    });
                }
            }
            "turncompleted" | "responsecompleted" | "completed" => {
                let token_usage =
                    extract_token_usage(payload).or_else(|| self.latest_token_usage.clone());
                out.push(EngineEvent::TurnCompleted {
                    token_usage,
                    status: extract_turn_completion_status(payload),
                });
                self.latest_token_usage = None;
            }
            "error" => {
                out.push(EngineEvent::Error {
                    message: extract_any_string(payload, &["message", "error", "details"])
                        .unwrap_or_else(|| "OpenCode reported an error".to_string()),
                    recoverable: payload
                        .get("recoverable")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
            }
            _ => {}
        }

        if let Some(token_usage) = extract_token_usage(payload) {
            self.latest_token_usage = Some(token_usage.clone());
            out.push(EngineEvent::UsageLimitsUpdated {
                usage: UsageLimitsSnapshot {
                    current_tokens: Some(token_usage.input.saturating_add(token_usage.output)),
                    ..Default::default()
                },
            });
        }

        if let Some(usage_event) = map_usage_limits(payload) {
            out.push(usage_event);
        }

        if let Some(progress_message) =
            extract_non_empty_string(payload, &["progress", "statusMessage"])
        {
            out.push(EngineEvent::Notice {
                kind: "progress".to_string(),
                level: "info".to_string(),
                title: "OpenCode progress".to_string(),
                message: progress_message,
            });
        }

        out
    }

    fn map_action_started(&mut self, payload: &Value) -> Option<EngineEvent> {
        let engine_action_id = extract_any_string(payload, &["actionId", "action_id", "id"]);
        let action_id = self.resolve_or_register_action(engine_action_id.as_deref());
        let action_type = map_action_type(payload);
        let summary = extract_any_string(payload, &["summary", "command", "tool", "name"])
            .unwrap_or_else(|| "OpenCode action".to_string());

        Some(EngineEvent::ActionStarted {
            action_id,
            engine_action_id,
            action_type,
            summary,
            details: payload.clone(),
        })
    }

    fn map_action_output_delta(&mut self, payload: &Value) -> Option<EngineEvent> {
        let action_id = self.resolve_action_id(payload)?;
        let content = extract_non_empty_string(payload, &["delta", "output", "text", "content"])?;
        let stream = extract_any_string(payload, &["stream", "channel", "target"])
            .map(|value| normalize_stream(&value))
            .unwrap_or(OutputStream::Stdout);

        Some(EngineEvent::ActionOutputDelta {
            action_id,
            stream,
            content,
        })
    }

    fn map_action_progress(&mut self, payload: &Value) -> Option<EngineEvent> {
        let action_id = self.resolve_action_id(payload)?;
        let message = extract_non_empty_string(payload, &["message", "progress", "status"])?;
        Some(EngineEvent::ActionProgressUpdated { action_id, message })
    }

    fn map_action_completed(&mut self, payload: &Value) -> Option<EngineEvent> {
        let action_id = self.resolve_action_id(payload)?;
        let status =
            extract_any_string(payload, &["status"]).unwrap_or_else(|| "completed".to_string());
        let normalized_status = status.to_lowercase();
        let success = matches!(normalized_status.as_str(), "completed" | "success" | "ok");

        Some(EngineEvent::ActionCompleted {
            action_id,
            result: ActionResult {
                success,
                output: extract_any_string(payload, &["output", "text", "content"]),
                error: extract_any_string(payload, &["error", "message"]).filter(|_| !success),
                diff: extract_any_string(payload, &["diff"]),
                duration_ms: extract_any_u64(payload, &["durationMs", "duration_ms"]).unwrap_or(0),
            },
        })
    }

    fn map_approval_requested(&mut self, payload: &Value) -> Option<EngineEvent> {
        let approval_id = extract_any_string(payload, &["approvalId", "approval_id", "id"])
            .unwrap_or_else(|| format!("approval-{}", Uuid::new_v4()));

        Some(EngineEvent::ApprovalRequested {
            approval_id,
            action_type: map_action_type(payload),
            summary: extract_any_string(payload, &["summary", "reason", "message"])
                .unwrap_or_else(|| "OpenCode requested approval".to_string()),
            details: payload.clone(),
        })
    }

    fn resolve_or_register_action(&mut self, engine_action_id: Option<&str>) -> String {
        if let Some(engine_action_id) = engine_action_id {
            if let Some(existing) = self.engine_action_to_internal.get(engine_action_id) {
                return existing.clone();
            }

            let action_id = format!("action-{}", Uuid::new_v4());
            self.engine_action_to_internal
                .insert(engine_action_id.to_string(), action_id.clone());
            return action_id;
        }

        format!("action-{}", Uuid::new_v4())
    }

    fn resolve_action_id(&mut self, payload: &Value) -> Option<String> {
        let engine_action_id = extract_any_string(payload, &["actionId", "action_id", "id"])?;

        if let Some(existing) = self.engine_action_to_internal.get(&engine_action_id) {
            return Some(existing.clone());
        }

        let created = format!("action-{}", Uuid::new_v4());
        self.engine_action_to_internal
            .insert(engine_action_id, created.clone());
        Some(created)
    }
}

fn extract_turn_completion_status(payload: &Value) -> TurnCompletionStatus {
    let status = extract_any_string(payload, &["status", "turnStatus", "turn_status"])
        .unwrap_or_else(|| "completed".to_string());
    if status.eq_ignore_ascii_case("failed") || status.eq_ignore_ascii_case("error") {
        TurnCompletionStatus::Failed
    } else if status.eq_ignore_ascii_case("interrupted") || status.eq_ignore_ascii_case("cancelled")
    {
        TurnCompletionStatus::Interrupted
    } else {
        TurnCompletionStatus::Completed
    }
}

fn map_action_type(payload: &Value) -> ActionType {
    let raw = extract_any_string(
        payload,
        &["actionType", "action_type", "kind", "toolType", "tool_type"],
    )
    .unwrap_or_else(|| "other".to_string());
    let normalized = normalize_event_key(&raw);
    match normalized.as_str() {
        "fileread" => ActionType::FileRead,
        "filewrite" => ActionType::FileWrite,
        "fileedit" | "filechange" | "applypatch" => ActionType::FileEdit,
        "filedelete" => ActionType::FileDelete,
        "command" | "shellexec" => ActionType::Command,
        "git" => ActionType::Git,
        "search" | "websearch" => ActionType::Search,
        _ => ActionType::Other,
    }
}

fn map_usage_limits(payload: &Value) -> Option<EngineEvent> {
    let usage = payload.get("usage")?.as_object()?;
    Some(EngineEvent::UsageLimitsUpdated {
        usage: UsageLimitsSnapshot {
            current_tokens: usage
                .get("currentTokens")
                .or_else(|| usage.get("current_tokens"))
                .and_then(Value::as_u64),
            max_context_tokens: usage
                .get("maxContextTokens")
                .or_else(|| usage.get("max_context_tokens"))
                .and_then(Value::as_u64),
            context_window_percent: usage
                .get("contextWindowPercent")
                .or_else(|| usage.get("context_window_percent"))
                .and_then(Value::as_u64)
                .and_then(|v| u8::try_from(v).ok()),
            five_hour_percent: usage
                .get("fiveHourPercent")
                .or_else(|| usage.get("five_hour_percent"))
                .and_then(Value::as_u64)
                .and_then(|v| u8::try_from(v).ok()),
            weekly_percent: usage
                .get("weeklyPercent")
                .or_else(|| usage.get("weekly_percent"))
                .and_then(Value::as_u64)
                .and_then(|v| u8::try_from(v).ok()),
            five_hour_resets_at: usage
                .get("fiveHourResetsAt")
                .or_else(|| usage.get("five_hour_resets_at"))
                .and_then(Value::as_i64),
            weekly_resets_at: usage
                .get("weeklyResetsAt")
                .or_else(|| usage.get("weekly_resets_at"))
                .and_then(Value::as_i64),
        },
    })
}

fn extract_token_usage(payload: &Value) -> Option<TokenUsage> {
    let usage = payload
        .get("tokenUsage")
        .or_else(|| payload.get("token_usage"))?;
    Some(TokenUsage {
        input: usage
            .get("input")
            .or_else(|| usage.get("inputTokens"))
            .or_else(|| usage.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output: usage
            .get("output")
            .or_else(|| usage.get("outputTokens"))
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

fn normalize_stream(raw: &str) -> OutputStream {
    let normalized = raw.to_lowercase();
    if normalized.contains("err") {
        OutputStream::Stderr
    } else if normalized.contains("in") {
        OutputStream::Stdin
    } else {
        OutputStream::Stdout
    }
}

fn extract_non_empty_string(payload: &Value, keys: &[&str]) -> Option<String> {
    extract_any_string(payload, keys).filter(|value| !value.is_empty())
}

fn extract_any_string(payload: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = payload.get(*key).and_then(Value::as_str) {
            return Some(value.to_string());
        }
    }
    None
}

fn extract_any_u64(payload: &Value, keys: &[&str]) -> Option<u64> {
    for key in keys {
        if let Some(value) = payload.get(*key).and_then(Value::as_u64) {
            return Some(value);
        }
    }
    None
}

fn normalize_event_key(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn maps_turn_and_text_deltas() {
        let mut mapper = OpenCodeEventMapper::default();
        let started = mapper.map_event(&json!({
            "type": "turn_started",
            "clientTurnId": "ct_1"
        }));
        assert!(matches!(
            started.first(),
            Some(EngineEvent::TurnStarted {
                client_turn_id: Some(client_turn_id)
            }) if client_turn_id == "ct_1"
        ));

        let text = mapper.map_event(&json!({
            "type": "text_delta",
            "delta": "Olá"
        }));
        assert!(matches!(
            text.first(),
            Some(EngineEvent::TextDelta { content }) if content == "Olá"
        ));
    }

    #[test]
    fn maps_action_lifecycle_and_streams() {
        let mut mapper = OpenCodeEventMapper::default();
        let started = mapper.map_event(&json!({
            "type": "action_started",
            "actionId": "a1",
            "actionType": "command",
            "summary": "Run tests"
        }));

        let action_id = match started.first() {
            Some(EngineEvent::ActionStarted { action_id, .. }) => action_id.clone(),
            _ => panic!("expected action start"),
        };

        let output = mapper.map_event(&json!({
            "type": "action_output_delta",
            "actionId": "a1",
            "stream": "stderr",
            "delta": "warning"
        }));

        assert!(matches!(
            output.first(),
            Some(EngineEvent::ActionOutputDelta {
                action_id: seen_action_id,
                stream: OutputStream::Stderr,
                content
            }) if seen_action_id == &action_id && content == "warning"
        ));

        let completed = mapper.map_event(&json!({
            "type": "action_completed",
            "actionId": "a1",
            "status": "failed",
            "error": "boom"
        }));

        assert!(matches!(
            completed.first(),
            Some(EngineEvent::ActionCompleted {
                result: ActionResult { success: false, .. },
                ..
            })
        ));
    }

    #[test]
    fn emits_approval_and_structured_usage() {
        let mut mapper = OpenCodeEventMapper::default();
        let approval = mapper.map_event(&json!({
            "type": "approval_requested",
            "approvalId": "ap_1",
            "actionType": "file_edit",
            "summary": "Apply patch"
        }));
        assert!(matches!(
            approval.first(),
            Some(EngineEvent::ApprovalRequested { approval_id, .. }) if approval_id == "ap_1"
        ));

        let usage = mapper.map_event(&json!({
            "type": "turn_completed",
            "status": "completed",
            "tokenUsage": {
                "input": 12,
                "output": 34
            },
            "usage": {
                "currentTokens": 100,
                "maxContextTokens": 200,
                "contextWindowPercent": 50
            }
        }));

        assert!(usage.iter().any(|event| matches!(
            event,
            EngineEvent::TurnCompleted {
                token_usage: Some(TokenUsage {
                    input: 12,
                    output: 34
                }),
                status: TurnCompletionStatus::Completed
            }
        )));
        assert!(usage
            .iter()
            .any(|event| matches!(event, EngineEvent::UsageLimitsUpdated { .. })));
    }
}
