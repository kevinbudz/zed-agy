use serde_json::Value;
use std::collections::{HashMap, HashSet};

use crate::tools::{apply_tool_schema_compat, build_tool_schema_map, try_reroute_edit_to_write};

pub fn build_prompt_from_messages(messages: &[Value], tools: &[Value]) -> String {
    let mut lines = Vec::new();
    let allowed = extract_allowed_names(tools);
    let schemas = build_tool_schema_map(tools);
    let profile = crate::tools::detect_host_profile(&allowed, &schemas);
    let mut tool_call_names: HashMap<String, String> = HashMap::new();

    if !tools.is_empty() {
        lines.push(build_tool_schema_block(tools));
    }

    for message in messages {
        let role = message
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user");

        if role == "tool" {
            let call_id = message
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let name = message
                .get("name")
                .and_then(|v| v.as_str())
                .or_else(|| tool_call_names.get(call_id).map(String::as_str));
            let body = message
                .get("content")
                .map(|c| {
                    if let Some(s) = c.as_str() {
                        s.to_string()
                    } else {
                        c.to_string()
                    }
                })
                .unwrap_or_default();
            lines.push(format_tool_result(call_id, name, &body));
            continue;
        }

        if role == "assistant" {
            if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
                if !tool_calls.is_empty() {
                    let mut tc_texts = Vec::new();
                    for tc in tool_calls {
                        let formatted = format_assistant_tool_call(tc, &allowed, &schemas, profile);
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            if formatted.name != "?" {
                                tool_call_names.insert(id.to_string(), formatted.name.clone());
                            }
                        }
                        tc_texts.push(formatted.text);
                    }
                    let text = message
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    lines.push(format!(
                        "ASSISTANT: {}{}",
                        if text.is_empty() { String::new() } else { format!("{text}\n") },
                        tc_texts.join("\n")
                    ));
                    continue;
                }
            }
        }

        if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
            lines.push(format!("{}: {content}", role.to_uppercase()));
        } else if let Some(parts) = message.get("content").and_then(|v| v.as_array()) {
            let text: Vec<&str> = parts
                .iter()
                .filter_map(|p| {
                    p.get("type")
                        .and_then(|t| t.as_str())
                        .filter(|t| *t == "text")
                        .and_then(|_| p.get("text"))
                        .and_then(|t| t.as_str())
                })
                .collect();
            if !text.is_empty() {
                lines.push(format!("{}: {}", role.to_uppercase(), text.join("\n")));
            }
        }
    }

    if messages.iter().any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool")) {
        lines.push(
            "The above tool calls have been executed. Continue your response based on these results."
                .to_string(),
        );
    }

    lines.join("\n\n")
}

struct FormattedToolCall {
    text: String,
    name: String,
}

fn format_assistant_tool_call(
    tc: &Value,
    allowed: &HashSet<String>,
    schemas: &HashMap<String, Value>,
    profile: crate::tools::HostProfile,
) -> FormattedToolCall {
    let function = tc.get("function");
    let mut name = function
        .and_then(|f| f.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let mut args = function
        .and_then(|f| f.get("arguments"))
        .and_then(|v| v.as_str())
        .unwrap_or("{}")
        .to_string();

    if let (Some(n), Some(a)) = (
        function.and_then(|f| f.get("name")).and_then(|v| v.as_str()),
        function.and_then(|f| f.get("arguments")).and_then(|v| v.as_str()),
    ) {
        let parsed: Value = serde_json::from_str(a).unwrap_or(Value::String(a.to_string()));
        let schema = schemas.get(n);
        let normalized = apply_tool_schema_compat(n, &parsed, schema, profile);
        if let Some((write_name, write_args)) =
            try_reroute_edit_to_write(n, &parsed, &normalized, allowed, schemas)
        {
            name = write_name;
            args = write_args.to_string();
        } else {
            args = normalized.to_string();
        }
    }

    FormattedToolCall {
        text: format!(
            "tool_call(id: {}, name: {}, args: {})",
            tc.get("id").and_then(|v| v.as_str()).unwrap_or("?"),
            name,
            args
        ),
        name,
    }
}

fn format_tool_result(call_id: &str, name: Option<&str>, body: &str) -> String {
    match name {
        Some(n) => format!("TOOL_RESULT (name: {n}, call_id: {call_id}): {body}"),
        None => format!("TOOL_RESULT (call_id: {call_id}): {body}"),
    }
}

fn build_tool_schema_block(tools: &[Value]) -> String {
    let tool_descs: Vec<String> = tools
        .iter()
        .map(|t| {
            let function = t.get("function").or(Some(t));
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let desc = function
                .and_then(|f| f.get("description"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let params = function
                .and_then(|f| f.get("parameters"))
                .map(|p| p.to_string())
                .unwrap_or_else(|| "{}".to_string());
            format!("- {name}: {desc}\n  Parameters: {params}")
        })
        .collect();
    format!(
        "SYSTEM: You have access to the following tools. When you need to use one, respond with a tool_call in the standard OpenAI format.\n\
         Tool guidance: prefer write/edit for file changes; use bash mainly to run commands/tests.\n\n\
         Available tools:\n{}",
        tool_descs.join("\n")
    )
}

fn extract_allowed_names(tools: &[Value]) -> HashSet<String> {
    crate::tools::extract_allowed_tool_names(tools)
}
