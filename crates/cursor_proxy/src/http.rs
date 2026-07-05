use crate::agent::{build_cursor_agent_command, resolve_cursor_agent_path};
use crate::config::ProxyConfig;
use crate::prompt::build_prompt_from_messages;
use crate::stream::{
    format_sse_done, parse_stream_json_line, LineBuffer, SseConverter,
};
use crate::tools::{
    apply_tool_schema_compat, build_tool_schema_map, detect_host_profile, extract_allowed_tool_names,
    extract_openai_tool_call, is_tool_call_ready_for_emit, try_reroute_edit_to_write,
    HostProfile, ToolArgsAccumulator, ToolExtraction,
};
use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
struct AppState {
    config: ProxyConfig,
}

pub async fn run_server(config: ProxyConfig) -> Result<()> {
    let bind_addr = config.bind_addr;
    let state = AppState { config };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "workspaceDirectory": state.config.workspace,
    }))
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    stream: Option<bool>,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatRequest>,
) -> Result<Response, StatusCode> {
    if body.stream == Some(false) {
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    let model = body.model.unwrap_or_else(|| "auto".to_string());
    let tools = body.tools.unwrap_or_default();
    let allowed = extract_allowed_tool_names(&tools);
    let schemas = build_tool_schema_map(&tools);
    let profile = detect_host_profile(&allowed, &schemas);
    let prompt = build_prompt_from_messages(&body.messages, &tools);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, Infallible>>(32);
    let config = state.config.clone();

    tokio::spawn(async move {
        if let Err(err) = run_agent_stream(
            config,
            model,
            prompt,
            allowed,
            schemas,
            profile,
            tx,
        )
        .await
        {
            log::error!("agent stream failed: {err:#}");
        }
    });

    let stream = ReceiverStream::new(rx).map(|result| result.map(axum::body::Bytes::from));
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

async fn run_agent_stream(
    config: ProxyConfig,
    model: String,
    prompt: String,
    allowed: std::collections::HashSet<String>,
    schemas: std::collections::HashMap<String, Value>,
    profile: HostProfile,
    tx: tokio::sync::mpsc::Sender<Result<String, Infallible>>,
) -> Result<()> {
    let agent_path = resolve_cursor_agent_path(config.cursor_agent_path.as_deref())?;
    let mut child = build_cursor_agent_command(
        &agent_path,
        &model,
        &config.workspace,
        config.force_tools,
    )
    .spawn()
    .context("failed to spawn cursor-agent")?;

    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        stdin.write_all(prompt.as_bytes()).await?;
        stdin.shutdown().await?;
    }

    let id = format!("cursor-proxy-{}", unix_now());
    let created = unix_now() as i64;
    let mut converter = SseConverter::new(&model, &id, created);
    let mut line_buffer = LineBuffer::default();
    let mut accumulator = ToolArgsAccumulator::default();

    let stdout = child
        .stdout
        .take()
        .context("cursor-agent stdout not piped")?;
    let mut reader = BufReader::new(stdout).lines();

    while let Some(line) = reader.next_line().await? {
        for chunk in line_buffer.push_str(&line) {
            let Some(event) = parse_stream_json_line(&chunk) else {
                continue;
            };

            if event.get("type").and_then(|v| v.as_str()) == Some("tool_call") {
                let merged = accumulator.merge_event(&event);
                match extract_openai_tool_call(&merged, &allowed, profile, &schemas) {
                    ToolExtraction::Intercept(raw) => {
                        let schema = schemas.get(&raw.name);
                        let original = raw.arguments.clone();
                        let normalized = apply_tool_schema_compat(
                            &raw.name,
                            &original,
                            schema,
                            profile,
                        );
                        let (final_name, final_args) =
                            if let Some((write_name, write_args)) = try_reroute_edit_to_write(
                                &raw.name,
                                &original,
                                &normalized,
                                &allowed,
                                &schemas,
                            ) {
                                (write_name, write_args)
                            } else {
                                (raw.name.clone(), normalized)
                            };

                        if !is_tool_call_ready_for_emit(&final_name, &final_args, schema, profile)
                        {
                            log::warn!(
                                "refusing incomplete tool call {} {}",
                                final_name,
                                final_args
                            );
                            continue;
                        }

                        let tool_call = json!({
                            "index": 0,
                            "id": raw.id,
                            "type": "function",
                            "function": {
                                "name": final_name,
                                "arguments": final_args.to_string()
                            }
                        });
                        for chunk in converter.tool_call_chunks(&tool_call) {
                            let _ = tx.send(Ok(chunk)).await;
                        }
                        let _ = tx.send(Ok(format_sse_done())).await;
                        let _ = child.kill().await;
                        return Ok(());
                    }
                    ToolExtraction::Skip => {}
                    ToolExtraction::Passthrough => {
                        log::debug!("passthrough tool call");
                    }
                }
                continue;
            }

            for sse in converter.handle_event(&event) {
                let _ = tx.send(Ok(sse)).await;
            }
        }
    }

    let status = child.wait().await?;
    if !status.success() {
        let stderr = child.stderr.take();
        if let Some(stderr) = stderr {
            let mut err_reader = BufReader::new(stderr).lines();
            while let Some(line) = err_reader.next_line().await? {
                log::warn!("cursor-agent stderr: {line}");
            }
        }
    }

    let _ = tx.send(Ok(format_sse_done())).await;
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
