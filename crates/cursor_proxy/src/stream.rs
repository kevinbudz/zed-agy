use serde_json::{json, Value};

pub struct LineBuffer {
    buffer: String,
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self {
            buffer: String::new(),
        }
    }
}

impl LineBuffer {
    pub fn push_str(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut lines = Vec::new();
        while let Some(pos) = self.buffer.find('\n') {
            let line = self.buffer[..pos].trim_end_matches('\r').to_string();
            self.buffer.drain(..=pos);
            if !line.is_empty() {
                lines.push(line);
            }
        }
        lines
    }
}

pub fn parse_stream_json_line(line: &str) -> Option<Value> {
    serde_json::from_str(line).ok()
}

pub fn extract_text(event: &Value) -> Option<String> {
    if event.get("type")?.as_str()? != "assistant" {
        return None;
    }
    let content = event.get("message")?.get("content")?.as_array()?;
    let mut parts = Vec::new();
    for part in content {
        if part.get("type")?.as_str()? == "text" {
            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                parts.push(text);
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(""))
    }
}

pub fn extract_thinking(event: &Value) -> Option<String> {
    if event.get("type")?.as_str()? != "thinking" {
        return None;
    }
    event
        .get("text")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

pub struct MixedDeltaTracker {
    text: String,
    thinking: String,
}

impl Default for MixedDeltaTracker {
    fn default() -> Self {
        Self {
            text: String::new(),
            thinking: String::new(),
        }
    }
}

impl MixedDeltaTracker {
    pub fn next_text(&mut self, full: &str) -> Option<String> {
        if full.len() <= self.text.len() || !full.starts_with(self.text.as_str()) {
            self.text = full.to_string();
            return Some(full.to_string());
        }
        let delta = full[self.text.len()..].to_string();
        self.text = full.to_string();
        if delta.is_empty() { None } else { Some(delta) }
    }

    pub fn next_thinking(&mut self, full: &str) -> Option<String> {
        if full.len() <= self.thinking.len() || !full.starts_with(self.thinking.as_str()) {
            self.thinking = full.to_string();
            return Some(full.to_string());
        }
        let delta = full[self.thinking.len()..].to_string();
        self.thinking = full.to_string();
        if delta.is_empty() { None } else { Some(delta) }
    }
}

pub struct SseConverter {
    id: String,
    created: i64,
    model: String,
    tracker: MixedDeltaTracker,
}

impl SseConverter {
    pub fn new(model: impl Into<String>, id: impl Into<String>, created: i64) -> Self {
        Self {
            id: id.into(),
            created,
            model: model.into(),
            tracker: MixedDeltaTracker::default(),
        }
    }

    pub fn handle_event(&mut self, event: &Value) -> Vec<String> {
        if let Some(text) = extract_text(event) {
            if let Some(delta) = self.tracker.next_text(&text) {
                return vec![self.chunk(json!({ "content": delta }))];
            }
        }
        if let Some(text) = extract_thinking(event) {
            if let Some(delta) = self.tracker.next_thinking(&text) {
                return vec![self.chunk(json!({ "reasoning_content": delta }))];
            }
        }
        Vec::new()
    }

    pub fn chunk(&self, delta: Value) -> String {
        format_sse_chunk(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null
            }]
        }))
    }

    pub fn tool_call_chunks(&self, tool_call: &Value) -> Vec<String> {
        vec![
            format_sse_chunk(json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [tool_call]
                    },
                    "finish_reason": null
                }]
            })),
            format_sse_chunk(json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }]
            })),
        ]
    }
}

pub fn format_sse_chunk(payload: Value) -> String {
    format!("data: {}\n\n", payload)
}

pub fn format_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}
