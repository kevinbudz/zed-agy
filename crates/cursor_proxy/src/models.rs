use std::collections::HashMap;
use std::sync::OnceLock;

static MODEL_ALIASES: &[(&str, &str)] = &[
    ("sonnet-4", "sonnet-4.6"),
    ("sonnet-4-thinking", "sonnet-4.6-thinking"),
    ("claude-4-sonnet", "sonnet-4.6"),
    ("gpt-5-mini", "gpt-5.4-mini"),
    ("gpt-5", "gpt-5.4"),
    ("composer-2.5", "composer-2"),
];

fn model_alias_map() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        MODEL_ALIASES
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    })
}

/// Maps legacy Zed / cursor model IDs to cursor-agent IDs.
pub fn resolve_cursor_model_id(model: &str) -> String {
    let normalized = model
        .trim()
        .trim_start_matches("cursor-acp/")
        .trim_start_matches("cursor/")
        .to_string();
    if normalized.is_empty() {
        return "auto".to_string();
    }
    let key = normalized.to_lowercase();
    model_alias_map()
        .get(&key)
        .cloned()
        .unwrap_or(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_sonnet_4() {
        assert_eq!(resolve_cursor_model_id("cursor/sonnet-4"), "sonnet-4.6");
    }
}
