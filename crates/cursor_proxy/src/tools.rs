use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostProfile {
    Zed,
    OpenCode,
}

pub fn detect_host_profile(allowed: &HashSet<String>, schemas: &HashMap<String, Value>) -> HostProfile {
    if allowed.contains("read_file") || allowed.contains("edit_file") {
        return HostProfile::Zed;
    }
    for (name, schema) in schemas {
        if is_zed_tool_schema(name, schema) {
            return HostProfile::Zed;
        }
    }
    if allowed.contains("terminal")
        || allowed.contains("find_path")
        || allowed.contains("write_file")
        || allowed.contains("fetch")
        || allowed.contains("diagnostics")
        || allowed.contains("spawn_agent")
        || allowed.contains("copy_path")
        || allowed.contains("move_path")
        || allowed.contains("skill")
    {
        return HostProfile::Zed;
    }
    HostProfile::OpenCode
}

pub fn is_zed_tool_schema(tool_name: &str, schema: &Value) -> bool {
    let tool = tool_name.to_lowercase();
    let props = schema_properties(schema);
    let required = schema_required(schema);
    match tool.as_str() {
        "grep" => props.contains_key("regex") || required.contains(&"regex".to_string()),
        "read_file" => props.contains_key("path") && !props.contains_key("pattern"),
        "terminal" => props.contains_key("cd"),
        "find_path" => props.contains_key("glob"),
        "edit_file" => props.contains_key("edits") || required.contains(&"edits".to_string()),
        "write_file" => {
            props.contains_key("path")
                && !props.contains_key("filePath")
                && !required.contains(&"filePath".to_string())
        }
        "list_directory" | "create_directory" | "delete_path" => props.contains_key("path"),
        "copy_path" | "move_path" => {
            props.contains_key("source_path") && props.contains_key("destination_path")
        }
        "fetch" => props.contains_key("url"),
        "diagnostics" => tool == "diagnostics",
        "spawn_agent" => props.contains_key("label") && props.contains_key("message"),
        "skill" => props.contains_key("name"),
        _ => false,
    }
}

pub fn is_zed_edits_schema(schema: &Value) -> bool {
    let props = schema_properties(schema);
    let required = schema_required(schema);
    props.contains_key("edits") || required.contains(&"edits".to_string())
}

pub fn has_minimum_tool_args(
    tool_name: &str,
    args: &Value,
    schema: Option<&Value>,
    profile: HostProfile,
) -> bool {
    if profile != HostProfile::Zed && !schema.is_some_and(|s| is_zed_tool_schema(tool_name, s)) {
        return true;
    }
    let Some(obj) = args.as_object() else {
        return false;
    };
    let tool = tool_name.to_lowercase();
    match tool.as_str() {
        "grep" => pick_string(obj, &["regex", "pattern", "query", "searchPattern"]).is_some(),
        "read_file" => pick_string(obj, &["path", "filePath"]).is_some(),
        "terminal" | "sandboxed_terminal" => pick_string(obj, &["command", "cmd"]).is_some(),
        "find_path" => pick_string(obj, &["glob", "pattern", "globPattern"]).is_some(),
        "write_file" => {
            pick_string(obj, &["path", "filePath"]).is_some()
                && pick_string(obj, &["content", "new_string", "newString", "streamContent"])
                    .is_some()
        }
        "edit_file" => {
            if pick_string(obj, &["path", "filePath"]).is_none() {
                return false;
            }
            if schema.is_some_and(is_zed_edits_schema) {
                if obj.get("edits").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty()) {
                    return true;
                }
                return pick_string(obj, &["old_string", "oldString", "old_text", "oldText"])
                    .is_some()
                    || pick_string(obj, &["new_string", "newString", "new_text", "newText"])
                        .is_some()
                    || pick_string(obj, &["content", "streamContent"]).is_some();
            }
            true
        }
        "list_directory" | "create_directory" | "delete_path" => {
            pick_string(obj, &["path", "filePath", "directory"]).is_some()
        }
        "fetch" => pick_string(obj, &["url", "uri", "link"]).is_some(),
        "diagnostics" => true,
        "spawn_agent" => {
            pick_string(obj, &["message", "prompt", "task"]).is_some()
        }
        "skill" => pick_string(obj, &["name", "skillName", "skill"]).is_some(),
        "copy_path" | "move_path" => {
            pick_string(obj, &["source_path", "source", "src", "sourcePath"]).is_some()
                && pick_string(
                    obj,
                    &[
                        "destination_path",
                        "destination",
                        "dest",
                        "destinationPath",
                        "target_path",
                        "target",
                    ],
                )
                .is_some()
        }
        _ => true,
    }
}

static ZED_ALIASES: &[(&str, &str)] = &[
    // read
    ("read", "read_file"),
    ("ocread", "read_file"),
    ("readfile", "read_file"),
    // edit / write
    ("edit", "edit_file"),
    ("ocedit", "edit_file"),
    ("strreplace", "edit_file"),
    ("applypatch", "edit_file"),
    ("searchreplace", "edit_file"),
    ("stringreplace", "edit_file"),
    ("write", "write_file"),
    ("ocwrite", "write_file"),
    ("writefile", "write_file"),
    // terminal
    ("bash", "terminal"),
    ("shell", "terminal"),
    ("terminal", "terminal"),
    ("runcommand", "terminal"),
    ("executecommand", "terminal"),
    ("runterminalcommand", "terminal"),
    ("terminalcommand", "terminal"),
    ("shellcommand", "terminal"),
    ("bashcommand", "terminal"),
    ("runbash", "terminal"),
    ("executebash", "terminal"),
    ("cmd", "terminal"),
    ("runterminalcmd", "terminal"),
    // grep
    ("ocgrep", "grep"),
    ("search", "grep"),
    // glob → find_path
    ("glob", "find_path"),
    ("findfiles", "find_path"),
    ("searchfiles", "find_path"),
    ("globfiles", "find_path"),
    ("fileglob", "find_path"),
    ("matchfiles", "find_path"),
    // list directory
    ("ls", "list_directory"),
    ("listdirectory", "list_directory"),
    ("listfiles", "list_directory"),
    ("listdir", "list_directory"),
    ("readdir", "list_directory"),
    // mkdir → create_directory
    ("mkdir", "create_directory"),
    ("createdirectory", "create_directory"),
    ("makedirectory", "create_directory"),
    ("mkdirp", "create_directory"),
    ("createdir", "create_directory"),
    ("makefolder", "create_directory"),
    // rm → delete_path
    ("rm", "delete_path"),
    ("delete", "delete_path"),
    ("deletefile", "delete_path"),
    ("deletepath", "delete_path"),
    ("deletedirectory", "delete_path"),
    ("remove", "delete_path"),
    ("removefile", "delete_path"),
    ("removepath", "delete_path"),
    ("unlink", "delete_path"),
    ("rmdir", "delete_path"),
    // network / diagnostics
    ("webfetch", "fetch"),
    ("readlints", "diagnostics"),
    ("lints", "diagnostics"),
    // delegation
    ("task", "spawn_agent"),
    ("spawnagent", "spawn_agent"),
    ("delegate", "spawn_agent"),
    ("subagent", "spawn_agent"),
    // skill
    ("useskill", "skill"),
    ("invokeskill", "skill"),
    ("runskill", "skill"),
];

fn zed_aliases() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| ZED_ALIASES.iter().copied().collect())
}

pub fn zed_tool_alias(canonical: &str) -> Option<&'static str> {
    zed_aliases()
        .get(normalize_alias_key(canonical).as_str())
        .copied()
}

pub fn resolve_allowed_tool_name(
    name: &str,
    allowed: &HashSet<String>,
    profile: HostProfile,
) -> Option<String> {
    if allowed.contains(name) {
        return Some(name.to_string());
    }
    let normalized = normalize_alias_key(name);
    for allowed_name in allowed {
        if normalize_alias_key(allowed_name) == normalized {
            return Some(allowed_name.clone());
        }
    }
    if profile != HostProfile::Zed {
        return None;
    }
    let aliased = zed_tool_alias(name)?;
    if allowed.contains(aliased) {
        return Some(aliased.to_string());
    }
    let canonical = normalize_alias_key(aliased);
    for allowed_name in allowed {
        if normalize_alias_key(allowed_name) == canonical {
            return Some(allowed_name.clone());
        }
    }
    None
}

pub fn build_tool_schema_map(tools: &[Value]) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    for tool in tools {
        let function = tool
            .get("function")
            .and_then(|v| v.as_object())
            .or_else(|| tool.as_object());
        let Some(function) = function else { continue };
        let Some(name) = function.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        if let Some(params) = function.get("parameters") {
            map.insert(name.to_string(), params.clone());
        }
    }
    map
}

pub fn extract_allowed_tool_names(tools: &[Value]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|f| f.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .collect()
}

pub fn apply_tool_schema_compat(
    tool_name: &str,
    args: &Value,
    schema: Option<&Value>,
    profile: HostProfile,
) -> Value {
    let mut normalized = normalize_argument_keys(args);
    if profile == HostProfile::Zed || schema.is_some_and(|s| is_zed_tool_schema(tool_name, s)) {
        normalized = normalize_zed_tool_args(tool_name, &normalized);
    }
    if is_edit_tool(tool_name) {
        normalized = convert_cursor_edit_to_zed_edits(&normalized);
    }
    if is_write_tool(tool_name) {
        normalized = normalize_write_args(&normalized, schema);
    }
    sanitize_for_schema(&normalized, schema)
}

pub fn try_reroute_edit_to_write(
    tool_name: &str,
    original: &Value,
    normalized: &Value,
    allowed: &HashSet<String>,
    schemas: &HashMap<String, Value>,
) -> Option<(String, Value)> {
    if !is_edit_tool(tool_name) {
        return None;
    }
    let write_name = resolve_write_tool_name(allowed)?;
    if !is_full_file_edit_payload(original, normalized) {
        return None;
    }
    let path = pick_edit_path(normalized, original)?;
    let content = pick_edit_body(normalized, original)?;
    let write_schema = schemas.get(&write_name);
    Some((
        write_name,
        build_write_args(&path, &content, write_schema),
    ))
}

pub fn is_tool_call_ready_for_emit(
    tool_name: &str,
    args: &Value,
    schema: Option<&Value>,
    profile: HostProfile,
) -> bool {
    has_minimum_tool_args(tool_name, args, schema, profile)
        && validate_against_schema(args, schema).is_ok()
}

// --- extract.rs inlined ---

#[derive(Debug, Clone)]
pub struct OpenAiToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub enum ToolExtraction {
    Intercept(OpenAiToolCall),
    Skip,
    Passthrough,
}

pub struct ToolArgsAccumulator {
    merged: HashMap<String, Value>,
}

impl Default for ToolArgsAccumulator {
    fn default() -> Self {
        Self {
            merged: HashMap::new(),
        }
    }
}

impl ToolArgsAccumulator {
    pub fn merge_event(&mut self, event: &Value) -> Value {
        let call_id = event
            .get("call_id")
            .or_else(|| event.get("tool_call_id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let Some(call_id) = call_id else {
            return event.clone();
        };
        if let Some(args) = extract_raw_args(event) {
            let prev = self
                .merged
                .get(&call_id)
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            let mut merged = prev;
            if let Some(new_obj) = args.as_object() {
                for (k, v) in new_obj {
                    merged.insert(k.clone(), v.clone());
                }
            }
            self.merged.insert(call_id.clone(), Value::Object(merged.clone()));
            return inject_args(event, &Value::Object(merged));
        }
        event.clone()
    }
}

pub fn extract_openai_tool_call(
    event: &Value,
    allowed: &HashSet<String>,
    profile: HostProfile,
    schemas: &HashMap<String, Value>,
) -> ToolExtraction {
    if allowed.is_empty() {
        return ToolExtraction::Skip;
    }
    let (name, args, skipped) = extract_tool_name_and_args(event);
    if skipped || name.is_none() {
        return ToolExtraction::Skip;
    }
    let name = name.unwrap();
    let Some(resolved) = resolve_allowed_tool_name(&name, allowed, profile) else {
        return ToolExtraction::Passthrough;
    };
    let schema = schemas.get(&resolved);
    let args_val = args.clone().unwrap_or(Value::Object(Map::new()));
    let args_not_ready = args.is_none() || is_empty_args(&args_val);
    let args_incomplete =
        !args_not_ready && !has_minimum_tool_args(&resolved, &args_val, schema, profile);
    if profile == HostProfile::Zed && (args_not_ready || args_incomplete) {
        return ToolExtraction::Skip;
    }
    let call_id = event
        .get("call_id")
        .or_else(|| event.get("tool_call_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("call_unknown")
        .to_string();
    ToolExtraction::Intercept(OpenAiToolCall {
        id: call_id,
        name: resolved,
        arguments: args_val,
    })
}

fn extract_tool_name_and_args(event: &Value) -> (Option<String>, Option<Value>, bool) {
    let mut name = event.get("name").and_then(|v| v.as_str()).map(str::to_string);
    let tool_call = event.get("tool_call").and_then(|v| v.as_object());
    let Some(tool_call) = tool_call else {
        return (name.map(|n| normalize_tool_name(&n)), None, false);
    };
    let Some((raw_name, payload)) = tool_call.iter().next().map(|(k, v)| (k.as_str(), v)) else {
        return (name, None, false);
    };
    if name.is_none() {
        name = Some(normalize_tool_name(raw_name));
    }
    let payload_obj = payload.as_object();
    let mut args = payload_obj.and_then(|p| p.get("args").cloned());
    if args.is_none() {
        args = payload_obj
            .and_then(|p| p.get("input"))
            .and_then(|v| v.as_object())
            .map(|o| Value::Object(o.clone()));
    }
    if args.is_none() {
        if let Some(p) = payload_obj {
            let rest: Map<String, Value> = p
                .iter()
                .filter(|(k, _)| *k != "result")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if rest.is_empty() {
                return (name, None, true);
            }
            args = Some(Value::Object(rest));
        }
    }
    (name, args, false)
}

fn extract_raw_args(event: &Value) -> Option<Value> {
    let (_, args, skipped) = extract_tool_name_and_args(event);
    if skipped {
        return None;
    }
    args
}

fn inject_args(event: &Value, merged: &Value) -> Value {
    let mut out = event.clone();
    let Some(tool_call) = out.get_mut("tool_call").and_then(|v| v.as_object_mut()) else {
        return out;
    };
    let Some((_, payload)) = tool_call.iter_mut().next() else {
        return out;
    };
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("args".into(), merged.clone());
    }
    out
}

fn is_empty_args(args: &Value) -> bool {
    match args {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

pub fn normalize_tool_name(raw: &str) -> String {
    if let Some(base) = raw.strip_suffix("ToolCall") {
        let mut chars = base.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        }
    } else {
        raw.to_string()
    }
}

pub fn normalize_alias_key(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

// --- private helpers ---

fn resolve_write_tool_name(allowed: &HashSet<String>) -> Option<String> {
    if allowed.contains("write_file") {
        return Some("write_file".to_string());
    }
    if allowed.contains("write") {
        return Some("write".to_string());
    }
    None
}

fn is_edit_tool(name: &str) -> bool {
    matches!(
        normalize_alias_key(name).as_str(),
        "edit" | "editfile" | "strreplace" | "applypatch" | "searchreplace" | "stringreplace"
    )
}

fn is_write_tool(name: &str) -> bool {
    matches!(normalize_alias_key(name).as_str(), "write" | "writefile")
}

fn is_full_file_edit_payload(original: &Value, normalized: &Value) -> bool {
    if had_old_string_in_payload(original) {
        return false;
    }
    pick_edit_path(normalized, original).is_some() && pick_edit_body(normalized, original).is_some()
}

fn had_old_string_in_payload(args: &Value) -> bool {
    args.as_object()
        .is_some_and(|o| o.keys().any(|k| normalize_alias_key(k) == "oldstring"))
}

fn pick_edit_path(normalized: &Value, original: &Value) -> Option<String> {
    normalized
        .as_object()
        .and_then(|o| pick_string(o, &["path", "filePath"]))
        .or_else(|| {
            original
                .as_object()
                .and_then(|o| pick_string(o, &["path", "filePath"]))
        })
}

fn pick_edit_body(normalized: &Value, original: &Value) -> Option<String> {
    normalized
        .as_object()
        .and_then(|o| pick_string(o, &["new_string", "newString", "content", "streamContent"]))
        .or_else(|| {
            original.as_object().and_then(|o| {
                pick_string(o, &["new_string", "newString", "content", "streamContent"])
            })
        })
}

fn build_write_args(path: &str, content: &str, schema: Option<&Value>) -> Value {
    let use_path = schema
        .and_then(|s| schema_required(s).contains(&"path".to_string()).then_some(true))
        .unwrap_or_else(|| {
            schema
                .and_then(|s| s.get("properties"))
                .and_then(|p| p.as_object())
                .is_some_and(|p| p.contains_key("path"))
        });
    if use_path {
        json!({ "path": path, "content": content })
    } else {
        json!({ "filePath": path, "content": content })
    }
}

fn normalize_write_args(args: &Value, schema: Option<&Value>) -> Value {
    let mut obj = args.as_object().cloned().unwrap_or_default();
    if obj.get("content").is_none() {
        if let Some(ns) = pick_string(&obj, &["new_string", "newString"]) {
            obj.insert("content".into(), Value::String(ns));
            obj.remove("new_string");
            obj.remove("newString");
        }
    }
    if let Some(schema) = schema {
        let props = schema_properties(schema);
        if props.contains_key("filePath") && !obj.contains_key("filePath") {
            if let Some(path) = obj.remove("path") {
                obj.insert("filePath".into(), path);
            }
        }
    }
    Value::Object(obj)
}

fn convert_cursor_edit_to_zed_edits(args: &Value) -> Value {
    let mut obj = args.as_object().cloned().unwrap_or_default();
    if let Some(edits) = obj.get("edits").and_then(|v| v.as_array()) {
        if !edits.is_empty() {
            let converted: Vec<Value> = edits
                .iter()
                .filter_map(normalize_zed_edit_entry)
                .collect();
            if !converted.is_empty() {
                obj.insert("edits".into(), Value::Array(converted));
            }
        }
    } else {
        let old = pick_string(&obj, &["old_string", "oldString", "old_text", "oldText"])
            .unwrap_or_default();
        let new = pick_string(&obj, &["new_string", "newString", "new_text", "newText"])
            .unwrap_or_default();
        if !old.is_empty() || !new.is_empty() {
            obj.insert(
                "edits".into(),
                json!([{ "old_text": old, "new_text": new }]),
            );
            for key in [
                "old_string",
                "oldString",
                "new_string",
                "newString",
                "content",
            ] {
                obj.remove(key);
            }
        }
    }
    Value::Object(obj)
}

fn normalize_zed_edit_entry(entry: &Value) -> Option<Value> {
    let obj = entry.as_object()?;
    let old = pick_string(obj, &["old_text", "oldText", "old_string", "oldString"])
        .unwrap_or_default();
    let new = pick_string(obj, &["new_text", "newText", "new_string", "newString", "content"])
        .unwrap_or_default();
    if old.is_empty() && new.is_empty() {
        return None;
    }
    Some(json!({ "old_text": old, "new_text": new }))
}

fn normalize_zed_tool_args(tool_name: &str, args: &Value) -> Value {
    let mut obj = args.as_object().cloned().unwrap_or_default();
    match tool_name.to_lowercase().as_str() {
        "grep" => {
            if let Some(regex) = pick_string(&obj, &["regex", "pattern", "query", "searchPattern"]) {
                obj.insert("regex".into(), Value::String(regex));
            }
            obj.remove("pattern");
            obj.remove("query");
            obj.remove("searchPattern");
            if let Some(include) = pick_string(
                &obj,
                &["include_pattern", "includePattern", "include"],
            ) {
                obj.insert("include_pattern".into(), Value::String(include));
            } else if let Some(path) = obj.get("path").and_then(|v| v.as_str()) {
                if path.contains('*') {
                    obj.insert("include_pattern".into(), Value::String(path.to_string()));
                    obj.remove("path");
                }
            }
            obj.remove("includePattern");
            obj.remove("include");
        }
        "find_path" => {
            if let Some(glob) = pick_string(&obj, &["glob", "pattern", "globPattern"]) {
                obj.insert("glob".into(), Value::String(glob));
            }
            obj.remove("pattern");
            obj.remove("globPattern");
        }
        "terminal" => {
            if let Some(cmd) = pick_string(&obj, &["command", "cmd"]) {
                obj.insert("command".into(), Value::String(cmd));
            }
            obj.remove("cmd");
            let cd = pick_string(&obj, &["cd", "cwd", "workdir", "workingDirectory"])
                .or_else(|| obj.get("path").and_then(|v| v.as_str()).map(str::to_string))
                .or_else(|| {
                    obj.contains_key("command")
                        .then(|| ".".to_string())
                });
            if let Some(cd) = cd {
                obj.insert("cd".into(), Value::String(cd));
            }
            obj.remove("cwd");
            obj.remove("workdir");
            obj.remove("workingDirectory");
            if obj.get("cd").is_some() && obj.get("path").and_then(|v| v.as_str()) == obj.get("cd").and_then(|v| v.as_str()) {
                obj.remove("path");
            }
        }
        "read_file" => {
            if let Some(path) = pick_string(&obj, &["path", "filePath"]) {
                obj.insert("path".into(), Value::String(path));
            }
            obj.remove("filePath");
            if obj.get("start_line").is_none() {
                if let Some(start) = obj.remove("startLine") {
                    obj.insert("start_line".into(), start);
                }
            }
            if obj.get("end_line").is_none() {
                if let Some(end) = obj.remove("endLine") {
                    obj.insert("end_line".into(), end);
                }
            }
        }
        "write_file" => {
            if let Some(path) = pick_string(&obj, &["path", "filePath"]) {
                obj.insert("path".into(), Value::String(path));
            }
            obj.remove("filePath");
            if obj.get("content").is_none() {
                if let Some(ns) = pick_string(&obj, &["new_string", "newString"]) {
                    obj.insert("content".into(), Value::String(ns));
                    obj.remove("new_string");
                    obj.remove("newString");
                }
            }
            if obj.get("content").is_none() {
                if let Some(content) = obj.remove("streamContent") {
                    obj.insert("content".into(), content);
                }
            }
        }
        "list_directory" | "create_directory" | "delete_path" => {
            if let Some(path) = pick_string(&obj, &["path", "filePath", "directory"]) {
                obj.insert("path".into(), Value::String(path));
            }
            obj.remove("filePath");
            obj.remove("directory");
        }
        "edit_file" => {
            if let Some(path) = pick_string(&obj, &["path", "filePath"]) {
                obj.insert("path".into(), Value::String(path));
            }
            obj.remove("filePath");
        }
        "fetch" => {
            if let Some(url) = pick_string(&obj, &["url", "uri", "link"]) {
                obj.insert("url".into(), Value::String(url));
            }
            obj.remove("uri");
            obj.remove("link");
        }
        "diagnostics" => {
            if let Some(path) = pick_string(&obj, &["path", "filePath"]) {
                obj.insert("path".into(), Value::String(path));
            }
            obj.remove("filePath");
        }
        "spawn_agent" => {
            if let Some(message) = pick_string(&obj, &["message", "prompt", "task"]) {
                obj.insert("message".into(), Value::String(message));
            }
            obj.remove("prompt");
            obj.remove("task");
            if obj.get("label").is_none() {
                let label = pick_string(&obj, &["label", "description", "title"])
                    .or_else(|| {
                        obj.get("message")
                            .and_then(|v| v.as_str())
                            .map(synthesize_spawn_label)
                    })
                    .unwrap_or_else(|| "Delegated task".to_string());
                obj.insert("label".into(), Value::String(label));
            }
            obj.remove("description");
            obj.remove("title");
            obj.remove("subagent_type");
            obj.remove("subagentType");
        }
        "skill" => {
            if let Some(name) = pick_string(&obj, &["name", "skillName", "skill"]) {
                obj.insert("name".into(), Value::String(name));
            }
            obj.remove("skillName");
            obj.remove("skill");
        }
        "copy_path" | "move_path" => {
            if let Some(source) = pick_string(&obj, &["source_path", "source", "src", "sourcePath"]) {
                obj.insert("source_path".into(), Value::String(source));
            }
            if let Some(dest) = pick_string(
                &obj,
                &[
                    "destination_path",
                    "destination",
                    "dest",
                    "destinationPath",
                    "target_path",
                    "target",
                ],
            ) {
                obj.insert("destination_path".into(), Value::String(dest));
            }
            obj.remove("source");
            obj.remove("src");
            obj.remove("sourcePath");
            obj.remove("destination");
            obj.remove("dest");
            obj.remove("destinationPath");
            obj.remove("target_path");
            obj.remove("target");
        }
        _ => {}
    }
    Value::Object(obj)
}

fn synthesize_spawn_label(message: &str) -> String {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return "Delegated task".to_string();
    }
    let mut end = trimmed.len().min(60);
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    let label = trimmed[..end].trim();
    if label.is_empty() {
        "Delegated task".to_string()
    } else {
        label.to_string()
    }
}

fn normalize_argument_keys(args: &Value) -> Value {
    let Some(obj) = args.as_object() else {
        return args.clone();
    };
    let mut out = obj.clone();
    for (key, value) in obj {
        if let Some(canonical) = arg_key_alias(key) {
            if !out.contains_key(canonical) {
                out.insert(canonical.to_string(), value.clone());
                out.remove(key);
            }
        }
    }
    Value::Object(out)
}

fn arg_key_alias(key: &str) -> Option<&'static str> {
    match normalize_alias_key(key).as_str() {
        "filepath" | "filename" | "file" => Some("path"),
        "globpattern" | "filepattern" | "searchpattern" => Some("pattern"),
        "includepattern" => Some("include_pattern"),
        "workingdirectory" | "workdir" => Some("cwd"),
        "cmd" | "shellcommand" => Some("command"),
        "contents" | "text" | "body" => Some("content"),
        "oldstring" => Some("old_string"),
        "newstring" => Some("new_string"),
        "sourcepath" | "src" => Some("source_path"),
        "destinationpath" | "dest" | "targetpath" | "target" => Some("destination_path"),
        "skillname" => Some("name"),
        "uri" | "link" => Some("url"),
        "startline" => Some("start_line"),
        "endline" => Some("end_line"),
        _ => None,
    }
}

fn sanitize_for_schema(args: &Value, schema: Option<&Value>) -> Value {
    let Some(schema) = schema else {
        return args.clone();
    };
    if schema.get("additionalProperties").and_then(|v| v.as_bool()) != Some(false) {
        return args.clone();
    }
    let props = schema_properties(schema);
    let Some(obj) = args.as_object() else {
        return args.clone();
    };
    let filtered: Map<String, Value> = obj
        .iter()
        .filter(|(k, _)| props.contains_key(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Value::Object(filtered)
}

fn validate_against_schema(args: &Value, schema: Option<&Value>) -> Result<(), ()> {
    let Some(schema) = schema else {
        return Ok(());
    };
    let Some(obj) = args.as_object() else {
        return Err(());
    };
    for key in schema_required(schema) {
        if !obj.contains_key(&key) {
            return Err(());
        }
    }
    Ok(())
}

fn schema_properties(schema: &Value) -> HashMap<String, Value> {
    schema
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

fn schema_required(schema: &Value) -> Vec<String> {
    schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn pick_string(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(s) = obj.get(*key).and_then(|v| v.as_str()) {
            if !s.trim().is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_edit_strings_to_edits() {
        let args = json!({
            "path": "main.rs",
            "old_string": "old",
            "new_string": "new"
        });
        let schema = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "edits": { "type": "array" }
            },
            "required": ["path", "edits"],
            "additionalProperties": false
        });
        let out = apply_tool_schema_compat("edit_file", &args, Some(&schema), HostProfile::Zed);
        assert_eq!(
            out["edits"],
            json!([{ "old_text": "old", "new_text": "new" }])
        );
    }

    fn allowed(names: &[&str]) -> HashSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn zed_alias_parity_matrix() {
        let cases = [
            ("read", "read_file"),
            ("webFetch", "fetch"),
            ("readLints", "diagnostics"),
            ("task", "spawn_agent"),
            ("useSkill", "skill"),
            ("glob", "find_path"),
            ("bash", "terminal"),
            ("searchreplace", "edit_file"),
        ];
        for (cursor_name, zed_name) in cases {
            let resolved = resolve_allowed_tool_name(
                &normalize_tool_name(&format!("{cursor_name}ToolCall")),
                &allowed(&[zed_name]),
                HostProfile::Zed,
            );
            assert_eq!(resolved.as_deref(), Some(zed_name), "alias for {cursor_name}");
        }
    }

    #[test]
    fn maps_webfetch_to_fetch_args() {
        let schema = json!({
            "type": "object",
            "properties": { "url": { "type": "string" } },
            "required": ["url"],
            "additionalProperties": false
        });
        let out = apply_tool_schema_compat(
            "fetch",
            &json!({ "uri": "https://example.com" }),
            Some(&schema),
            HostProfile::Zed,
        );
        assert_eq!(out["url"], "https://example.com");
    }

    #[test]
    fn maps_task_to_spawn_agent_args() {
        let schema = json!({
            "type": "object",
            "properties": {
                "label": { "type": "string" },
                "message": { "type": "string" }
            },
            "required": ["label", "message"],
            "additionalProperties": false
        });
        let out = apply_tool_schema_compat(
            "spawn_agent",
            &json!({ "prompt": "Investigate flaky login test" }),
            Some(&schema),
            HostProfile::Zed,
        );
        assert_eq!(out["message"], "Investigate flaky login test");
        assert_eq!(out["label"], "Investigate flaky login test");
    }

    #[test]
    fn maps_read_lints_to_diagnostics_path() {
        let schema = json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "additionalProperties": false
        });
        let out = apply_tool_schema_compat(
            "diagnostics",
            &json!({ "filePath": "src/main.rs" }),
            Some(&schema),
            HostProfile::Zed,
        );
        assert_eq!(out["path"], "src/main.rs");
    }

    #[test]
    fn maps_copy_path_source_and_destination() {
        let schema = json!({
            "type": "object",
            "properties": {
                "source_path": { "type": "string" },
                "destination_path": { "type": "string" }
            },
            "required": ["source_path", "destination_path"],
            "additionalProperties": false
        });
        let out = apply_tool_schema_compat(
            "copy_path",
            &json!({ "source": "a.txt", "dest": "b.txt" }),
            Some(&schema),
            HostProfile::Zed,
        );
        assert_eq!(out["source_path"], "a.txt");
        assert_eq!(out["destination_path"], "b.txt");
    }

    #[test]
    fn maps_read_file_line_ranges() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "start_line": { "type": "integer" },
                "end_line": { "type": "integer" }
            },
            "required": ["path"],
            "additionalProperties": false
        });
        let out = apply_tool_schema_compat(
            "read_file",
            &json!({ "path": "main.rs", "startLine": 10, "endLine": 20 }),
            Some(&schema),
            HostProfile::Zed,
        );
        assert_eq!(out["path"], "main.rs");
        assert_eq!(out["start_line"], 10);
        assert_eq!(out["end_line"], 20);
    }

    #[test]
    fn intercepts_cursor_native_tool_events() {
        let allowed = allowed(&["fetch", "spawn_agent", "diagnostics"]);
        let fetch_event = json!({
            "type": "tool_call",
            "call_id": "c1",
            "tool_call": {
                "webFetchToolCall": {
                    "args": { "url": "https://example.com" }
                }
            }
        });
        match extract_openai_tool_call(&fetch_event, &allowed, HostProfile::Zed, &HashMap::new()) {
            ToolExtraction::Intercept(call) => {
                assert_eq!(call.name, "fetch");
                assert_eq!(call.arguments["url"], "https://example.com");
            }
            other => panic!("expected intercept, got {other:?}"),
        }

        let task_event = json!({
            "type": "tool_call",
            "call_id": "c2",
            "tool_call": {
                "taskToolCall": {
                    "args": { "prompt": "analyze repo" }
                }
            }
        });
        match extract_openai_tool_call(&task_event, &allowed, HostProfile::Zed, &HashMap::new()) {
            ToolExtraction::Intercept(call) => {
                assert_eq!(call.name, "spawn_agent");
                assert_eq!(call.arguments["prompt"], "analyze repo");
            }
            other => panic!("expected intercept, got {other:?}"),
        }
    }

    #[test]
    fn detects_zed_host_from_fetch_schema() {
        let mut schemas = HashMap::new();
        schemas.insert(
            "fetch".to_string(),
            json!({
                "type": "object",
                "properties": { "url": { "type": "string" } },
                "required": ["url"]
            }),
        );
        assert_eq!(
            detect_host_profile(&allowed(&[]), &schemas),
            HostProfile::Zed
        );
    }
}
