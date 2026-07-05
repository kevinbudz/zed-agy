use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn resolve_cursor_agent_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
    }

    if let Ok(val) = std::env::var("CURSOR_AGENT_EXECUTABLE") {
        let path = PathBuf::from(val);
        if path.exists() {
            return Ok(path);
        }
    }

    let exe_name = if cfg!(windows) {
        "cursor-agent.exe"
    } else {
        "cursor-agent"
    };

    if let Ok(path) = which::which(exe_name) {
        return Ok(path);
    }

    let home = dirs::home_dir().context("could not determine home directory")?;
    let candidates = if cfg!(windows) {
        vec![home
            .join("AppData")
            .join("Local")
            .join("Programs")
            .join("cursor-agent")
            .join("cursor-agent.exe")]
    } else {
        vec![
            home.join(".local").join("bin").join("cursor-agent"),
            PathBuf::from("/opt/homebrew/bin/cursor-agent"),
            PathBuf::from("/usr/local/bin/cursor-agent"),
        ]
    };

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }

    let zed_managed = paths::data_dir().join("cursor-agent").join(exe_name);
    if zed_managed.exists() {
        return Ok(zed_managed);
    }

    anyhow::bail!("cursor-agent not found")
}

pub fn build_cursor_agent_command(
    agent_path: &Path,
    model: &str,
    workspace: &Path,
    force: bool,
) -> tokio::process::Command {
    let resolved_model = crate::models::resolve_cursor_model_id(model);
    let mut cmd = tokio::process::Command::new(agent_path);
    cmd.arg("--print")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--stream-partial-output")
        .arg("--workspace")
        .arg(workspace)
        .arg("--model")
        .arg(resolved_model)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if force {
        cmd.arg("--force");
    }
    cmd
}
