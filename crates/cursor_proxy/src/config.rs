use anyhow::{Context, Result};
use std::net::SocketAddr;
use std::path::PathBuf;

pub const DEFAULT_PORT: u16 = 32124;
pub const DEFAULT_HOST: &str = "127.0.0.1";

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub bind_addr: SocketAddr,
    pub workspace: PathBuf,
    pub cursor_agent_path: Option<PathBuf>,
    pub force_tools: bool,
}

impl ProxyConfig {
    pub fn from_env() -> Result<Self> {
        let port = std::env::var("CURSOR_PROXY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_PORT);
        let host = std::env::var("CURSOR_PROXY_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
        let bind_addr: SocketAddr = format!("{host}:{port}")
            .parse()
            .context("invalid CURSOR_PROXY_HOST/CURSOR_PROXY_PORT")?;

        let workspace = std::env::var("ZED_WORKSPACE_ROOT")
            .or_else(|_| std::env::var("CURSOR_ACP_WORKSPACE"))
            .map(PathBuf::from)
            .or_else(|_| std::env::current_dir())
            .context("could not resolve workspace directory")?;

        let cursor_agent_path = std::env::var("ZED_CURSOR_AGENT_PATH")
            .or_else(|_| std::env::var("CURSOR_AGENT_PATH"))
            .ok()
            .map(PathBuf::from);

        let force_tools = std::env::var("CURSOR_PROXY_FORCE")
            .map(|v| v != "false" && v != "0")
            .unwrap_or(true);

        Ok(Self {
            bind_addr,
            workspace,
            cursor_agent_path,
            force_tools,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.bind_addr)
    }
}
