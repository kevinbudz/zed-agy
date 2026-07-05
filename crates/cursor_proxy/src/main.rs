use anyhow::Result;
use cursor_proxy::{ProxyConfig, run_server};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    if env::var_os("RUST_LOG").is_none() {
        // SAFETY: called before any other threads are spawned in main.
        unsafe { env::set_var("RUST_LOG", "cursor_proxy=info") };
    }
    env_logger::init();

    let config = ProxyConfig::from_env()?;
    log::info!(
        "cursor-proxy listening on {} workspace={}",
        config.bind_addr,
        config.workspace.display()
    );
    run_server(config).await
}
