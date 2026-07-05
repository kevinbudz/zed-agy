use anyhow::{Result, anyhow};
use collections::BTreeMap;
use credentials_provider::CredentialsProvider;
use futures::{AsyncBufReadExt as _, FutureExt, StreamExt, future::BoxFuture, io::BufReader};
use gpui::{App, AsyncApp, Context, Entity, SharedString, Task, Window};
use http_client::{CustomHeaders, HttpClient};
use language_model::{
    AuthenticateError, IconOrSvg, InlineDescription, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelToolChoice, LanguageModelToolSchemaFormat,
    ProviderSettingsView, RateLimiter,
};
use open_ai::{self, ResponseStreamEvent, stream_completion};
use serde::{Deserialize, Serialize};
use settings::Settings;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use ui::{ConfiguredApiCard, prelude::*};
use util::ResultExt as _;

use crate::provider::open_ai::{ChatCompletionMaxTokensParameter, OpenAiEventMapper, into_open_ai};

fn resolve_cursor_agent_path() -> Result<PathBuf> {
    // 1. ZED_CURSOR_AGENT_PATH or CURSOR_AGENT_PATH env var
    if let Ok(val) =
        std::env::var("ZED_CURSOR_AGENT_PATH").or_else(|_| std::env::var("CURSOR_AGENT_PATH"))
    {
        let path = PathBuf::from(val);
        if path.exists() {
            return Ok(path);
        }
    }

    // 2. Search on PATH
    let exe_name = if cfg!(target_os = "windows") {
        "cursor-agent.exe"
    } else {
        "cursor-agent"
    };
    if let Ok(path) = which::which(exe_name) {
        return Ok(path);
    }

    // 3. Check known directories
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))?;
    let candidates = if cfg!(target_os = "windows") {
        vec![
            home.join("AppData")
                .join("Local")
                .join("Programs")
                .join("cursor-agent")
                .join("cursor-agent.exe"),
        ]
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

    // 4. Zed-managed local app support directory
    let zed_managed_path = paths::data_dir().join("cursor-agent").join(exe_name);
    if zed_managed_path.exists() {
        return Ok(zed_managed_path);
    }

    Err(anyhow!("cursor-agent not found"))
}

struct ProxyProcess {
    child: smol::process::Child,
}

impl Drop for ProxyProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn resolve_cursor_proxy_binary() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("ZED_CURSOR_PROXY_PATH") {
        let path = PathBuf::from(val);
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join("cursor-proxy");
            if sibling.exists() {
                return Some(sibling);
            }
        }
    }

    which::which("cursor-proxy").ok()
}

fn start_cursor_proxy(cx: &mut App) -> Option<ProxyProcess> {
    let proxy_binary = resolve_cursor_proxy_binary()?;
    log::info!("Found cursor-proxy binary at: {:?}", proxy_binary);

    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut cmd = smol::process::Command::new(&proxy_binary);
    cmd.env("ZED_WORKSPACE_ROOT", &workspace)
        .env("CURSOR_PROXY_FORCE", "true")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.spawn() {
        Ok(mut child) => {
            log::info!("Successfully spawned Rust cursor-proxy process");

            if let Some(stdout) = child.stdout.take() {
                cx.spawn(async move |_cx| {
                    let mut reader = BufReader::new(stdout);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        log::info!("[Cursor Proxy] {}", line.trim_end());
                        line.clear();
                    }
                })
                .detach();
            }

            if let Some(stderr) = child.stderr.take() {
                cx.spawn(async move |_cx| {
                    let mut reader = BufReader::new(stderr);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        log::warn!("[Cursor Proxy Err] {}", line.trim_end());
                        line.clear();
                    }
                })
                .detach();
            }

            Some(ProxyProcess { child })
        }
        Err(err) => {
            log::error!("Failed to spawn cursor-proxy process: {:?}", err);
            None
        }
    }
}

// Constants
pub const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("cursor");
pub const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Cursor");

const CURSOR_API_KEY_ENV_VAR: &str = "CURSOR_API_KEY";
const CURSOR_DEFAULT_API_URL: &str = "http://127.0.0.1:32124/v1";
const CREDENTIALS_KEY: &str = "cursor-api-key";

pub use settings::CursorAvailableModel as AvailableModel;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct CursorSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CursorCredentials {
    pub api_key: String,
}

pub struct CursorLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    _proxy_process: Option<ProxyProcess>,
}

pub struct State {
    credentials: Option<CursorCredentials>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    load_task: Option<futures::future::Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    sign_in_task: Option<Task<Result<()>>>,
    install_task: Option<Task<Result<()>>>,
    last_auth_error: Option<SharedString>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.credentials.is_some()
    }
    fn is_signing_in(&self) -> bool {
        self.sign_in_task.is_some()
    }
    fn is_installing(&self) -> bool {
        self.install_task.is_some()
    }
}

impl CursorLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|_cx| State {
            credentials: None,
            credentials_provider,
            load_task: None,
            sign_in_task: None,
            install_task: None,
            last_auth_error: None,
        });

        let proxy_process = start_cursor_proxy(cx);

        let this = Self {
            http_client,
            state,
            _proxy_process: proxy_process,
        };
        this.load_credentials(cx);
        this
    }

    fn load_credentials(&self, cx: &mut App) {
        use futures::FutureExt as _;

        let state = self.state.downgrade();
        let load_task = cx
            .spawn(async move |cx| {
                let credentials_provider =
                    state.read_with(&*cx, |s, _| s.credentials_provider.clone())?;

                // First try env var
                if let Ok(api_key) = std::env::var(CURSOR_API_KEY_ENV_VAR) {
                    if !api_key.is_empty() && api_key.to_lowercase() != "cursor-agent" {
                        state.update(&mut *cx, |s, cx| {
                            s.credentials = Some(CursorCredentials { api_key });
                            s.load_task = None;
                            cx.notify();
                        })?;
                        return Ok::<(), Arc<anyhow::Error>>(());
                    }
                }

                // Then try stored credentials
                let result = credentials_provider
                    .read_credentials(CREDENTIALS_KEY, &*cx)
                    .await;
                state.update(&mut *cx, |s, cx| {
                    if let Ok(Some((_, bytes))) = result {
                        match serde_json::from_slice::<CursorCredentials>(&bytes) {
                            Ok(creds) => s.credentials = Some(creds),
                            Err(err) => {
                                log::warn!("Failed to deserialize Cursor credentials: {err}");
                            }
                        }
                    }
                    s.load_task = None;
                    cx.notify();
                })?;

                // Also try reading from Cursor's cli-config.json
                if state
                    .read_with(&*cx, |s, _| s.credentials.is_none())
                    .unwrap_or(true)
                {
                    if let Some(creds) = try_read_cursor_cli_config() {
                        state.update(&mut *cx, |s, cx| {
                            s.credentials = Some(creds);
                            cx.notify();
                        })?;
                    }
                }

                Ok::<(), Arc<anyhow::Error>>(())
            })
            .shared();

        self.state.update(cx, |s, _| {
            s.load_task = Some(load_task);
        });
    }

    fn create_language_model(&self, model: CursorModel) -> Arc<dyn LanguageModel> {
        Arc::new(CursorLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn settings(cx: &App) -> &CursorSettings {
        &crate::AllLanguageModelSettings::get_global(cx).cursor
    }

    fn api_url(cx: &App) -> String {
        let url = &Self::settings(cx).api_url;
        if url.is_empty() {
            CURSOR_DEFAULT_API_URL.to_string()
        } else {
            url.clone()
        }
    }
}

impl LanguageModelProviderState for CursorLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for CursorLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiCursor)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(CursorModel::Auto))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(CursorModel::Sonnet4_6))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Standard predefined models, matching the current Cursor model picker.
        for m in &[
            CursorModel::Auto,
            CursorModel::Composer2_5,
            CursorModel::Opus4_8,
            CursorModel::Gpt5_5,
            CursorModel::Fable5,
            CursorModel::Sonnet5,
            CursorModel::Sonnet4_6,
            CursorModel::Codex5_3,
            CursorModel::Opus4_7,
            CursorModel::GrokBuild0_1,
            CursorModel::Gpt5_4,
            CursorModel::Opus4_6,
            CursorModel::Opus4_5,
            CursorModel::Gpt5_2,
            CursorModel::Gemini31Pro,
            CursorModel::Gpt5_4Mini,
            CursorModel::Gpt5_4Nano,
            CursorModel::Haiku4_5,
            CursorModel::Grok4_3,
            CursorModel::Sonnet4_5,
            CursorModel::Codex5_2,
            CursorModel::Codex5_1Max,
            CursorModel::Gpt5_1,
            CursorModel::Gemini3Flash,
            CursorModel::Gemini35Flash,
            CursorModel::Codex5_1Mini,
            CursorModel::Sonnet4,
            CursorModel::Gpt5Mini,
            CursorModel::Gemini25Flash,
            CursorModel::KimiK2_7Code,
            CursorModel::Glm5_2,
        ] {
            models.insert(m.id().to_string(), m.clone());
        }

        // Custom models from settings
        for model in &Self::settings(cx).available_models {
            models.insert(
                model.name.clone(),
                CursorModel::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                },
            );
        }

        models
            .into_values()
            .map(|model| {
                Arc::new(CursorLanguageModel {
                    id: LanguageModelId::from(model.id().to_string()),
                    model,
                    state: self.state.clone(),
                    http_client: self.http_client.clone(),
                    request_limiter: RateLimiter::new(4),
                }) as Arc<dyn LanguageModel>
            })
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated(cx) {
            return Task::ready(Ok(()));
        }
        let load_task = self.state.read(cx).load_task.clone();
        if let Some(load_task) = load_task {
            let weak_state = self.state.downgrade();
            cx.spawn(async move |cx| {
                let _ = load_task.await;
                let is_auth = weak_state
                    .read_with(cx, |s, _| s.is_authenticated())
                    .unwrap_or(false);
                if is_auth {
                    Ok(())
                } else {
                    Err(anyhow!("Set your Cursor API key to use this provider.").into())
                }
            })
        } else {
            Task::ready(Err(anyhow!(
                "Set your Cursor API key to use this provider."
            )
            .into()))
        }
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let title = if is_authenticated {
            None
        } else {
            Some("Configure Cursor".into())
        };
        let description = if is_authenticated {
            None
        } else {
            Some(InlineDescription::Text(
                "Sign in with Cursor using cursor-agent. This opens Cursor's browser login flow."
                    .into(),
            ))
        };

        Some(ProviderSettingsView::Inline(
            language_model::InlineProviderSettings {
                title,
                description,
                create_view: Arc::new({
                    let state = self.state.clone();
                    let http_client = self.http_client.clone();
                    move |_window, cx| {
                        cx.new(|_cx| ConfigurationView {
                            state: state.clone(),
                            http_client: http_client.clone(),
                        })
                        .into()
                    }
                }),
            },
        ))
    }
}

struct ConfigurationView {
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);

        if state.is_authenticated() {
            let weak_state = self.state.downgrade();
            return v_flex()
                .child(
                    ConfiguredApiCard::new(
                        "cursor-sign-out",
                        SharedString::from("Signed in to Cursor"),
                    )
                    .button_label("Sign Out")
                    .on_click(cx.listener(move |_this, _, _window, cx| {
                        do_sign_out(&weak_state, cx).detach_and_log_err(cx);
                    })),
                )
                .into_any_element();
        }

        let is_installed = resolve_cursor_agent_path().is_ok();
        let is_installing = state.is_installing();
        let is_signing_in = state.is_signing_in();
        let last_auth_error = state.last_auth_error.clone();

        if !is_installed {
            return v_flex()
                .gap_2()
                .child(
                    Button::new(
                        "cursor-install",
                        if is_installing {
                            "Installing Cursor Agent…"
                        } else {
                            "Install Cursor Agent"
                        },
                    )
                    .full_width()
                    .style(ButtonStyle::Filled)
                    .size(ButtonSize::Medium)
                    .loading(is_installing)
                    .disabled(is_installing)
                    .on_click({
                        let state = self.state.clone();
                        move |_, _window, cx| {
                            do_install_cursor_agent(&state, cx);
                        }
                    }),
                )
                .child(
                    Button::new("cursor-sign-in-disabled", "Sign In with Cursor")
                        .full_width()
                        .style(ButtonStyle::Outlined)
                        .size(ButtonSize::Medium)
                        .disabled(true),
                )
                .when_some(last_auth_error, |this, error| {
                    this.child(
                        h_flex()
                            .gap_1()
                            .justify_center()
                            .child(
                                Icon::new(IconName::XCircle)
                                    .color(Color::Error)
                                    .size(IconSize::Small),
                            )
                            .child(Label::new(error).color(Color::Muted)),
                    )
                })
                .into_any_element();
        }

        v_flex()
            .gap_2()
            .child(
                Button::new(
                    "cursor-sign-in",
                    if is_signing_in {
                        "Signing in…"
                    } else {
                        "Sign In with Cursor"
                    },
                )
                .full_width()
                .style(ButtonStyle::Outlined)
                .size(ButtonSize::Medium)
                .loading(is_signing_in)
                .disabled(is_signing_in)
                .on_click({
                    let state = self.state.clone();
                    let http_client = self.http_client.clone();
                    move |_, _window, cx| {
                        do_sign_in(&state, &http_client, cx);
                    }
                }),
            )
            .when_some(last_auth_error, |this, error| {
                this.child(
                    h_flex()
                        .gap_1()
                        .justify_center()
                        .child(
                            Icon::new(IconName::XCircle)
                                .color(Color::Error)
                                .size(IconSize::Small),
                        )
                        .child(Label::new(error).color(Color::Muted)),
                )
            })
            .into_any_element()
    }
}

fn do_install_cursor_agent(state: &Entity<State>, cx: &mut App) {
    if state.read(cx).is_installing() {
        return;
    }

    let weak_state = state.downgrade();
    let task = cx.spawn(async move |cx| {
        let install_result = if cfg!(target_os = "windows") {
            smol::process::Command::new("powershell")
                .args(&["-Command", "irm https://cursor.com/install.ps1 | iex"])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        } else {
            smol::process::Command::new("sh")
                .args(&["-c", "curl -fsS https://cursor.com/install | bash"])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        };

        match install_result {
            Ok(output) if output.status.success() => {
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.install_task = None;
                        s.last_auth_error = None;
                        cx.notify();
                    })
                    .log_err();
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let message = if stderr.trim().is_empty() {
                    stdout.trim().to_string()
                } else {
                    stderr.trim().to_string()
                };
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.install_task = None;
                        s.last_auth_error = Some(format!("Installation failed: {message}").into());
                        cx.notify();
                    })
                    .log_err();
            }
            Err(err) => {
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.install_task = None;
                        s.last_auth_error =
                            Some(format!("Installation failed to start: {err}").into());
                        cx.notify();
                    })
                    .log_err();
            }
        }
        Ok(())
    });

    state.update(cx, |s, _| {
        s.install_task = Some(task);
    });
}

fn do_sign_in(state: &Entity<State>, _http_client: &Arc<dyn HttpClient>, cx: &mut App) {
    if state.read(cx).is_signing_in() {
        return;
    }

    let weak_state = state.downgrade();
    let task = cx.spawn(async move |cx| {
        let agent_path = match resolve_cursor_agent_path() {
            Ok(path) => path,
            Err(err) => {
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.sign_in_task = None;
                        s.last_auth_error = Some(
                            format!("Cursor Agent not found: {err}").into(),
                        );
                        cx.notify();
                    })
                    .log_err();
                return Ok(());
            }
        };

        let login_result = smol::process::Command::new(agent_path)
            .arg("login")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        if let Err(err) = login_result {
            weak_state
                .update(&mut *cx, |s, cx| {
                    s.sign_in_task = None;
                    s.last_auth_error = Some(
                        format!(
                            "Failed to run `cursor-agent login`: {err}. Install Cursor Agent, then try again."
                        )
                        .into(),
                    );
                    cx.notify();
                })
                .log_err();
            return Ok(());
        }

        let output = login_result.unwrap();
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let message = if stderr.trim().is_empty() {
                stdout.trim().to_string()
            } else {
                stderr.trim().to_string()
            };
            weak_state
                .update(&mut *cx, |s, cx| {
                    s.sign_in_task = None;
                    s.last_auth_error = Some(
                        format!("Cursor login failed: {message}").into(),
                    );
                    cx.notify();
                })
                .log_err();
            return Ok(());
        }

        let Some(creds) = try_read_cursor_cli_config() else {
            weak_state
                .update(&mut *cx, |s, cx| {
                    s.sign_in_task = None;
                    s.last_auth_error = Some(
                        "Cursor login completed, but no Cursor auth token was found.".into(),
                    );
                    cx.notify();
                })
                .log_err();
            return Ok(());
        };

        let persist_result = async {
            let credentials_provider =
                weak_state.read_with(&*cx, |s, _| s.credentials_provider.clone())?;
            let json = serde_json::to_vec(&creds)?;
            credentials_provider
                .write_credentials(CREDENTIALS_KEY, "Bearer", &json, &*cx)
                .await?;
            anyhow::Ok(())
        }
        .await;

        match persist_result {
            Ok(()) => {
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.credentials = Some(creds);
                        s.sign_in_task = None;
                        s.last_auth_error = None;
                        cx.notify();
                    })
                    .log_err();
            }
            Err(err) => {
                log::error!("Cursor sign-in failed to persist credentials: {err:?}");
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.sign_in_task = None;
                        s.last_auth_error = Some("Failed to save Cursor credentials".into());
                        cx.notify();
                    })
                    .log_err();
            }
        }

        Ok(())
    });

    state.update(cx, |s, _| {
        s.sign_in_task = Some(task);
    });
}

fn do_sign_out(state: &gpui::WeakEntity<State>, cx: &mut App) -> Task<Result<()>> {
    let weak_state = state.clone();
    weak_state
        .update(cx, |s, cx| {
            s.credentials = None;
            s.last_auth_error = None;
            cx.notify();
        })
        .log_err();

    cx.spawn(async move |cx| {
        let credentials_provider =
            weak_state.read_with(&*cx, |s, _| s.credentials_provider.clone())?;
        credentials_provider
            .delete_credentials(CREDENTIALS_KEY, &*cx)
            .await
            .log_err();
        Ok(())
    })
}

/// Try to read Cursor's access token from its cli-config.json files.
fn try_read_cursor_cli_config() -> Option<CursorCredentials> {
    let home = dirs::home_dir()?;

    let auth_files = ["cli-config.json", "auth.json"];

    let mut search_dirs = Vec::new();
    if cfg!(target_os = "macos") {
        search_dirs.push(home.join(".cursor"));
        search_dirs.push(home.join(".config").join("cursor"));
    } else {
        search_dirs.push(home.join(".config").join("cursor"));
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            let xdg_path = std::path::PathBuf::from(&xdg);
            if xdg_path != home.join(".config") {
                search_dirs.push(xdg_path.join("cursor"));
            }
        }
        search_dirs.push(home.join(".cursor"));
    }

    for dir in &search_dirs {
        for file_name in &auth_files {
            let path = dir.join(file_name);
            if let Ok(contents) = std::fs::read_to_string(&path) {
                // Try cli-config.json format: { "accessToken": "..." }
                #[derive(Deserialize)]
                #[serde(rename_all = "camelCase")]
                struct CliConfig {
                    access_token: Option<String>,
                }
                if let Ok(config) = serde_json::from_str::<CliConfig>(&contents) {
                    if let Some(token) = config.access_token.filter(|t| !t.is_empty()) {
                        return Some(CursorCredentials { api_key: token });
                    }
                }

                // Try legacy auth.json format: { "token": "..." }
                #[derive(Deserialize)]
                struct LegacyAuth {
                    token: Option<String>,
                }
                if let Ok(auth) = serde_json::from_str::<LegacyAuth>(&contents) {
                    if let Some(token) = auth.token.filter(|t| !t.is_empty()) {
                        return Some(CursorCredentials { api_key: token });
                    }
                }
            }
        }
    }

    None
}

#[derive(Clone, Debug, PartialEq)]
pub enum CursorModel {
    Auto,
    Composer2_5,
    Opus4_8,
    Gpt5_5,
    Fable5,
    Sonnet5,
    Sonnet4_6,
    Codex5_3,
    Opus4_7,
    GrokBuild0_1,
    Gpt5_4,
    Opus4_6,
    Opus4_5,
    Gpt5_2,
    Gemini31Pro,
    Gpt5_4Mini,
    Gpt5_4Nano,
    Haiku4_5,
    Grok4_3,
    Sonnet4_5,
    Codex5_2,
    Codex5_1Max,
    Gpt5_1,
    Gemini3Flash,
    Gemini35Flash,
    Codex5_1Mini,
    Sonnet4,
    Gpt5Mini,
    Gemini25Flash,
    KimiK2_7Code,
    Glm5_2,

    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: u64,
        max_output_tokens: Option<u64>,
    },
}

impl CursorModel {
    fn id(&self) -> &str {
        match self {
            Self::Auto => "cursor/auto",
            Self::Composer2_5 => "cursor/composer-2.5",
            Self::Opus4_8 => "cursor/opus-4.8",
            Self::Gpt5_5 => "cursor/gpt-5.5",
            Self::Fable5 => "cursor/fable-5",
            Self::Sonnet5 => "cursor/sonnet-5",
            Self::Sonnet4_6 => "cursor/sonnet-4.6",
            Self::Codex5_3 => "cursor/codex-5.3",
            Self::Opus4_7 => "cursor/opus-4.7",
            Self::GrokBuild0_1 => "cursor/grok-build-0.1",
            Self::Gpt5_4 => "cursor/gpt-5.4",
            Self::Opus4_6 => "cursor/opus-4.6",
            Self::Opus4_5 => "cursor/opus-4.5",
            Self::Gpt5_2 => "cursor/gpt-5.2",
            Self::Gemini31Pro => "cursor/gemini-3.1-pro",
            Self::Gpt5_4Mini => "cursor/gpt-5.4-mini",
            Self::Gpt5_4Nano => "cursor/gpt-5.4-nano",
            Self::Haiku4_5 => "cursor/haiku-4.5",
            Self::Grok4_3 => "cursor/grok-4.3",
            Self::Sonnet4_5 => "cursor/sonnet-4.5",
            Self::Codex5_2 => "cursor/codex-5.2",
            Self::Codex5_1Max => "cursor/codex-5.1-max",
            Self::Gpt5_1 => "cursor/gpt-5.1",
            Self::Gemini3Flash => "cursor/gemini-3-flash",
            Self::Gemini35Flash => "cursor/gemini-3.5-flash",
            Self::Codex5_1Mini => "cursor/codex-5.1-mini",
            Self::Sonnet4 => "cursor/sonnet-4",
            Self::Gpt5Mini => "cursor/gpt-5-mini",
            Self::Gemini25Flash => "cursor/gemini-2.5-flash",
            Self::KimiK2_7Code => "cursor/kimi-k2.7-code",
            Self::Glm5_2 => "cursor/glm-5.2",
            Self::Custom { name, .. } => name,
        }
    }

    /// The model ID sent in the API request (without the "cursor/" prefix).
    fn request_model_id(&self) -> &str {
        self.id().strip_prefix("cursor/").unwrap_or(self.id())
    }

    fn display_name(&self) -> &str {
        match self {
            Self::Auto => "Auto",
            Self::Composer2_5 => "Composer 2.5",
            Self::Opus4_8 => "Opus 4.8",
            Self::Gpt5_5 => "GPT-5.5",
            Self::Fable5 => "Fable 5",
            Self::Sonnet5 => "Sonnet 5",
            Self::Sonnet4_6 => "Sonnet 4.6",
            Self::Codex5_3 => "Codex 5.3",
            Self::Opus4_7 => "Opus 4.7",
            Self::GrokBuild0_1 => "Grok Build 0.1",
            Self::Gpt5_4 => "GPT-5.4",
            Self::Opus4_6 => "Opus 4.6",
            Self::Opus4_5 => "Opus 4.5",
            Self::Gpt5_2 => "GPT-5.2",
            Self::Gemini31Pro => "Gemini 3.1 Pro",
            Self::Gpt5_4Mini => "GPT-5.4 Mini",
            Self::Gpt5_4Nano => "GPT-5.4 Nano",
            Self::Haiku4_5 => "Haiku 4.5",
            Self::Grok4_3 => "Grok 4.3",
            Self::Sonnet4_5 => "Sonnet 4.5",
            Self::Codex5_2 => "Codex 5.2",
            Self::Codex5_1Max => "Codex 5.1 Max",
            Self::Gpt5_1 => "GPT-5.1",
            Self::Gemini3Flash => "Gemini 3 Flash",
            Self::Gemini35Flash => "Gemini 3.5 Flash",
            Self::Codex5_1Mini => "Codex 5.1 Mini",
            Self::Sonnet4 => "Sonnet 4",
            Self::Gpt5Mini => "GPT-5 Mini",
            Self::Gemini25Flash => "Gemini 2.5 Flash",
            Self::KimiK2_7Code => "Kimi K2.7 Code",
            Self::Glm5_2 => "GLM 5.2",
            Self::Custom {
                display_name, name, ..
            } => display_name.as_deref().unwrap_or(name),
        }
    }

    fn max_token_count(&self) -> u64 {
        match self {
            Self::Gemini31Pro | Self::Gemini3Flash | Self::Gemini35Flash | Self::Gemini25Flash => {
                400_000
            }
            Self::Gpt5_5
            | Self::Gpt5_4
            | Self::Gpt5_4Mini
            | Self::Gpt5_4Nano
            | Self::Gpt5_2
            | Self::Gpt5_1
            | Self::Gpt5Mini
            | Self::Codex5_3
            | Self::Codex5_2
            | Self::Codex5_1Max
            | Self::Codex5_1Mini => 128_000,
            Self::Custom { max_tokens, .. } => *max_tokens,
            _ => 200_000,
        }
    }

    fn max_output_tokens(&self) -> Option<u64> {
        match self {
            Self::Gemini31Pro | Self::Gemini3Flash | Self::Gemini35Flash | Self::Gemini25Flash => {
                Some(20480)
            }
            Self::Opus4_8
            | Self::Opus4_7
            | Self::Opus4_6
            | Self::Opus4_5
            | Self::Sonnet5
            | Self::Sonnet4_6
            | Self::Sonnet4_5
            | Self::Sonnet4
            | Self::Haiku4_5 => Some(8192),
            Self::Custom {
                max_output_tokens, ..
            } => *max_output_tokens,
            _ => Some(16384),
        }
    }

    fn supports_thinking(&self) -> bool {
        false
    }
}

pub struct CursorLanguageModel {
    id: LanguageModelId,
    model: CursorModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl CursorLanguageModel {
    fn stream_completion_impl(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>>>
    {
        let http_client = self.http_client.clone();
        let state = self.state.downgrade();
        let (settings_api_url, custom_headers) = self.state.read_with(cx, |_state, cx| {
            (
                CursorLanguageModelProvider::api_url(cx),
                CursorLanguageModelProvider::settings(cx)
                    .custom_headers
                    .clone(),
            )
        });

        let future = cx.spawn(async move |_cx| {
            let creds = state
                .read_with(&*_cx, |s, _| s.credentials.clone())
                .map_err(|e| anyhow!(e))?
                .ok_or_else(|| {
                    anyhow!("Not authenticated with Cursor. Set CURSOR_API_KEY or run 'cursor-agent login'.")
                })?;

            let stream = stream_completion(
                http_client.as_ref(),
                "Cursor",
                &settings_api_url,
                &creds.api_key,
                request,
                &custom_headers,
            )
            .await
            .map_err(|e| anyhow!(e))?;

            Ok(stream)
        });

        future.boxed()
    }
}

impl LanguageModel for CursorLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_thinking()
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        LanguageModelToolSchemaFormat::JsonSchema
    }

    fn telemetry_id(&self) -> String {
        format!("cursor/{}", self.model.request_model_id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let open_ai_request = into_open_ai(
            request,
            self.model.request_model_id(),
            true,  // supports_parallel_tool_calls
            false, // supports_prompt_cache_key
            self.model.max_output_tokens(),
            ChatCompletionMaxTokensParameter::MaxCompletionTokens,
            None,  // reasoning_effort
            false, // interleaved_reasoning
        );

        let completions = self.stream_completion_impl(open_ai_request, cx);
        let future = self.request_limiter.stream(async move {
            let response = completions
                .await
                .map_err(LanguageModelCompletionError::from)?;
            let mapper = OpenAiEventMapper::new();
            Ok(mapper.map_stream(response))
        });

        async move {
            let stream = future.await?;
            Ok(stream.boxed())
        }
        .boxed()
    }
}
