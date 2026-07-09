use anyhow::{Context as _, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, StreamExt, future::BoxFuture, future::Shared};
use gpui::{App, AsyncApp, Context, Entity, SharedString, Task, Window};
use http_client::{AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest};
use language_model::{
    AuthenticateError, IconOrSvg, InlineDescription, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelEffortLevel, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoice,
    LanguageModelToolSchemaFormat, ProviderSettingsView, RateLimiter, Speed,
};
use open_ai::ResponseStreamEvent;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use ui::{ConfiguredApiCard, prelude::*};
use url::form_urlencoded;
use util::ResultExt as _;
use x_ai::XAI_API_URL;

// Provider IDs are ordered alphabetically in the LLM Providers settings list
// (via BTreeMap). Use a key that sorts immediately after `google`.
const PROVIDER_ID: LanguageModelProviderId =
    LanguageModelProviderId::new("google_grok_subscription");
const PROVIDER_NAME: LanguageModelProviderName =
    LanguageModelProviderName::new("Grok Subscription");

const SUBSCRIPTION_DESCRIPTION: &str = "Sign in with your SuperGrok or X Premium+ subscription \
    to use Grok models in Zed's agent. No API key required.";

// Public xAI Grok CLI OAuth client used by OpenCode, Kilo, Hermes, and OpenClaw.
const XAI_OAUTH_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_OAUTH_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const XAI_DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";

const CREDENTIALS_KEY: &str = "xai-oauth";
// Access tokens are short-lived (~6h). Refresh up to an hour early so idle
// agent sessions do not hit a narrow expiry window.
const TOKEN_REFRESH_BUFFER_MS: u64 = 60 * 60 * 1000;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct XaiOAuthCredentials {
    access_token: String,
    refresh_token: String,
    expires_at_ms: u64,
    email: Option<String>,
}

impl XaiOAuthCredentials {
    fn is_expired(&self) -> bool {
        now_ms() + TOKEN_REFRESH_BUFFER_MS >= self.expires_at_ms
    }
}

pub struct State {
    credentials: Option<XaiOAuthCredentials>,
    sign_in_task: Option<Task<Result<()>>>,
    refresh_task: Option<Shared<Task<Result<XaiOAuthCredentials, Arc<anyhow::Error>>>>>,
    load_task: Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    auth_generation: u64,
    last_auth_error: Option<SharedString>,
    /// Shown while the device-code flow is waiting for browser approval.
    pending_user_code: Option<SharedString>,
}

#[derive(Debug)]
enum RefreshError {
    Fatal(anyhow::Error),
    Transient(anyhow::Error),
}

impl std::fmt::Display for RefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefreshError::Fatal(e) => write!(f, "{e}"),
            RefreshError::Transient(e) => write!(f, "{e}"),
        }
    }
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.credentials.is_some()
    }

    fn email(&self) -> Option<&str> {
        self.credentials.as_ref().and_then(|c| c.email.as_deref())
    }

    fn is_signing_in(&self) -> bool {
        self.sign_in_task.is_some()
    }
}

pub struct XaiSubscribedProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

impl XaiSubscribedProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|_cx| State {
            credentials: None,
            sign_in_task: None,
            refresh_task: None,
            load_task: None,
            credentials_provider,
            auth_generation: 0,
            last_auth_error: None,
            pending_user_code: None,
        });

        let provider = Self { http_client, state };
        provider.load_credentials(cx);
        provider
    }

    fn load_credentials(&self, cx: &mut App) {
        let state = self.state.downgrade();
        let load_task = cx
            .spawn(async move |cx| {
                let credentials_provider =
                    state.read_with(&*cx, |s, _| s.credentials_provider.clone())?;
                let result = credentials_provider
                    .read_credentials(CREDENTIALS_KEY, &*cx)
                    .await;
                state.update(cx, |s, cx| {
                    if let Ok(Some((_, bytes))) = result {
                        match serde_json::from_slice::<XaiOAuthCredentials>(&bytes) {
                            Ok(creds) => s.credentials = Some(creds),
                            Err(err) => {
                                log::warn!(
                                    "Failed to deserialize Grok subscription credentials: {err}"
                                );
                            }
                        }
                    }
                    s.load_task = None;
                    cx.notify();
                })?;
                Ok::<(), Arc<anyhow::Error>>(())
            })
            .shared();

        self.state.update(cx, |s, _| {
            s.load_task = Some(load_task);
        });
    }

    fn create_language_model(&self, model: GrokSubscriptionModel) -> Arc<dyn LanguageModel> {
        Arc::new(XaiSubscribedLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }
}

impl LanguageModelProviderState for XaiSubscribedProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for XaiSubscribedProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiXAi)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(GrokSubscriptionModel::Grok45))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(GrokSubscriptionModel::GrokBuild))
    }

    fn provided_models(&self, _cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        GrokSubscriptionModel::all()
            .into_iter()
            .map(|m| self.create_language_model(m))
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
                    .read_with(&*cx, |s, _| s.is_authenticated())
                    .unwrap_or(false);
                if is_auth {
                    Ok(())
                } else {
                    Err(anyhow!(
                        "Sign in with your SuperGrok or X Premium+ subscription to use this provider."
                    )
                    .into())
                }
            })
        } else {
            Task::ready(Err(anyhow!(
                "Sign in with your SuperGrok or X Premium+ subscription to use this provider."
            )
            .into()))
        }
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let title = if is_authenticated {
            None
        } else {
            Some("Configure Grok Subscription".into())
        };
        let description = if is_authenticated {
            None
        } else {
            Some(InlineDescription::Text(SUBSCRIPTION_DESCRIPTION.into()))
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
                            compact: true,
                        })
                        .into()
                    }
                }),
            },
        ))
    }

    fn authentication_error_message(&self) -> SharedString {
        "Your Grok subscription session is invalid or has expired. \
        Sign in again via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn missing_credentials_error_message(&self) -> SharedString {
        "You are not signed in to your Grok account. \
        Sign in via Settings > AI > LLM Providers to continue."
            .into()
    }
}

// Models available through SuperGrok / X Premium+ OAuth. Separate from the
// API-key xAI catalog so subscription defaults stay clear.
#[derive(Clone, Debug, PartialEq)]
enum GrokSubscriptionModel {
    GrokBuild,
    Grok45,
    /// Cursor's Composer 2.5. Standard and fast variants share this catalog
    /// entry; Fast Mode selects `grok-composer-2.5-fast` at request time.
    Composer25,
}

impl GrokSubscriptionModel {
    fn all() -> Vec<Self> {
        vec![Self::GrokBuild, Self::Grok45, Self::Composer25]
    }

    fn id(&self) -> &str {
        match self {
            Self::GrokBuild => "grok-build-0.1",
            Self::Grok45 => "grok-4.5",
            Self::Composer25 => "grok-composer-2.5",
        }
    }

    fn request_model_id(&self, speed: Option<Speed>) -> &str {
        match self {
            Self::Composer25 => match speed {
                Some(Speed::Fast) => "grok-composer-2.5-fast",
                Some(Speed::Standard) | None => "grok-composer-2.5",
            },
            other => other.id(),
        }
    }

    fn display_name(&self) -> &str {
        match self {
            Self::GrokBuild => "Grok Build",
            Self::Grok45 => "Grok 4.5",
            Self::Composer25 => "Composer 2.5",
        }
    }

    fn max_token_count(&self) -> u64 {
        match self {
            Self::GrokBuild => 256_000,
            // https://docs.x.ai/developers/models/grok-4.5
            Self::Grok45 => 500_000,
            // Cursor Composer 2.5 context window (standard and fast).
            Self::Composer25 => 200_000,
        }
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(64_000)
    }

    fn supports_images(&self) -> bool {
        // Composer 2.5 is text-only; Grok models accept images.
        !matches!(self, Self::Composer25)
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn supports_reasoning_effort(&self) -> bool {
        matches!(self, Self::Grok45)
    }

    fn supports_fast_mode(&self) -> bool {
        matches!(self, Self::Composer25)
    }

    fn requires_json_schema_subset(&self) -> bool {
        true
    }
}

struct XaiSubscribedLanguageModel {
    id: LanguageModelId,
    model: GrokSubscriptionModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

fn xai_subscribed_reasoning_efforts(
    model: &GrokSubscriptionModel,
) -> &'static [open_ai::ReasoningEffort] {
    if model.supports_reasoning_effort() {
        &[
            open_ai::ReasoningEffort::None,
            open_ai::ReasoningEffort::Low,
            open_ai::ReasoningEffort::Medium,
            open_ai::ReasoningEffort::High,
        ]
    } else {
        &[]
    }
}

fn default_thinking_reasoning_effort(
    model: &GrokSubscriptionModel,
) -> Option<open_ai::ReasoningEffort> {
    if model.supports_reasoning_effort() {
        Some(open_ai::ReasoningEffort::Low)
    } else {
        None
    }
}

fn reasoning_effort_for_request(
    request: &LanguageModelRequest,
    model: &GrokSubscriptionModel,
) -> Option<open_ai::ReasoningEffort> {
    let supported_efforts = xai_subscribed_reasoning_efforts(model);
    if supported_efforts.is_empty() {
        return None;
    }

    if request.thinking_allowed {
        request
            .thinking_effort
            .as_deref()
            .and_then(|effort| effort.parse::<open_ai::ReasoningEffort>().ok())
            .filter(|effort| supported_efforts.contains(effort))
            .filter(|effort| *effort != open_ai::ReasoningEffort::None)
            .or_else(|| default_thinking_reasoning_effort(model))
    } else if supported_efforts.contains(&open_ai::ReasoningEffort::None) {
        Some(open_ai::ReasoningEffort::None)
    } else {
        None
    }
}

fn supported_thinking_effort_levels(model: &GrokSubscriptionModel) -> Vec<LanguageModelEffortLevel> {
    let default_effort = default_thinking_reasoning_effort(model);
    xai_subscribed_reasoning_efforts(model)
        .iter()
        .copied()
        .filter_map(|effort| {
            let (name, value) = match effort {
                open_ai::ReasoningEffort::None => return None,
                open_ai::ReasoningEffort::Minimal => ("Minimal", "minimal"),
                open_ai::ReasoningEffort::Low => ("Low", "low"),
                open_ai::ReasoningEffort::Medium => ("Medium", "medium"),
                open_ai::ReasoningEffort::High => ("High", "high"),
                open_ai::ReasoningEffort::XHigh => ("Extra High", "xhigh"),
                open_ai::ReasoningEffort::Max => return None,
            };

            Some(LanguageModelEffortLevel {
                name: name.into(),
                value: value.into(),
                is_default: Some(effort) == default_effort,
            })
        })
        .collect()
}

impl XaiSubscribedLanguageModel {
    fn stream_completion_request(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let state = self.state.downgrade();
        let request_limiter = self.request_limiter.clone();

        let future = cx.spawn(async move |cx| {
            let creds = get_fresh_credentials(&state, &http_client, cx).await?;
            let access_token = creds.access_token.clone();
            let extra_headers = CustomHeaders::new(vec![]);

            request_limiter
                .stream(async move {
                    open_ai::stream_completion(
                        http_client.as_ref(),
                        PROVIDER_NAME.0.as_str(),
                        XAI_API_URL,
                        &access_token,
                        request,
                        &extra_headers,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::from)
                })
                .await
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for XaiSubscribedLanguageModel {
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
        self.model.supports_tools()
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_reasoning_effort()
    }

    fn supports_fast_mode(&self) -> bool {
        self.model.supports_fast_mode()
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        supported_thinking_effort_levels(&self.model)
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        if self.model.requires_json_schema_subset() {
            LanguageModelToolSchemaFormat::JsonSchemaSubset
        } else {
            LanguageModelToolSchemaFormat::JsonSchema
        }
    }

    fn telemetry_id(&self) -> String {
        format!("google_grok_subscription/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn supports_split_token_display(&self) -> bool {
        true
    }

    fn stream_completion(
        &self,
        mut request: LanguageModelRequest,
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
        let model_id = self.model.request_model_id(request.speed).to_string();
        // Fast Mode is mapped to a different model id for Composer; do not
        // forward `service_tier` (OpenAI priority) to the xAI endpoint.
        request.speed = None;
        let reasoning_effort = reasoning_effort_for_request(&request, &self.model);
        let request = crate::provider::open_ai::into_open_ai(
            request,
            &model_id,
            self.model.supports_parallel_tool_calls(),
            /*supports_prompt_cache_key*/ false,
            self.max_output_tokens(),
            crate::provider::open_ai::ChatCompletionMaxTokensParameter::MaxCompletionTokens,
            reasoning_effort,
            false,
        );
        let completions = self.stream_completion_request(request, cx);
        async move {
            let mapper = crate::provider::open_ai::OpenAiEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

async fn get_fresh_credentials(
    state: &gpui::WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    cx: &mut AsyncApp,
) -> Result<XaiOAuthCredentials, LanguageModelCompletionError> {
    let (creds, existing_task) = state
        .read_with(&*cx, |s, _| (s.credentials.clone(), s.refresh_task.clone()))
        .map_err(LanguageModelCompletionError::Other)?;

    let creds = creds.ok_or(LanguageModelCompletionError::NoApiKey {
        provider: PROVIDER_NAME,
    })?;

    if !creds.is_expired() {
        return Ok(creds);
    }

    if let Some(shared_task) = existing_task {
        return shared_task
            .await
            .map_err(|e| LanguageModelCompletionError::Other(anyhow::anyhow!("{e}")));
    }

    let http_client_clone = http_client.clone();
    let state_clone = state.clone();
    let refresh_token_value = creds.refresh_token.clone();
    let generation = state
        .read_with(&*cx, |s, _| s.auth_generation)
        .map_err(LanguageModelCompletionError::Other)?;

    let shared_task = cx
        .spawn(async move |cx| {
            let result = refresh_token(&http_client_clone, &refresh_token_value).await;

            match result {
                Ok(refreshed) => {
                    let persist_result: Result<XaiOAuthCredentials, Arc<anyhow::Error>> = async {
                        let current_generation = state_clone
                            .read_with(&*cx, |s, _| s.auth_generation)
                            .map_err(|e| Arc::new(e))?;
                        if current_generation != generation {
                            return Err(Arc::new(anyhow!(
                                "Sign-out occurred during token refresh"
                            )));
                        }

                        let credentials_provider = state_clone
                            .read_with(&*cx, |s, _| s.credentials_provider.clone())
                            .map_err(|e| Arc::new(e))?;

                        let json =
                            serde_json::to_vec(&refreshed).map_err(|e| Arc::new(e.into()))?;

                        credentials_provider
                            .write_credentials(CREDENTIALS_KEY, "Bearer", &json, &*cx)
                            .await
                            .map_err(|e| Arc::new(e))?;

                        state_clone
                            .update(cx, |s, _| {
                                s.credentials = Some(refreshed.clone());
                                s.refresh_task = None;
                            })
                            .map_err(|e| Arc::new(e))?;

                        Ok(refreshed)
                    }
                    .await;

                    if persist_result.is_err() {
                        let _ = state_clone.update(cx, |s, _| {
                            s.refresh_task = None;
                        });
                    }

                    persist_result
                }
                Err(RefreshError::Fatal(e)) => {
                    log::error!("Grok subscription token refresh failed fatally: {e:?}");
                    let _ = state_clone.update(cx, |s, cx| {
                        s.refresh_task = None;
                        s.credentials = None;
                        s.last_auth_error =
                            Some("Your session has expired. Please sign in again.".into());
                        cx.notify();
                    });
                    if let Ok(credentials_provider) =
                        state_clone.read_with(&*cx, |s, _| s.credentials_provider.clone())
                    {
                        credentials_provider
                            .delete_credentials(CREDENTIALS_KEY, &*cx)
                            .await
                            .log_err();
                    }
                    Err(Arc::new(e))
                }
                Err(RefreshError::Transient(e)) => {
                    log::warn!("Grok subscription token refresh failed transiently: {e:?}");
                    let _ = state_clone.update(cx, |s, _| {
                        s.refresh_task = None;
                    });
                    Err(Arc::new(e))
                }
            }
        })
        .shared();

    state
        .update(cx, |s, _| {
            s.refresh_task = Some(shared_task.clone());
        })
        .map_err(LanguageModelCompletionError::Other)?;

    shared_task
        .await
        .map_err(|e| LanguageModelCompletionError::Other(anyhow::anyhow!("{e}")))
}

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    expires_in: u64,
    interval: u64,
}

// Poll responses may be either a successful token payload or an OAuth error
// object (`authorization_pending`, `slow_down`, etc.) without tokens.
#[derive(Deserialize)]
struct TokenResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

async fn request_device_code(http_client: &Arc<dyn HttpClient>) -> Result<DeviceCodeResponse> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("client_id", XAI_OAUTH_CLIENT_ID)
        .append_pair("scope", XAI_OAUTH_SCOPE)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(XAI_DEVICE_CODE_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(AsyncBody::from(body))?;

    let mut response = http_client.send(request).await?;
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "xAI device-code request failed (HTTP {}): {body}",
            response.status()
        ));
    }

    serde_json::from_str(&body).context("Failed to parse xAI device-code response")
}

async fn poll_for_device_token(
    http_client: &Arc<dyn HttpClient>,
    device_code: &str,
    expires_in: u64,
    poll_interval: u64,
) -> Result<TokenResponse> {
    let deadline = now_ms() + expires_in.saturating_mul(1000);
    let mut current_interval = poll_interval.max(1);

    while now_ms() < deadline {
        let body = form_urlencoded::Serializer::new(String::new())
            .append_pair(
                "grant_type",
                "urn:ietf:params:oauth:grant-type:device_code",
            )
            .append_pair("client_id", XAI_OAUTH_CLIENT_ID)
            .append_pair("device_code", device_code)
            .finish();

        let request = HttpRequest::builder()
            .method(Method::POST)
            .uri(XAI_TOKEN_URL)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .body(AsyncBody::from(body))?;

        let mut response = http_client.send(request).await?;
        let mut body = String::new();
        smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;

        let tokens: TokenResponse = serde_json::from_str(&body).with_context(|| {
            format!("Failed to parse xAI token response (HTTP {}): {body}", response.status())
        })?;

        let access_token = tokens
            .access_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty());

        if response.status().is_success() {
            if access_token.is_none() {
                return Err(anyhow!(
                    "xAI device-code token response was missing access_token"
                ));
            }
            if tokens.refresh_token.as_deref().unwrap_or("").is_empty() {
                return Err(anyhow!(
                    "xAI device-code token response was missing refresh_token"
                ));
            }
            return Ok(tokens);
        }

        match tokens.error.as_deref() {
            Some("authorization_pending") => {
                smol::Timer::after(Duration::from_secs(current_interval)).await;
                continue;
            }
            Some("slow_down") => {
                current_interval = (current_interval + 1).min(30);
                smol::Timer::after(Duration::from_secs(current_interval)).await;
                continue;
            }
            Some(code) => {
                let description = tokens
                    .error_description
                    .as_deref()
                    .unwrap_or(code);
                return Err(anyhow!("xAI device authorization failed: {description}"));
            }
            None => {
                return Err(anyhow!(
                    "xAI device-code token polling failed (HTTP {}): {body}",
                    response.status()
                ));
            }
        }
    }

    Err(anyhow!("Timed out waiting for xAI device authorization"))
}

async fn do_oauth_flow(
    http_client: Arc<dyn HttpClient>,
    state: gpui::WeakEntity<State>,
    cx: &mut AsyncApp,
) -> Result<XaiOAuthCredentials> {
    let device = request_device_code(&http_client).await?;
    let verification_url = device
        .verification_uri_complete
        .clone()
        .unwrap_or_else(|| device.verification_uri.clone());

    let _ = state.update(cx, |s, cx| {
        s.pending_user_code = Some(device.user_code.clone().into());
        s.last_auth_error = None;
        cx.notify();
    });

    let _ = cx.update(|cx| cx.open_url(&verification_url));

    let tokens = poll_for_device_token(
        &http_client,
        &device.device_code,
        device.expires_in,
        device.interval,
    )
    .await?;

    let access_token = tokens
        .access_token
        .filter(|t| !t.is_empty())
        .context("xAI token response missing access_token")?;
    let refresh_token = tokens
        .refresh_token
        .filter(|t| !t.is_empty())
        .context("xAI token response missing refresh_token")?;
    let expires_in = tokens.expires_in.unwrap_or(6 * 60 * 60);
    let email = tokens
        .id_token
        .as_deref()
        .and_then(extract_email_from_jwt);

    Ok(XaiOAuthCredentials {
        access_token,
        refresh_token,
        expires_at_ms: now_ms() + expires_in.saturating_mul(1000),
        email,
    })
}

async fn refresh_token(
    client: &Arc<dyn HttpClient>,
    refresh_token: &str,
) -> Result<XaiOAuthCredentials, RefreshError> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("client_id", XAI_OAUTH_CLIENT_ID)
        .append_pair("refresh_token", refresh_token)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(XAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(AsyncBody::from(body))
        .map_err(|e| RefreshError::Transient(e.into()))?;

    let mut response = client
        .send(request)
        .await
        .map_err(RefreshError::Transient)?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body)
        .await
        .map_err(|e| RefreshError::Transient(e.into()))?;

    if !status.is_success() {
        let err = anyhow!("Token refresh failed (HTTP {status}): {body}");
        // 403 often means tier/entitlement gating rather than a revocable token,
        // but re-auth will not help either way for API access.
        if status == http_client::StatusCode::BAD_REQUEST
            || status == http_client::StatusCode::UNAUTHORIZED
            || status == http_client::StatusCode::FORBIDDEN
        {
            return Err(RefreshError::Fatal(err));
        }
        return Err(RefreshError::Transient(err));
    }

    let tokens: TokenResponse =
        serde_json::from_str(&body).map_err(|e| RefreshError::Transient(e.into()))?;
    let access_token = tokens
        .access_token
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            RefreshError::Fatal(anyhow!(
                "xAI token refresh response was missing access_token"
            ))
        })?;

    let new_refresh = tokens
        .refresh_token
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| refresh_token.to_string());
    let expires_in = tokens.expires_in.unwrap_or(6 * 60 * 60);
    let email = tokens
        .id_token
        .as_deref()
        .and_then(extract_email_from_jwt);

    Ok(XaiOAuthCredentials {
        access_token,
        refresh_token: new_refresh,
        expires_at_ms: now_ms() + expires_in.saturating_mul(1000),
        email,
    })
}

fn extract_email_from_jwt(jwt: &str) -> Option<String> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    claims
        .get("email")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|err| {
            log::error!("System clock is before UNIX epoch: {err}");
            0
        })
}

fn do_sign_in(state: &Entity<State>, http_client: &Arc<dyn HttpClient>, cx: &mut App) {
    if state.read(cx).is_signing_in() {
        return;
    }

    let weak_state = state.downgrade();
    let http_client = http_client.clone();

    let task = cx.spawn(async move |cx| {
        match do_oauth_flow(http_client, weak_state.clone(), cx).await {
            Ok(creds) => {
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
                            .update(cx, |s, cx| {
                                s.credentials = Some(creds);
                                s.sign_in_task = None;
                                s.pending_user_code = None;
                                s.last_auth_error = None;
                                cx.notify();
                            })
                            .log_err();
                    }
                    Err(err) => {
                        log::error!(
                            "Grok subscription sign-in failed to persist credentials: {err:?}"
                        );
                        weak_state
                            .update(cx, |s, cx| {
                                s.sign_in_task = None;
                                s.pending_user_code = None;
                                s.last_auth_error =
                                    Some("Failed to save credentials. Please try again.".into());
                                cx.notify();
                            })
                            .log_err();
                    }
                }
            }
            Err(err) => {
                log::error!("Grok subscription sign-in failed: {err:?}");
                weak_state
                    .update(cx, |s, cx| {
                        s.sign_in_task = None;
                        s.pending_user_code = None;
                        s.last_auth_error = Some("Sign-in failed. Please try again.".into());
                        cx.notify();
                    })
                    .log_err();
            }
        }
        anyhow::Ok(())
    });

    state.update(cx, |s, cx| {
        s.last_auth_error = None;
        s.pending_user_code = None;
        s.sign_in_task = Some(task);
        cx.notify();
    });
}

fn do_sign_out(state: &gpui::WeakEntity<State>, cx: &mut App) -> Task<Result<()>> {
    let weak_state = state.clone();
    weak_state
        .update(cx, |s, cx| {
            s.auth_generation += 1;
            s.credentials = None;
            s.sign_in_task = None;
            s.refresh_task = None;
            s.pending_user_code = None;
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
            .context("Failed to delete Grok subscription credentials from keychain")?;
        anyhow::Ok(())
    })
}

struct ConfigurationView {
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    compact: bool,
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);

        if state.is_authenticated() {
            let label = state
                .email()
                .map(|e| format!("Signed in as {e}"))
                .unwrap_or_else(|| "Signed in to Grok".to_string());

            let weak_state = self.state.downgrade();

            return v_flex()
                .child(
                    ConfiguredApiCard::new("xai-subscribed-sign-out", SharedString::from(label))
                        .button_label("Sign Out")
                        .on_click(cx.listener(move |_this, _, _window, cx| {
                            do_sign_out(&weak_state, cx).detach_and_log_err(cx);
                        })),
                )
                .into_any_element();
        }

        let last_auth_error = state.last_auth_error.clone();
        let pending_user_code = state.pending_user_code.clone();
        let provider_state = self.state.clone();
        let http_client = self.http_client.clone();

        let is_signing_in = state.is_signing_in();
        let button_label = if is_signing_in {
            "Waiting for approval…"
        } else {
            "Sign In"
        };

        v_flex()
            .gap_2()
            .when(!self.compact, |this| {
                this.child(Label::new(SUBSCRIPTION_DESCRIPTION))
            })
            .child(
                Button::new("xai-subscribed-sign-in", button_label)
                    .when(!self.compact, |this| this.full_width())
                    .style(ButtonStyle::Outlined)
                    .size(ButtonSize::Medium)
                    .loading(is_signing_in)
                    .disabled(is_signing_in)
                    .on_click(move |_, _window, cx| {
                        do_sign_in(&provider_state, &http_client, cx);
                    }),
            )
            .when_some(pending_user_code, |this, code| {
                this.child(
                    Label::new(format!(
                        "If prompted, enter code {code} in your browser to approve access."
                    ))
                    .color(Color::Muted)
                    .size(LabelSize::Small),
                )
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscription_catalog_models() {
        let models = GrokSubscriptionModel::all();
        assert_eq!(
            models
                .iter()
                .map(|m| m.id())
                .collect::<Vec<_>>(),
            vec!["grok-build-0.1", "grok-4.5", "grok-composer-2.5"]
        );
    }

    #[test]
    fn composer_25_fast_mode_selects_fast_model_id() {
        let model = GrokSubscriptionModel::Composer25;
        assert!(model.supports_fast_mode());
        assert_eq!(model.request_model_id(None), "grok-composer-2.5");
        assert_eq!(
            model.request_model_id(Some(Speed::Standard)),
            "grok-composer-2.5"
        );
        assert_eq!(
            model.request_model_id(Some(Speed::Fast)),
            "grok-composer-2.5-fast"
        );
        assert!(!model.supports_images());
        assert!(!model.supports_reasoning_effort());
    }

    #[test]
    fn grok_45_supports_selectable_thinking_effort_levels() {
        let effort_levels = supported_thinking_effort_levels(&GrokSubscriptionModel::Grok45);
        let values = effort_levels
            .iter()
            .map(|level| level.value.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(values, ["low", "medium", "high"]);
    }

    #[test]
    fn token_response_parses_authorization_pending_without_access_token() {
        let body = r#"{"error":"authorization_pending","error_description":"The authorization request is still pending"}"#;
        let tokens: TokenResponse = serde_json::from_str(body).unwrap();
        assert_eq!(tokens.error.as_deref(), Some("authorization_pending"));
        assert!(tokens.access_token.is_none());
    }

    #[test]
    fn token_response_parses_successful_tokens() {
        let body = r#"{"access_token":"at","refresh_token":"rt","expires_in":3600,"token_type":"Bearer"}"#;
        let tokens: TokenResponse = serde_json::from_str(body).unwrap();
        assert_eq!(tokens.access_token.as_deref(), Some("at"));
        assert_eq!(tokens.refresh_token.as_deref(), Some("rt"));
        assert_eq!(tokens.expires_in, Some(3600));
    }
}
