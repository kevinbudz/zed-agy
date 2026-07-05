use anyhow::{Context as _, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use collections::BTreeMap;
use credentials_provider::CredentialsProvider;
use futures::{
    AsyncBufReadExt as _, AsyncReadExt as _, FutureExt as _, StreamExt as _, future::BoxFuture,
    future::Shared, io::BufReader,
};
use gpui::{App, AsyncApp, Context, Entity, SharedString, Task, Window};
use http_client::{AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest};
use language_model::{
    AuthenticateError, IconOrSvg, InlineDescription, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelEffortLevel, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoice,
    LanguageModelToolSchemaFormat, ProviderSettingsView, RateLimiter,
};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use settings::Settings;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use ui::{ConfiguredApiCard, prelude::*};
use url::form_urlencoded;
use util::ResultExt as _;

use google_ai::completion::{GoogleEventMapper, into_google};
use google_ai::{GenerateContentResponse, GoogleModelMode, Part};

// Constants
pub const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("antigravity");
pub const PROVIDER_NAME: LanguageModelProviderName =
    LanguageModelProviderName::new("Google Antigravity");

const ANTIGRAVITY_CLIENT_ID_ENV_VAR: &str = "ZED_ANTIGRAVITY_CLIENT_ID";
const ANTIGRAVITY_CLIENT_SECRET_ENV_VAR: &str = "ZED_ANTIGRAVITY_CLIENT_SECRET";
const ANTIGRAVITY_DEFAULT_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
const ANTIGRAVITY_DEFAULT_CLIENT_SECRET: &str = "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";
const ANTIGRAVITY_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];

const ANTIGRAVITY_ENDPOINT: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const ANTIGRAVITY_DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";
const CREDENTIALS_KEY: &str = "antigravity-oauth";
const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";
const MIN_THOUGHT_SIGNATURE_LENGTH: usize = 50;

fn antigravity_client_id() -> String {
    std::env::var(ANTIGRAVITY_CLIENT_ID_ENV_VAR)
        .unwrap_or_else(|_| ANTIGRAVITY_DEFAULT_CLIENT_ID.to_string())
}

fn antigravity_client_secret() -> String {
    std::env::var(ANTIGRAVITY_CLIENT_SECRET_ENV_VAR)
        .unwrap_or_else(|_| ANTIGRAVITY_DEFAULT_CLIENT_SECRET.to_string())
}

pub use settings::AntigravityAvailableModel as AvailableModel;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct AntigravitySettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AntigravityCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at_ms: u64,
    pub project_id: String,
    pub email: Option<String>,
}

impl AntigravityCredentials {
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.expires_at_ms <= now + 60 * 1000 // 1 minute safety buffer
    }
}

pub struct AntigravityLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    credentials: Option<AntigravityCredentials>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    load_task: Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    refresh_task: Option<Shared<Task<Result<AntigravityCredentials, Arc<anyhow::Error>>>>>,
    sign_in_task: Option<Task<Result<()>>>,
    auth_generation: usize,
    last_auth_error: Option<SharedString>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.credentials.is_some()
    }
    fn is_signing_in(&self) -> bool {
        self.sign_in_task.is_some()
    }
    fn email(&self) -> Option<String> {
        self.credentials.as_ref().and_then(|c| c.email.clone())
    }
}

impl AntigravityLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|_cx| State {
            credentials: None,
            credentials_provider,
            load_task: None,
            refresh_task: None,
            sign_in_task: None,
            auth_generation: 0,
            last_auth_error: None,
        });

        let this = Self { http_client, state };
        this.load_credentials(cx);
        this
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
                state.update(&mut *cx, |s, cx| {
                    if let Ok(Some((_, bytes))) = result {
                        match serde_json::from_slice::<AntigravityCredentials>(&bytes) {
                            Ok(creds) => s.credentials = Some(creds),
                            Err(err) => {
                                log::warn!("Failed to deserialize Antigravity credentials: {err}");
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

    fn create_language_model(&self, model: AntigravityModel) -> Arc<dyn LanguageModel> {
        Arc::new(AntigravityLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn settings(cx: &App) -> &AntigravitySettings {
        &crate::AllLanguageModelSettings::get_global(cx).antigravity
    }
}

impl LanguageModelProviderState for AntigravityLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for AntigravityLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiGoogle)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(AntigravityModel::ClaudeOpus4_6))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(AntigravityModel::Gemini35Flash))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Standard predefined models
        for m in &[
            AntigravityModel::ClaudeOpus4_6,
            AntigravityModel::ClaudeSonnet4_6,
            AntigravityModel::Gemini31Pro,
            AntigravityModel::Gemini35Flash,
            AntigravityModel::GptOss120B,
        ] {
            models.insert(m.id().to_string(), m.clone());
        }

        // Custom models from settings
        for model in &Self::settings(cx).available_models {
            let mode = match model.mode.unwrap_or_default() {
                settings::ModelMode::Thinking { budget_tokens } => {
                    GoogleModelMode::Thinking { budget_tokens }
                }
                _ => GoogleModelMode::Default,
            };
            models.insert(
                model.name.clone(),
                AntigravityModel::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    mode,
                },
            );
        }

        models
            .into_values()
            .map(|model| {
                Arc::new(AntigravityLanguageModel {
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
                    Err(
                        anyhow!("Sign in with your Google account via OAuth to use this provider.")
                            .into(),
                    )
                }
            })
        } else {
            Task::ready(Err(anyhow!(
                "Sign in with your Google account via OAuth to use this provider."
            )
            .into()))
        }
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let title = if is_authenticated {
            None
        } else {
            Some("Configure Antigravity".into())
        };
        let description = if is_authenticated {
            None
        } else {
            Some(InlineDescription::Text(
                "Sign in with Google via OAuth to use Antigravity models in Zed.".into(),
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
            let label = state
                .email()
                .map(|e| format!("Signed in to Antigravity as {e}"))
                .unwrap_or_else(|| "Signed in to Antigravity".to_string());

            let weak_state = self.state.downgrade();

            return v_flex()
                .child(
                    ConfiguredApiCard::new("antigravity-sign-out", SharedString::from(label))
                        .button_label("Sign Out")
                        .on_click(cx.listener(move |_this, _, _window, cx| {
                            do_sign_out(&weak_state, cx).detach_and_log_err(cx);
                        })),
                )
                .into_any_element();
        }

        let is_signing_in = state.is_signing_in();
        let last_auth_error = state.last_auth_error.clone();
        v_flex()
            .gap_2()
            .child(
                Button::new(
                    "antigravity-sign-in",
                    if is_signing_in {
                        "Signing in…"
                    } else {
                        "Sign In"
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

fn do_sign_in(state: &Entity<State>, http_client: &Arc<dyn HttpClient>, cx: &mut App) {
    if state.read(cx).is_signing_in() {
        return;
    }

    let weak_state = state.downgrade();
    let http_client = http_client.clone();

    let task = cx.spawn(async move |cx| {
        match do_oauth_flow(http_client, &cx).await {
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
                            .update(&mut *cx, |s, cx| {
                                s.credentials = Some(creds);
                                s.sign_in_task = None;
                                s.last_auth_error = None;
                                cx.notify();
                            })
                            .log_err();
                    }
                    Err(err) => {
                        log::error!("Antigravity sign-in failed to persist credentials: {err:?}");
                        weak_state
                            .update(&mut *cx, |s, cx| {
                                s.sign_in_task = None;
                                s.last_auth_error = Some("Failed to save credentials".into());
                                cx.notify();
                            })
                            .log_err();
                    }
                }
            }
            Err(err) => {
                log::error!("Antigravity sign-in failed: {err:?}");
                weak_state
                    .update(&mut *cx, |s, cx| {
                        s.sign_in_task = None;
                        s.last_auth_error = Some(err.to_string().into());
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
            s.auth_generation += 1;
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

#[derive(Clone, Debug, PartialEq)]
pub enum AntigravityModel {
    ClaudeOpus4_6,
    ClaudeSonnet4_6,
    Gemini31Pro,
    Gemini35Flash,
    GptOss120B,

    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: u64,
        mode: GoogleModelMode,
    },
}

impl AntigravityModel {
    fn id(&self) -> &str {
        match self {
            Self::ClaudeOpus4_6 => "antigravity-claude-opus-4-6-thinking",
            Self::ClaudeSonnet4_6 => "antigravity-claude-sonnet-4-6",
            Self::Gemini31Pro => "antigravity-gemini-3.1-pro-high",
            Self::Gemini35Flash => "antigravity-gemini-3.5-flash-medium",
            Self::GptOss120B => "antigravity-gpt-oss-120b-medium",

            Self::Custom { name, .. } => name,
        }
    }

    fn request_id(&self) -> &str {
        match self {
            Self::ClaudeOpus4_6 => "claude-opus-4-6-thinking",
            Self::ClaudeSonnet4_6 => "claude-sonnet-4-6",
            Self::Gemini31Pro => "gemini-3.1-pro-low",
            Self::Gemini35Flash => "gemini-3-flash",
            Self::GptOss120B => "gpt-oss-120b-medium",

            Self::Custom { name, .. } => name,
        }
    }

    fn display_name(&self) -> &str {
        match self {
            Self::ClaudeOpus4_6 => "Claude Opus 4.6",
            Self::ClaudeSonnet4_6 => "Claude Sonnet 4.6",
            Self::Gemini31Pro => "Gemini 3.1 Pro",
            Self::Gemini35Flash => "Gemini 3.5 Flash",
            Self::GptOss120B => "GPT-OSS 120B",

            Self::Custom {
                display_name, name, ..
            } => display_name.as_deref().unwrap_or(name),
        }
    }

    fn mode(&self) -> GoogleModelMode {
        match self {
            Self::ClaudeOpus4_6 | Self::ClaudeSonnet4_6 | Self::GptOss120B => {
                GoogleModelMode::Default
            }
            Self::Gemini31Pro | Self::Gemini35Flash => GoogleModelMode::Thinking {
                budget_tokens: None,
            },

            Self::Custom { mode, .. } => *mode,
        }
    }

    fn max_token_count(&self) -> u64 {
        match self {
            Self::ClaudeOpus4_6 | Self::ClaudeSonnet4_6 => 200_000,
            Self::Gemini31Pro | Self::Gemini35Flash => 1_048_576,
            Self::GptOss120B => 131_072,

            Self::Custom { max_tokens, .. } => *max_tokens,
        }
    }
}

pub struct AntigravityLanguageModel {
    id: LanguageModelId,
    model: AntigravityModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AntigravityRequestWrapper {
    project: String,
    model: String,
    request_type: String,
    user_agent: String,
    request_id: String,
    request: google_ai::GenerateContentRequest,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AntigravityResponseWrapper {
    response: Option<GenerateContentResponse>,
}

fn fill_missing_candidate_roles(value: &mut serde_json::Value) {
    let candidates = if value.get("response").is_some() {
        value
            .get_mut("response")
            .and_then(|response| response.get_mut("candidates"))
    } else {
        value.get_mut("candidates")
    };

    let Some(candidates) = candidates.and_then(|candidates| candidates.as_array_mut()) else {
        return;
    };

    for candidate in candidates {
        if let Some(content) = candidate
            .get_mut("content")
            .and_then(|content| content.as_object_mut())
            && !content.contains_key("role")
        {
            content.insert(
                "role".to_string(),
                serde_json::Value::String("model".to_string()),
            );
        }
    }
}

fn parse_antigravity_sse_response(line: &str) -> Result<Option<GenerateContentResponse>> {
    let line = line.trim();
    if line.is_empty() || line == "[DONE]" {
        return Ok(None);
    }

    let mut value = serde_json::from_str::<serde_json::Value>(line)?;
    fill_missing_candidate_roles(&mut value);

    if value.get("response").is_some() {
        let wrapper = serde_json::from_value::<AntigravityResponseWrapper>(value)?;
        Ok(wrapper.response)
    } else {
        Ok(Some(serde_json::from_value::<GenerateContentResponse>(
            value,
        )?))
    }
}

fn sanitize_gemini31_function_call_signatures(request: &mut google_ai::GenerateContentRequest) {
    for content in &mut request.contents {
        let mut signature = content.parts.iter().find_map(|part| match part {
            Part::FunctionCallPart(function_call_part) => function_call_part
                .thought_signature
                .as_ref()
                .filter(|signature| {
                    signature.as_str() == SKIP_THOUGHT_SIGNATURE
                        || signature.len() >= MIN_THOUGHT_SIGNATURE_LENGTH
                })
                .cloned(),
            _ => None,
        });

        let mut seen_function_call = false;
        for part in &mut content.parts {
            if let Part::FunctionCallPart(function_call_part) = part {
                if seen_function_call {
                    // Gemini 3.1 rejects parallel function-call turns when the
                    // signature is duplicated across every functionCall part. The
                    // Antigravity gateway expects only the first functionCall in a
                    // content block to carry the thought signature.
                    function_call_part.thought_signature = None;
                    continue;
                }

                seen_function_call = true;
                let first_signature = function_call_part.thought_signature.as_ref();
                let first_signature_is_valid = first_signature
                    .map(|signature| {
                        signature.as_str() == SKIP_THOUGHT_SIGNATURE
                            || signature.len() >= MIN_THOUGHT_SIGNATURE_LENGTH
                    })
                    .unwrap_or(false);

                if first_signature_is_valid {
                    signature = function_call_part.thought_signature.clone();
                } else {
                    function_call_part.thought_signature = Some(
                        signature
                            .clone()
                            .unwrap_or_else(|| SKIP_THOUGHT_SIGNATURE.to_string()),
                    );
                }
            }
        }
    }
}

fn sanitize_claude_request(request: &mut google_ai::GenerateContentRequest) {
    for content in &mut request.contents {
        for part in &mut content.parts {
            match part {
                Part::TextPart(text_part) => {
                    text_part.thought_signature = None;
                }
                Part::FunctionCallPart(function_call_part) => {
                    function_call_part.thought_signature = None;
                }
                Part::FunctionResponsePart(_) | Part::InlineDataPart(_) => {}
            }
        }

        content
            .parts
            .retain(|part| !matches!(part, Part::TextPart(text_part) if text_part.thought));
    }

    request.contents.retain(|content| !content.parts.is_empty());
}

fn normalize_claude_thinking_config(value: &mut serde_json::Value) {
    let Some(thinking_config) = value
        .pointer_mut("/request/generationConfig/thinkingConfig")
        .and_then(|config| config.as_object_mut())
    else {
        return;
    };

    let include_thoughts = thinking_config
        .remove("includeThoughts")
        .unwrap_or(serde_json::Value::Bool(true));
    let thinking_budget = thinking_config.remove("thinkingBudget");
    thinking_config.remove("thinkingLevel");

    thinking_config.insert("include_thoughts".to_string(), include_thoughts);
    if let Some(thinking_budget) = thinking_budget
        && !thinking_budget.is_null()
    {
        thinking_config.insert("thinking_budget".to_string(), thinking_budget);
    }
}

fn normalize_gemini_thinking_config(value: &mut serde_json::Value) {
    let Some(thinking_config) = value
        .pointer_mut("/request/generationConfig/thinkingConfig")
        .and_then(|config| config.as_object_mut())
    else {
        return;
    };

    if let Some(serde_json::Value::String(thinking_level)) =
        thinking_config.get_mut("thinkingLevel")
    {
        *thinking_level = thinking_level.to_ascii_lowercase();
    }
}

fn normalize_antigravity_system_instruction(value: &mut serde_json::Value) {
    let Some(system_instruction) = value
        .pointer_mut("/request/systemInstruction")
        .and_then(|system_instruction| system_instruction.as_object_mut())
    else {
        return;
    };

    system_instruction
        .entry("role".to_string())
        .or_insert_with(|| serde_json::Value::String("user".to_string()));
}

fn is_unsupported_gemini_schema_field(key: &str) -> bool {
    matches!(
        key,
        "additionalProperties"
            | "$schema"
            | "$id"
            | "$comment"
            | "$ref"
            | "$defs"
            | "definitions"
            | "const"
            | "contentMediaType"
            | "contentEncoding"
            | "if"
            | "then"
            | "else"
            | "not"
            | "patternProperties"
            | "unevaluatedProperties"
            | "unevaluatedItems"
            | "dependentRequired"
            | "dependentSchemas"
            | "propertyNames"
            | "minContains"
            | "maxContains"
            | "default"
            | "examples"
            | "format"
            | "pattern"
            | "minLength"
            | "maxLength"
            | "minimum"
            | "maximum"
            | "exclusiveMinimum"
            | "exclusiveMaximum"
            | "minItems"
            | "maxItems"
            | "nullable"
            | "readOnly"
            | "writeOnly"
            | "deprecated"
            | "discriminator"
            | "xml"
            | "externalDocs"
    )
}

fn normalize_gemini_schema(schema: &mut serde_json::Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };

    object.retain(|key, _| !is_unsupported_gemini_schema_field(key));

    if let Some(schema_type) = object.get_mut("type") {
        match schema_type {
            serde_json::Value::String(schema_type) => {
                *schema_type = schema_type.to_ascii_uppercase();
            }
            serde_json::Value::Array(schema_types) => {
                let first_non_null = schema_types
                    .iter()
                    .filter_map(|schema_type| schema_type.as_str())
                    .find(|schema_type| !schema_type.eq_ignore_ascii_case("null"))
                    .unwrap_or("string")
                    .to_ascii_uppercase();
                *schema_type = serde_json::Value::String(first_non_null);
            }
            _ => {}
        }
    }

    if object.get("type").and_then(|value| value.as_str()) == Some("ARRAY")
        && !object.contains_key("items")
    {
        object.insert("items".to_string(), serde_json::json!({ "type": "STRING" }));
    }

    let property_names = if let Some(properties) = object
        .get_mut("properties")
        .and_then(|properties| properties.as_object_mut())
    {
        for property in properties.values_mut() {
            normalize_gemini_schema(property);
        }
        properties.keys().cloned().collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    if !property_names.is_empty() {
        if let Some(required) = object
            .get_mut("required")
            .and_then(|required| required.as_array_mut())
        {
            required.retain(|required_property| {
                required_property
                    .as_str()
                    .map(|required_property| {
                        property_names
                            .iter()
                            .any(|property_name| property_name == required_property)
                    })
                    .unwrap_or(false)
            });
            if required.is_empty() {
                object.remove("required");
            }
        }
    }

    for key in ["items", "anyOf", "oneOf", "allOf"] {
        if let Some(value) = object.get_mut(key) {
            match value {
                serde_json::Value::Array(values) => {
                    for value in values {
                        normalize_gemini_schema(value);
                    }
                }
                serde_json::Value::Object(_) => normalize_gemini_schema(value),
                _ => {}
            }
        }
    }
}

fn normalize_antigravity_tool_schemas(value: &mut serde_json::Value, use_gemini_schema: bool) {
    let Some(tools) = value
        .pointer_mut("/request/tools")
        .and_then(|tools| tools.as_array_mut())
    else {
        return;
    };

    for tool in tools {
        let Some(function_declarations) = tool
            .get_mut("functionDeclarations")
            .and_then(|function_declarations| function_declarations.as_array_mut())
        else {
            continue;
        };

        for function_declaration in function_declarations {
            let Some(parameters) = function_declaration.get_mut("parameters") else {
                continue;
            };

            if use_gemini_schema {
                normalize_gemini_schema(parameters);
            }
        }
    }
}

fn normalize_antigravity_tool_config(value: &mut serde_json::Value) {
    let Some(mode) = value
        .pointer_mut("/request/toolConfig/functionCallingConfig/mode")
        .and_then(|mode| mode.as_str())
        .map(str::to_ascii_uppercase)
    else {
        return;
    };

    if let Some(mode_value) = value.pointer_mut("/request/toolConfig/functionCallingConfig/mode") {
        *mode_value = serde_json::Value::String(mode);
    }
}

fn remove_empty_antigravity_fields(value: &mut serde_json::Value) {
    if value
        .pointer("/request/generationConfig/stopSequences")
        .and_then(|stop_sequences| stop_sequences.as_array())
        .is_some_and(|stop_sequences| stop_sequences.is_empty())
    {
        if let Some(generation_config) = value
            .pointer_mut("/request/generationConfig")
            .and_then(|generation_config| generation_config.as_object_mut())
        {
            generation_config.remove("stopSequences");
        }
    }

    if let Some(tools) = value
        .pointer_mut("/request/tools")
        .and_then(|tools| tools.as_array_mut())
    {
        tools.retain(|tool| {
            tool.get("functionDeclarations")
                .and_then(|function_declarations| function_declarations.as_array())
                .map(|function_declarations| !function_declarations.is_empty())
                .unwrap_or(true)
        });
    }

    if value
        .pointer("/request/tools")
        .and_then(|tools| tools.as_array())
        .is_some_and(|tools| tools.is_empty())
    {
        if let Some(request) = value
            .pointer_mut("/request")
            .and_then(|request| request.as_object_mut())
        {
            request.remove("tools");
        }
    }
}

fn redacted_request_for_log(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => serde_json::Value::Object(
            object
                .iter()
                .map(|(key, value)| {
                    let value = match key.as_str() {
                        "text" | "data" | "args" | "response" => {
                            serde_json::Value::String("<redacted>".to_string())
                        }
                        _ => redacted_request_for_log(value),
                    };
                    (key.clone(), value)
                })
                .collect(),
        ),
        serde_json::Value::Array(values) => serde_json::Value::Array(
            values
                .iter()
                .map(redacted_request_for_log)
                .collect::<Vec<_>>(),
        ),
        value => value.clone(),
    }
}

fn antigravity_request_body(
    wrapper: &AntigravityRequestWrapper,
    use_claude_thinking_config: bool,
    use_gemini_thinking_config: bool,
) -> Result<String> {
    let mut value = serde_json::to_value(wrapper)?;
    normalize_antigravity_system_instruction(&mut value);
    normalize_antigravity_tool_config(&mut value);
    normalize_antigravity_tool_schemas(&mut value, use_gemini_thinking_config);
    remove_empty_antigravity_fields(&mut value);
    if use_claude_thinking_config {
        normalize_claude_thinking_config(&mut value);
    }
    if use_gemini_thinking_config {
        normalize_gemini_thinking_config(&mut value);
    }
    Ok(serde_json::to_string(&value)?)
}

impl AntigravityLanguageModel {
    fn stream_completion(
        &self,
        mut request: google_ai::GenerateContentRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<GenerateContentResponse>>>,
    > {
        let http_client = self.http_client.clone();
        let state = self.state.downgrade();
        let model_id = self.model.request_id().to_string();
        let model_full_id = self.model.id().to_string();
        let is_claude = matches!(
            self.model,
            AntigravityModel::ClaudeOpus4_6 | AntigravityModel::ClaudeSonnet4_6
        );
        let use_claude_thinking_config = matches!(self.model, AntigravityModel::ClaudeOpus4_6);

        // The Antigravity wrapper carries the model at the top level. The
        // working opencode integration removes the nested request.model field;
        // leaving it present can confuse routing for some gateway models.
        request.model.model_id.clear();

        // Antigravity rejects Claude thinking requests when maxOutputTokens is not
        // greater than the thinking budget. Zed's Google request builder leaves this
        // unset, so set the same 64k ceiling used by the working opencode plugin.
        if use_claude_thinking_config {
            let config =
                request
                    .generation_config
                    .get_or_insert_with(|| google_ai::GenerationConfig {
                        candidate_count: Some(1),
                        stop_sequences: None,
                        max_output_tokens: None,
                        temperature: None,
                        top_p: None,
                        top_k: None,
                        thinking_config: None,
                    });
            config.max_output_tokens = Some(64_000);
            if config.thinking_config.is_none() {
                config.thinking_config = Some(google_ai::ThinkingConfig {
                    thinking_budget: Some(32_768),
                    thinking_level: None,
                    include_thoughts: Some(true),
                });
            }
        }

        if is_claude {
            sanitize_claude_request(&mut request);
        } else if model_id.starts_with("gemini-3.1-pro") {
            sanitize_gemini31_function_call_signatures(&mut request);
        }

        // Antigravity Gemini 3 uses thinkingLevel, not thinkingBudget. The shared
        // Google request builder only knows about numeric budgets, so normalize the
        // generated request here before wrapping it.
        if model_full_id.contains("gemini-3") {
            if let Some(config) = request.generation_config.as_mut() {
                if let Some(tc) = config.thinking_config.as_mut() {
                    tc.thinking_budget = None;
                    if tc.thinking_level.is_none() {
                        tc.thinking_level = match model_full_id.as_str() {
                            "antigravity-gemini-3.1-pro-high" => {
                                Some(google_ai::ThinkingLevel::High)
                            }
                            "antigravity-gemini-3.5-flash-medium" => {
                                Some(google_ai::ThinkingLevel::Medium)
                            }
                            _ => Some(google_ai::ThinkingLevel::Low),
                        };
                    }
                }
            }
        }

        let future = cx.spawn(async move |cx| {
            let creds = get_fresh_credentials(&state, &http_client, cx).await?;
            let request_id = format!("search-{}-{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(), rand::rng().next_u64());

            let wrapper = AntigravityRequestWrapper {
                project: creds.project_id.clone(),
                model: model_id.clone(),
                request_type: "agent".to_string(),
                user_agent: "antigravity".to_string(),
                request_id,
                request,
            };

            let uri = format!("{}/v1internal:streamGenerateContent?alt=sse", ANTIGRAVITY_ENDPOINT);
            let request_body = antigravity_request_body(
                &wrapper,
                use_claude_thinking_config,
                model_id.starts_with("gemini-3"),
            )
            .map_err(|e| anyhow!(e))?;
            let request_body_for_log = request_body.clone();

            let request = HttpRequest::builder()
                .method(Method::POST)
                .uri(uri)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("Authorization", &format!("Bearer {}", creds.access_token))
                .header("User-Agent", "antigravity/1.18.3 darwin/arm64")
                .header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
                .header("Client-Metadata", "{\"ideType\":\"ANTIGRAVITY\",\"platform\":\"MACOS\",\"pluginType\":\"GEMINI\"}")
                .body(AsyncBody::from(request_body))
                .map_err(|e| anyhow!(e))?;

            let mut response = http_client.send(request).await?;

            if response.status().is_success() {
                let reader = BufReader::new(response.into_body());
                Ok(reader
                    .lines()
                    .filter_map(|line| async move {
                        match line {
                            Ok(line) => {
                                if let Some(line) = line.strip_prefix("data: ") {
                                    match parse_antigravity_sse_response(line) {
                                        Ok(Some(response)) => Some(Ok(response)),
                                        Ok(None) => None,
                                        Err(error) => Some(Err(anyhow!(format!(
                                            "Error parsing JSON: {error:?}\n{line:?}"
                                        )))),
                                    }
                                } else {
                                    None
                                }
                            }
                            Err(error) => Some(Err(anyhow!(error))),
                        }
                    })
                    .boxed())
            } else {
                let mut text = String::new();
                response.body_mut().read_to_string(&mut text).await?;
                if model_id.starts_with("gemini-3.1-pro") {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&request_body_for_log) {
                        log::warn!(
                            "Antigravity Gemini 3.1 request rejected. Redacted request body: {}",
                            redacted_request_for_log(&value)
                        );
                    }
                }
                Err(anyhow!(
                    "error during streamGenerateContent, status code: {:?}, body: {}",
                    response.status(),
                    text
                ))
            }
        });

        future.boxed()
    }
}

impl LanguageModel for AntigravityLanguageModel {
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
        matches!(
            self.model,
            AntigravityModel::Gemini31Pro | AntigravityModel::Gemini35Flash
        )
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        match self.model {
            AntigravityModel::Gemini31Pro => vec![
                LanguageModelEffortLevel {
                    name: "Low".into(),
                    value: "low".into(),
                    is_default: false,
                },
                LanguageModelEffortLevel {
                    name: "High".into(),
                    value: "high".into(),
                    is_default: true,
                },
            ],
            AntigravityModel::Gemini35Flash => vec![
                LanguageModelEffortLevel {
                    name: "Low".into(),
                    value: "low".into(),
                    is_default: false,
                },
                LanguageModelEffortLevel {
                    name: "Medium".into(),
                    value: "medium".into(),
                    is_default: true,
                },
                LanguageModelEffortLevel {
                    name: "High".into(),
                    value: "high".into(),
                    is_default: false,
                },
            ],
            _ => Vec::new(),
        }
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        LanguageModelToolSchemaFormat::JsonSchemaSubset
    }

    fn telemetry_id(&self) -> String {
        format!("antigravity/{}", self.model.request_id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        match &self.model {
            AntigravityModel::ClaudeOpus4_6 => Some(64000),
            AntigravityModel::ClaudeSonnet4_6 => Some(8192),
            AntigravityModel::Gemini31Pro | AntigravityModel::Gemini35Flash => Some(20480),
            AntigravityModel::GptOss120B => Some(4096),
            _ => Some(4096),
        }
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
        let request = into_google(
            request,
            self.model.request_id().to_string(),
            self.model.mode(),
        );
        let request = self.stream_completion(request, cx);
        let future = self.request_limiter.stream(async move {
            let response = request.await.map_err(LanguageModelCompletionError::from)?;
            Ok(GoogleEventMapper::new().map_stream(response))
        });
        async move {
            let stream = future.await?;
            Ok(stream.boxed())
        }
        .boxed()
    }
}

async fn get_fresh_credentials(
    state: &gpui::WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    cx: &mut AsyncApp,
) -> Result<AntigravityCredentials, LanguageModelCompletionError> {
    let (creds, existing_task) = state
        .read_with(cx, |s, _| (s.credentials.clone(), s.refresh_task.clone()))
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
    let prev_email = creds.email.clone();

    let generation = state
        .read_with(cx, |s, _| s.auth_generation)
        .map_err(LanguageModelCompletionError::Other)?;

    let shared_task = cx
        .spawn(async move |cx| {
            let result = refresh_token(&http_client_clone, &refresh_token_value).await;

            match result {
                Ok(mut refreshed) => {
                    refreshed.email = prev_email;
                    let persist_result: Result<AntigravityCredentials, Arc<anyhow::Error>> =
                        async {
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
                                .update(&mut *cx, |s, _| {
                                    s.credentials = Some(refreshed.clone());
                                    s.refresh_task = None;
                                })
                                .map_err(|e| Arc::new(e))?;

                            Ok(refreshed)
                        }
                        .await;

                    if persist_result.is_err() {
                        let _ = state_clone.update(&mut *cx, |s, _| {
                            s.refresh_task = None;
                        });
                    }

                    persist_result
                }
                Err(RefreshError::Fatal(e)) => {
                    log::error!("Antigravity token refresh failed fatally: {e:?}");
                    let _ = state_clone.update(&mut *cx, |s, cx| {
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
                    let _ = state_clone.update(&mut *cx, |s, _| {
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

#[derive(Debug)]
pub enum RefreshError {
    Fatal(anyhow::Error),
    Transient(anyhow::Error),
}

async fn refresh_token(
    client: &Arc<dyn HttpClient>,
    refresh_token: &str,
) -> Result<AntigravityCredentials, RefreshError> {
    let client_id = antigravity_client_id();
    let client_secret = antigravity_client_secret();

    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("refresh_token", refresh_token)
        .append_pair("client_id", &client_id)
        .append_pair("client_secret", &client_secret)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri("https://oauth2.googleapis.com/token")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))
        .map_err(|e| RefreshError::Fatal(e.into()))?;

    let mut response = client
        .send(request)
        .await
        .map_err(RefreshError::Transient)?;

    let mut body_str = String::new();
    response
        .body_mut()
        .read_to_string(&mut body_str)
        .await
        .map_err(|e| RefreshError::Transient(e.into()))?;

    if !response.status().is_success() {
        let is_fatal = response.status() == 400 || response.status() == 401;
        let err = anyhow!(
            "Failed to refresh token: status: {}, body: {}",
            response.status(),
            body_str
        );
        return Err(if is_fatal {
            RefreshError::Fatal(err)
        } else {
            RefreshError::Transient(err)
        });
    }

    #[derive(Deserialize)]
    struct TokenResponse {
        access_token: String,
        expires_in: u64,
        refresh_token: Option<String>,
    }

    let payload: TokenResponse = serde_json::from_str(&body_str).map_err(|e| {
        RefreshError::Fatal(anyhow!("Failed to parse token refresh response: {e:?}"))
    })?;

    let new_refresh_token = payload
        .refresh_token
        .unwrap_or_else(|| refresh_token.to_string());

    let project_id = fetch_project_id(client, &payload.access_token)
        .await
        .unwrap_or_else(|_| ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string());

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    Ok(AntigravityCredentials {
        access_token: payload.access_token,
        refresh_token: new_refresh_token,
        expires_at_ms: now + payload.expires_in * 1000,
        project_id,
        email: None,
    })
}

async fn fetch_project_id(client: &Arc<dyn HttpClient>, access_token: &str) -> Result<String> {
    let endpoints = [
        "https://cloudcode-pa.googleapis.com",
        "https://daily-cloudcode-pa.sandbox.googleapis.com",
        "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    ];

    #[derive(Serialize)]
    struct LoadCodeAssistRequest {
        metadata: LoadCodeAssistMetadata,
    }

    #[derive(Serialize)]
    struct LoadCodeAssistMetadata {
        #[serde(rename = "ideType")]
        ide_type: &'static str,
        platform: &'static str,
        #[serde(rename = "pluginType")]
        plugin_type: &'static str,
    }

    let body = serde_json::to_string(&LoadCodeAssistRequest {
        metadata: LoadCodeAssistMetadata {
            ide_type: "ANTIGRAVITY",
            platform: "MACOS",
            plugin_type: "GEMINI",
        },
    })?;

    let mut last_err = anyhow!("No endpoints to query");
    for endpoint in endpoints {
        let uri = format!("{}/v1internal:loadCodeAssist", endpoint);
        let request = HttpRequest::builder()
            .method(Method::POST)
            .uri(&uri)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", access_token))
            .header("User-Agent", "google-api-nodejs-client/9.15.1")
            .header(
                "X-Goog-Api-Client",
                "google-cloud-sdk vscode_cloudshelleditor/0.1",
            )
            .header(
                "Client-Metadata",
                "{\"ideType\":\"ANTIGRAVITY\",\"platform\":\"MACOS\",\"pluginType\":\"GEMINI\"}",
            )
            .body(AsyncBody::from(body.clone()))?;

        let mut response = match client.send(request).await {
            Ok(resp) => resp,
            Err(e) => {
                last_err = e;
                continue;
            }
        };

        if response.status().is_success() {
            let mut body_str = String::new();
            if response
                .body_mut()
                .read_to_string(&mut body_str)
                .await
                .is_ok()
            {
                #[derive(Deserialize)]
                #[serde(rename_all = "camelCase")]
                struct LoadCodeAssistResponse {
                    cloudaicompanion_project: Option<CloudAiCompanionProject>,
                    project: Option<String>,
                }

                #[derive(Deserialize)]
                #[serde(untagged)]
                enum CloudAiCompanionProject {
                    Id(String),
                    Object { id: Option<String> },
                }

                if let Ok(res) = serde_json::from_str::<LoadCodeAssistResponse>(&body_str) {
                    if let Some(project) = res.cloudaicompanion_project {
                        let project = match project {
                            CloudAiCompanionProject::Id(id) => Some(id),
                            CloudAiCompanionProject::Object { id } => id,
                        };
                        if let Some(project) = project.filter(|project| !project.is_empty()) {
                            return Ok(project);
                        }
                    }
                    if let Some(project) = res.project.filter(|project| !project.is_empty()) {
                        return Ok(project);
                    }
                }
            }
        }
    }
    Err(last_err)
}

async fn do_oauth_flow(
    http_client: Arc<dyn HttpClient>,
    cx: &AsyncApp,
) -> Result<AntigravityCredentials> {
    let (redirect_uri, callback_rx) =
        oauth_callback_server::start_oauth_callback_server_with_config(
            oauth_callback_server::OAuthCallbackServerConfig {
                host: "localhost",
                preferred_port: 51121,
                fallback_port: None,
                path: "/oauth-callback",
            },
        )
        .context("Failed to start OAuth callback server")?;

    let mut verifier_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize().as_slice());

    #[derive(Serialize, Deserialize)]
    struct AuthState {
        verifier: String,
        project_id: String,
    }

    let state_json = serde_json::to_string(&AuthState {
        verifier: verifier.clone(),
        project_id: "".to_string(),
    })?;
    let state_encoded = URL_SAFE_NO_PAD.encode(state_json.as_bytes());

    let mut auth_url =
        url::Url::parse("https://accounts.google.com/o/oauth2/v2/auth").expect("valid URL");
    let client_id = antigravity_client_id();
    let client_secret = antigravity_client_secret();

    auth_url
        .query_pairs_mut()
        .append_pair("client_id", &client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", &redirect_uri)
        .append_pair("scope", &ANTIGRAVITY_SCOPES.join(" "))
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", &state_encoded)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");

    let _ = cx.update(|cx| cx.open_url(auth_url.as_str()));

    let callback = callback_rx
        .await
        .map_err(|_| anyhow!("OAuth callback was cancelled"))?
        .context("OAuth callback failed")?;

    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "authorization_code")
        .append_pair("client_id", &client_id)
        .append_pair("client_secret", &client_secret)
        .append_pair("code", &callback.code)
        .append_pair("redirect_uri", &redirect_uri)
        .append_pair("code_verifier", &verifier)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri("https://oauth2.googleapis.com/token")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?;

    let mut response = http_client.send(request).await?;
    let mut body_str = String::new();
    response.body_mut().read_to_string(&mut body_str).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to exchange authorization code: status: {}, body: {}",
            response.status(),
            body_str
        ));
    }

    #[derive(Deserialize)]
    struct GoogleTokenResponse {
        access_token: String,
        expires_in: u64,
        refresh_token: String,
    }

    let token_payload: GoogleTokenResponse = serde_json::from_str(&body_str)?;

    let user_info_request = HttpRequest::builder()
        .method(Method::GET)
        .uri("https://www.googleapis.com/oauth2/v1/userinfo?alt=json")
        .header(
            "Authorization",
            &format!("Bearer {}", token_payload.access_token),
        )
        .body(AsyncBody::empty())?;

    let mut user_info_resp = http_client.send(user_info_request).await?;
    let mut user_info_str = String::new();
    user_info_resp
        .body_mut()
        .read_to_string(&mut user_info_str)
        .await?;

    let email = if user_info_resp.status().is_success() {
        #[derive(Deserialize)]
        struct UserInfo {
            email: Option<String>,
        }
        serde_json::from_str::<UserInfo>(&user_info_str)
            .ok()
            .and_then(|ui| ui.email)
    } else {
        None
    };

    let project_id = fetch_project_id(&http_client, &token_payload.access_token)
        .await
        .unwrap_or_else(|_| ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string());

    Ok(AntigravityCredentials {
        access_token: token_payload.access_token,
        refresh_token: token_payload.refresh_token,
        expires_at_ms: start_time + token_payload.expires_in * 1000,
        project_id,
        email,
    })
}
