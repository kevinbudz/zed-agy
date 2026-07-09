# zed-agy

This is a fork of Zed that adds subscription-backed LLM providers for Zed Agent threads.

## Providers

### Antigravity

Adds support for Antigravity as a provider. ***This kind of integration violates Google's TOS, and can risk you getting blacklisted from Antigravity***. That being said, there are safeguards in place, like matching/lowering context windows to be more in-line with what Antigravity provides.

In this case, Sonnet/Opus models give you 200k context, Flash/Pro gives you 400k, and GPT-OSS; 128k.

### Grok Subscription (SuperGrok / X Premium+)

Sign in with your SuperGrok or X Premium+ account via xAI's official device-code OAuth flow (same client used by OpenCode, Kilo, Hermes, and OpenClaw). No `XAI_API_KEY` is required.

1. Build and run this fork.
2. Open **Settings → AI → LLM Providers → Grok Subscription**.
3. Click **Sign In**, approve access in the browser, then start a **New Zed Agent** thread and pick a Grok model (defaults to **Grok Build**).

API-key xAI access remains available under the separate **xAI** provider.

Note: xAI may gate OAuth inference by subscription tier or quota. If sign-in works but requests return HTTP 403, use the API-key **xAI** provider instead (or check your plan at [x.ai/grok](https://x.ai/grok)).

---

### Installation

I do not intend on packaging binaries for this fork. Please refer to the 'Installing a development build.' in the documentation provided below.

- [Building Zed for macOS](./docs/src/development/macos.md)
- [Building Zed for Linux](./docs/src/development/linux.md)
- [Building Zed for Windows](./docs/src/development/windows.md)

### Planned
- [x] Add a matching icon for Antigravity.
- [x] Align context windows with Antigravity.
- [x] Add Grok Subscription (SuperGrok / X Premium+) OAuth provider.
- [ ] Add provider support for other services like Cursor. (not ACP)

---

### Credits

- [NoeFabris](https://github.com/NoeFabris) for their [opencode-antigravity-auth](https://github.com/NoeFabris/opencode-antigravity-auth) repository. A lot of logic is taken from that.
