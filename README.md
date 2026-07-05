# zed-agy

This is a fork of Zed that allows you to add support for Antigravity as a provider. ***This kind of integration violates Google's TOS, and can risk you getting blacklisted from Antigravity***. That being said, there are safeguards in place, like matching/lowering context windows to be more in-line with what Antigravity provides.

In this case, Sonnet/Opus models give you 200k context, Flash/Pro gives you 400k, and GPT-OSS; 128k.

---

### Installation

I do not intend on packaging binaries for this fork. Please refer to the 'Installing a development build.' in the documentation provided below.

- [Building Zed for macOS](./docs/src/development/macos.md)
- [Building Zed for Linux](./docs/src/development/linux.md)
- [Building Zed for Windows](./docs/src/development/windows.md)

### Planned
- [x] Add a matching icon for Antigravity.
- [x] Align context windows with Antigravity.
- [ ] Add provider support for other services like Cursor. (not ACP)

---

### Credits

- [NoeFabris](https://github.com/NoeFabris) for their [opencode-antigravity-auth](https://github.com/NoeFabris/opencode-antigravity-auth) repository. A lot of logic is taken from that.
