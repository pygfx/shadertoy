# Changelog / release notes

This changelog documents major user facing changes, for all changes see the diff on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Possible sections in each release:

* Added: for new features.
* Changed: for changes in existing functionality.
* Deprecated: for soon-to-be removed features.
* Removed: for now removed features.
* Fixed: for any bug fixes.
* Security: in case of vulnerabilities.

### [unreleased]

### [v0.2.0] - 2025-06-06

Added:
* `iChannelResolution` uniform https://github.com/pygfx/shadertoy/pull/18
* "Common" tab support https://github.com/pygfx/shadertoy/pull/19
* Run shaders from the website API and CLI https://github.com/pygfx/shadertoy/pull/25
* `vflip` option for channel inputs https://github.com/pygfx/shadertoy/pull/26
* Attribute `.complete` to mark if shaders use unsupported features https://github.com/pygfx/shadertoy/pull/29
* Additional Uniforms are now supported in `.snapshot()` https://github.com/pygfx/shadertoy/pull/37
* Multipass shaders using `BufferRenderPass` and `ShadertoyChannelBuffer` https://github.com/pygfx/shadertoy/pull/43

Changed:
* `ShadertoyChannel` is now more specific `ShadertoyChannelTexture` https://github.com/pygfx/shadertoy/pull/40

Removed:
* Removed GLSL uniform aliases in favor of just Shadertoy syntax https://github.com/pygfx/shadertoy/pull/42

### [v0.1.0] - 2024-01-21

Fixed:
* Mouse events not releasing button down https://github.com/pygfx/shadertoy/pull/14


**Note**: For development history before this release, see [wgpu-py changelog up to v0.13.2](https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md#v0132---21-12-2023)