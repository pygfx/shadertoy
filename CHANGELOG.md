# Changelog / release notes

WebGPU and wgpu-native are still changing fast, and with that we do to. We do
not yet attempt to make things backwards compatible. Instead we try to
be precise about tracking changes to the public API.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Possible sections in each release:

* Added: for new features.
* Changed: for changes in existing functionality.
* Deprecated: for soon-to-be removed features.
* Removed: for now removed features.
* Fixed: for any bug fixes.
* Security: in case of vulnerabilities.


### [Unreleased]

Added:
* Run shaders from the website API https://github.com/pygfx/shadertoy/pull/25
* Additional Uniforms are now supported in `.screenshot()` https://github.com/pygfx/shadertoy/pull/37

### [v0.1.0] - 2024-01-21

Fixed:
* Mouse events not releasing button down https://github.com/pygfx/shadertoy/pull/14


**Note**: For development history before this release, see [wgpu-py changelog up to v0.13.2](https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md#v0132---21-12-2023)