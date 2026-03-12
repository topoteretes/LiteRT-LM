# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteRT-LM is a C++ library (C++20) for efficiently running language models across edge platforms (Android, Linux, macOS, Windows, Embedded Linux). It builds on top of Google's LiteRT runtime. The project is in Early Preview (v0.8.0) and is not yet accepting external OSS contributions.

## Build System

**Primary build tool: Bazel 7.6.1** (use Bazelisk for automatic version management).

### Common Build Commands

```bash
# Build the main demo binary (Linux x86_64)
bazel build --config=linux_x86_64 //runtime/engine:litert_lm_main

# Build for Android ARM64
bazel build --config=android_arm64 //runtime/engine:litert_lm_main

# Build for Android x86_64 (added March 2026)
bazel build --config=android_x86_64 //runtime/engine:litert_lm_main

# Build for macOS ARM64
bazel build --config=darwin_arm64 //runtime/engine:litert_lm_main

# Build for Windows
bazelisk build --config=windows //runtime/engine:litert_lm_main

# GPU support requires additional flags:
#   --define=litert_link_capi_so=true --define=resolve_symbols_in_exec=false
# Plus copying prebuilt shared libs from prebuilt/<platform>/ next to the binary.
```

### Running Tests

```bash
# Run all tests (Linux)
bazel test --config=linux_x86_64 //...

# Run a specific test
bazel test --config=linux_x86_64 //runtime/components:tokenizer_hf_test

# Tests are under runtime/components/ (39+ test files covering tokenizers,
# preprocessors, samplers, constrained decoding, etc.)
```

### CMake (Alternative)

```bash
cmake -B build && cmake --build build
```

### Rust Dependencies (managed via Bazel)

Rust code is compiled through Bazel, not `cargo build` directly. To update the Cargo.lock:
```bash
touch src/lib.rs
CARGO_BAZEL_REPIN=1 bazel sync --only=crate_index
rm src/lib.rs
```

## Architecture

### Core APIs (in order of abstraction level)

1. **Conversation** (`runtime/conversation/`) — Highest-level stateful API. Recommended for most users. Handles chat turns, prompt templating, and streaming.
2. **Engine** (`runtime/engine/`) — Heavyweight singleton that loads `.litertlm` models and manages resources. Entry point: `litert_lm_main.cc`.
3. **Session** (`runtime/core/`) — Lower-level API for fine-grained control over prefill/decode phases. Implementations: basic, advanced, legacy engines.

### Key Components (`runtime/components/`)

- **Tokenizers**: HuggingFace (`tokenizers` Rust crate) and SentencePiece implementations
- **Preprocessors**: Image, audio, and video input processing for multimodal models
- **Constrained decoding**: LLGuidance integration for grammar-guided generation
- **Samplers/Scoring**: Token sampling strategies and scoring utilities
- **LoRA**: Low-rank adaptation support
- **Prompt templating**: Jinja2 via `minijinja` Rust crate

### Other Key Directories

- `runtime/executor/` — Model execution backends (CPU, GPU, NPU)
- `runtime/framework/` — Threading and resource management
- `runtime/proto/` — Protocol buffer definitions
- `runtime/util/` — File utilities, logging, zip handling
- `schema/` — `.litertlm` file format (FlatBuffers schema in `core/`, Python packaging tools in `py/`)
- `c/` — C API bindings
- `python/` — Python API bindings (`litert_lm.cc` C extension + `interfaces.py` abstract interfaces)
- `kotlin/` — Kotlin API and JVM bindings
- `android/` — Android-specific code
- `prebuilt/` — Platform-specific prebuilt shared libraries (stored in Git LFS); includes `android_x86_64/` as of upstream March 2026
- `tools/test/` — End-to-end sanity check tests and utilities

### Model Format

Models use the `.litertlm` format (based on `.task` format with FlatBuffers headers). Schema defined in `schema/core/litertlm_header_schema.fbs`.

## Prerequisites

- **Bazel 7.6.1** (via Bazelisk)
- **Git LFS** — Required for prebuilt GPU binaries (`git lfs install && git lfs pull`)
- **Android builds**: NDK r28b+
- **Windows builds**: Visual Studio 2022, Python 3.13, Java (for JAVA_HOME)
