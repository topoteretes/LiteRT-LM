# LiteRT-LM Custom Changes — Implementation Plan

This document records all custom changes made on top of the upstream
`google-ai-edge/LiteRT-LM` repository (branched at commit
`811fd479d807aeb4ca8b14ec75114f5fdd013993`). When rebasing or cherry-picking
onto a newer upstream revision, use this plan to re-implement each change.

---

## Implementation Status Summary

| # | Change | Status | Notes |
|---|--------|--------|-------|
| 1 | Knowledge Graph Extraction Demo | ✅ Done | `examples/simple_chat/` created |
| 2 | Speculative GPU Constrained Decoding | ✅ Done | `--define=async_constraint_masking=true` enables async path |
| 3a | NPU Constrained Decoding | ✅ Done | `ApplyConstrainedGreedySampling` added; `Decode()` no longer hard-fails |
| 3b | NPU Dynamic Prefill Signature | ✅ Done | `kPrefillSignature`/`kPrefillSize` replaced with dynamic lookup |
| 4 | Soft Degradation for Non-SP Tokenizers | ✅ Done | Both data processors log warning and skip instead of hard-failing |
| 5a | Optional LlmMetadata | ✅ Done | `GetLlmMetadata()` uses `has_value()` guard |
| 5b | Prefill Signature Fallback Input Names | ✅ Done | Fallback input names in `litert_compiled_model_executor_utils.cc` |
| 5c | `.gitignore` additions | ✅ Done | `/build/` in `.gitignore`; `/target/` not needed (cmake uses `build/`) |

All custom changes are preserved in `git stash@{0}` ("all custom changes backup"). Use `git stash show -p stash@{0}` to inspect the full diff.

---

## Change 1: Knowledge Graph Extraction Demo (`examples/simple_chat/`)

### What it does
A new standalone binary (`simple_chat`) that demonstrates LiteRT-LM end-to-end
with structured JSON output via constrained decoding. Despite the name, it is
primarily a **knowledge graph extraction** tool: given a text document, the LLM
is prompted to extract entities (nodes) and relationships (edges) and output
them as a JSON object conforming to a hardcoded schema.

Supports `--backend=cpu|gpu|npu`, `--litert_dispatch_lib_dir` (for QNN/NPU
dispatch shared libraries), `--input_text` or `--input_text_file`, and
`--add_constraint` (toggle constrained decoding).

The BUILD file includes Android-specific link options (`-lEGL`, `-lGLESv3`) and
`--export-dynamic-symbol=LiteRt*` for dynamic dispatch — designed to run on
Android NPU.

### Status

**✅ Done.** Both `examples/simple_chat/simple_chat.cc` and `examples/simple_chat/BUILD` have been created.

### Files to create (new)
- `examples/simple_chat/simple_chat.cc`
- `examples/simple_chat/BUILD`

### Implementation plan

#### `examples/simple_chat/BUILD`
Create a `cc_binary` target named `simple_chat` with:
- `srcs = ["simple_chat.cc"]`
- `linkopts` using `select()`:
  - `@litert//litert:litert_link_capi_so` → empty
  - `@platforms//os:ios` / `@platforms//os:macos` → `["-Wl,-exported_symbol,_LiteRt*"]`
  - `@platforms//os:windows` → empty
  - `@platforms//os:linux` → `["-Wl,--export-dynamic-symbol=LiteRt*"]`
  - `//conditions:default` (Android etc.) → `["-Wl,--export-dynamic-symbol=LiteRt*"]`
- Additional Android linkopts: `-lEGL -lGLESv3`
- `deps`:
  - `//runtime/components/constrained_decoding:constraint_provider_config`
  - `//runtime/components/constrained_decoding:llg_constraint_config`
  - `//runtime/conversation`
  - `//runtime/conversation:io_types`
  - `//runtime/engine:engine_factory`
  - `//runtime/engine:engine_impl_selected`
  - `//runtime/engine:engine_interface`
  - `//runtime/engine:engine_settings`
  - `//runtime/engine:io_types`
  - `//runtime/executor:executor_settings_base`
  - `//runtime/util:litert_status_util`
  - `@com_google_absl//absl/base:log_severity`
  - `@com_google_absl//absl/flags:flag`
  - `@com_google_absl//absl/flags:parse`
  - `@com_google_absl//absl/functional:any_invocable`
  - `@com_google_absl//absl/log:absl_check`
  - `@com_google_absl//absl/log:globals`
  - `@com_google_absl//absl/status`, `@com_google_absl//absl/status:statusor`
  - `@com_google_absl//absl/time`
  - `@nlohmann_json//:json`

#### `examples/simple_chat/simple_chat.cc`

**Flags:**
```cpp
ABSL_FLAG(std::string, model_path, "", "Path to the .litertlm model file.");
ABSL_FLAG(std::string, backend, "cpu", "Backend to use (cpu, gpu, npu).");
ABSL_FLAG(bool, add_constraint, true, "Enable constrained decoding.");
ABSL_FLAG(std::string, input_text, "", "Input text for extraction.");
ABSL_FLAG(std::string, input_text_file, "", "Path to input text file.");
ABSL_FLAG(std::string, litert_dispatch_lib_dir, "",
          "Directory of the LiteRT dispatch library (e.g. for NPU/QNN).");
```

**Embedded constants:**
- `kKnowledgeGraphSystemPrompt` — multi-paragraph system prompt instructing the
  model to extract nodes (id, name, type, description) and edges
  (source_node_id, target_node_id, relationship_name) as a knowledge graph.
- `kKnowledgeGraphSchema` — full JSON Schema (draft-07) defining `KnowledgeGraph`
  with `nodes: Node[]` and `edges: Edge[]`. The schema is embedded as a raw
  string literal.

**`Run()` function flow:**
1. Validate `--model_path`, `--input_text` / `--input_text_file` flags.
2. Read text from file if `--input_text_file` is set (`ReadTextFile` helper
   using `std::ifstream`).
3. Suppress verbose logs: `absl::SetMinLogLevel(kFatal)` +
   `absl::SetStderrThreshold(kFatal)`.
4. Call `ModelAssets::Create(model_path)`.
5. Map `--backend` string to `Backend` enum via `litert::lm::GetBackendFromString()`.
6. Call `EngineSettings::CreateDefault(model_assets, backend)`.
7. If `--litert_dispatch_lib_dir` is set, call
   `engine_settings.GetMutableMainExecutorSettings().SetLitertDispatchLibDir(dir)`.
8. Enable benchmarking: `engine_settings.GetMutableBenchmarkParams() = proto::BenchmarkParams()`.
9. Call `EngineFactory::CreateAny(engine_settings)`.
10. Create `SessionConfig::CreateDefault()` with `SetMaxOutputTokens(2048)`.
11. Build `ConversationConfig` via `ConversationConfig::Builder()`:
    - `.SetSessionConfig(session_config)`
    - `.SetEnableConstrainedDecoding(enable_constrained_decoding)`
    - `.SetConstraintProviderConfig(LlGuidanceConfig())`
    - `.Build(*engine)`
12. Call `Conversation::Create(*engine, conversation_config)`.
13. Build `LlGuidanceConstraintArg` with `constraint_type = kJsonSchema` and
    `constraint_string = kKnowledgeGraphSchema`.
14. Serialize the schema to compact JSON for the prompt
    (`CompactJsonSchemaForPrompt` helper using `json::parse` + `.dump()`).
15. Build the prompt: `kKnowledgeGraphSystemPrompt + compact_schema + source_text`.
16. Call `conversation->SendMessageAsync(json_message, callback, optional_args)`.
17. Call `engine->WaitUntilDone(absl::Minutes(10))`.
18. Attempt to parse `full_response` as JSON; fall back to extracting the first
    complete `{...}` object by brace-counting if full parse fails.
19. Print benchmark results via `conversation->GetBenchmarkInfo()`.

### Verification

```bash
# Build and run on Linux (CPU)
bazel build --config=linux_x86_64 //examples/simple_chat:simple_chat
bazel-bin/examples/simple_chat/simple_chat \
  --model_path=/path/to/model.litertlm \
  --input_text="Alice works at Google. Bob works at Meta." \
  --backend=cpu

# Build for Android
bazel build --config=android_arm64 //examples/simple_chat:simple_chat
```

Expected: valid JSON output with `{"nodes": [...], "edges": [...]}` structure.

---

## Change 2: Speculative GPU Constrained Decoding (`LITERT_LM_ASYNC_CONSTRAINT_MASKING`)

### Status

**✅ Done.** All 10 files modified: `build_config/BUILD`, `runtime/components/constrained_decoding/constrained_decoder.{h,cc}`, `runtime/components/constrained_decoding/BUILD`, `runtime/executor/llm_executor_base.h`, `runtime/executor/llm_litert_compiled_model_executor.{h,cc}`, `runtime/executor/BUILD`, `runtime/core/tasks.cc`, `runtime/core/BUILD`. Enable with `--define=async_constraint_masking=true`.

### What it does
The problem: constrained decoding previously required a full GPU→CPU→GPU logits
round-trip (~1 MB) every decode step to apply the token bitmap mask. On mobile
GPUs this is very expensive.

The fix: a new execution mode (opt-in via Bazel
`--define=async_constraint_masking=true`) that overlaps constraint bitmap
computation with the GPU forward pass and speculatively samples a token on the
GPU without any copy. Only when the speculative token is invalid does it fall
back to the old copy-mask-resample path.

**Algorithm per decode step:**
1. After `UpdateConstraintState` for the previous token, fire off async bitmap
   precomputation on a 1-thread background pool.
2. Run the GPU forward pass (`DecodeInternal` / `DecodeLogits`) in parallel —
   logits remain on GPU.
3. Call `SampleToken(logits)` to sample a token ID directly on the GPU without
   any copy.
4. Call `ValidateSpeculativeTokens(token_ids)` — waits for bitmap, then checks
   if the sampled token is allowed.
5. **Fast path (token valid):** call `AcceptSpeculativeTokens(token_ids)` to
   advance constraint state, write token to output, return immediately. No
   logits copy.
6. **Slow path (token invalid or `SampleToken` unsupported):** call
   `ApplyPrecomputedMask(logits)` which waits for bitmap and masks logits
   in-place; then resample normally.

### Files to modify

#### `build_config/BUILD`
Add a new `config_setting`:
```python
config_setting(
    name = "async_constraint_masking",
    define_values = {"async_constraint_masking": "true"},
)
```

#### `runtime/components/constrained_decoding/BUILD`
In the `constrained_decoder` `cc_library`:
- Add `local_defines = select({"//build_config:async_constraint_masking": ["LITERT_LM_ASYNC_CONSTRAINT_MASKING"], "//conditions:default": []})`.
- Append to `deps` via `select`: when `async_constraint_masking`:
  - `@com_google_absl//absl/synchronization`
  - `//runtime/framework:threadpool`

#### `runtime/components/constrained_decoding/constrained_decoder.h`
Under `#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING`, add includes:
- `absl/synchronization/mutex.h`
- `runtime/framework/threadpool.h`

Add public methods to `ConstrainedDecoder` (all inside `#ifdef`):
```cpp
// Starts computing the bitmap mask asynchronously on the provided thread pool.
// Must be called AFTER UpdateConstraintState for the current step.
absl::Status StartPrecomputeMask(ThreadPool& thread_pool);

// Waits for the precomputed mask and applies it to the logits buffer.
// Overload 1: operates directly on a TensorBuffer (calls Overload 2 internally).
absl::Status ApplyPrecomputedMask(::litert::TensorBuffer& logits);
// Overload 2: operates on a flat float span + dimension info.
absl::Status ApplyPrecomputedMask(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims);

// Checks per-batch whether each token_id is allowed by the precomputed bitmap.
// Waits for bitmap completion. Returns true only if ALL batch entries are valid.
absl::StatusOr<bool> ValidateSpeculativeTokens(absl::Span<const int> token_ids);

// Advances constraint state for each batch entry using the given token IDs.
// Clears precomputed bitmaps. Must be called instead of UpdateConstraintState
// when speculative sampling succeeds.
absl::Status AcceptSpeculativeTokens(absl::Span<const int> token_ids);
```

Add private members (inside `#ifdef`):
```cpp
absl::Mutex mask_mutex_;
bool mask_ready_ ABSL_GUARDED_BY(mask_mutex_) = false;
absl::Status mask_status_ ABSL_GUARDED_BY(mask_mutex_);
std::vector<std::unique_ptr<Bitmap>> precomputed_bitmaps_
    ABSL_GUARDED_BY(mask_mutex_);
```

#### `runtime/components/constrained_decoding/constrained_decoder.cc`
Implement all four methods inside `#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING`:

**`StartPrecomputeMask`:** Reset `mask_ready_=false`, clear `precomputed_bitmaps_`,
resize to `batch_size_`. Schedule a lambda on `thread_pool` that calls
`constraint_->ComputeBitmap(*constraint_states_[b])` for each batch, stores
results, then acquires mutex, stores bitmaps/status, sets `mask_ready_=true`.

**`ApplyPrecomputedMask(TensorBuffer&)`:** Use
`ReferTensorBufferAsSpan<float>`, delegate to the span overload.

**`ApplyPrecomputedMask(Span<float>, Span<Dim>)`:** Validate dims
(`[batch, 1, vocab]`). Lock mutex, `Await(mask_ready_)`, check status. For each
batch element, set `logits[b*vocab+i] = lowest()` for all `i` where
`!bitmap->Get(i)`. Reset `mask_ready_=false`, clear bitmaps.

**`ValidateSpeculativeTokens`:** Lock mutex, `Await(mask_ready_)`, check status.
For each batch element, return false if `!precomputed_bitmaps_[b]->Get(token_ids[b])`.
Return true if all valid.

**`AcceptSpeculativeTokens`:** For each batch element, call
`constraint_->ComputeNext(*constraint_states_[i], token_ids[i])`, reset state,
call `constraint_->IsEnded()` and restart if ended. Clear precomputed bitmaps
under lock.

#### `runtime/core/BUILD`
In the `tasks` `cc_library`:
- Add `local_defines = select({"//build_config:async_constraint_masking": ["LITERT_LM_ASYNC_CONSTRAINT_MASKING"], "//conditions:default": []})`.
- Append to `deps` via `select`: when `async_constraint_masking`:
  - `//runtime/framework:threadpool`

#### `runtime/core/tasks.cc`
In the `DecodeOneStep` class (inside the anonymous namespace):

1. Add `#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING` include for `threadpool.h`.

2. In the constructor, after creating `constrained_decoder_`, create:
   ```cpp
   mask_thread_pool_ = std::make_unique<ThreadPool>("constraint_mask", 1);
   ```

3. Add a new `#ifdef`-guarded block **before** the existing standard decode path
   in the main decode method. The block:
   - Calls `constrained_decoder_->StartPrecomputeMask(*mask_thread_pool_)`
   - Calls `executor_.DecodeLogits(inputs)` — GPU forward pass
   - Calls `executor_.SampleToken(output_logits)` — speculative GPU sample
   - On success: calls `ValidateSpeculativeTokens`
     - If valid: calls `AcceptSpeculativeTokens`, copies IDs to `decoded_ids`,
       sets zero scores, `return std::move(decoded_ids)`
   - On failure (invalid token or `SampleToken` unimplemented): falls through to
     `ApplyPrecomputedMask(output_logits)`, then standard `SampleToIdAndScoreBuffer`.

4. Add a comment `// --- STANDARD PATH ---` before the existing decode block.

5. Add private member:
   ```cpp
   #ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING
   std::unique_ptr<ThreadPool> mask_thread_pool_;
   #endif
   ```

#### `runtime/executor/llm_executor_base.h`
Add virtual method (with default `UnimplementedError`):
```cpp
// Samples token(s) from logits using the executor's internal sampler.
// Logits may be GPU-resident; sampling happens on the same device.
// Returns token IDs tensor buffer of shape [output_heads].
virtual absl::StatusOr<::litert::TensorBuffer> SampleToken(
    const ::litert::TensorBuffer& logits) {
  return absl::UnimplementedError(absl::StrCat(
      "SampleToken not implemented for backend: ", ExecutorBackendName()));
}
```

#### `runtime/executor/BUILD`
In `llm_litert_compiled_model_executor` `cc_library`:
- Add `local_defines = select({"//build_config:async_constraint_masking": ["LITERT_LM_ASYNC_CONSTRAINT_MASKING"], "//conditions:default": []})`.
- Append to `deps` via `select`: when `async_constraint_masking`:
  - `//runtime/framework:threadpool`

#### `runtime/executor/llm_litert_compiled_model_executor.h`
Add inside `#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING`:
```cpp
// DecodeLogits variant that optionally skips constraint mask application.
absl::StatusOr<TensorBuffer> DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params,
    bool skip_constraint_masking);
```

Add public override (always present):
```cpp
absl::StatusOr<TensorBuffer> SampleToken(const TensorBuffer& logits) override;
```

Add private member:
```cpp
#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING
std::unique_ptr<ThreadPool> mask_thread_pool_;
#endif
```

#### `runtime/executor/llm_litert_compiled_model_executor.cc`

**Override `Decode(TensorBuffer&, const ExecutorDecodeParams&)`:**

Wrap the entire body in `#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING` / `#else` / `#endif`.

When `LITERT_LM_ASYNC_CONSTRAINT_MASKING`:
- If `decode_params.HasConstraintDecoder()`:
  - Call `DecodeLogits(inputs, decode_params, /*skip_constraint_masking=*/true)`
    (bitmap precomputation already started inside `DecodeLogits`)
  - Call `SampleToken(decoded_logits)` speculatively
  - If ok: call `ValidateSpeculativeTokens`
    - If valid: `AcceptSpeculativeTokens`, write token to `output_tokens`,
      set `sampled_speculatively = true`
  - If not speculatively sampled: apply precomputed mask
    (handles both host-memory and GPU tensor buffer cases by conditionally
    copying to float vector, masking, writing back), then `SampleLogits`.
    After fallback: `UpdateConstraintState` with the fallback token.
- Else (no constraint): `DecodeLogits` + `SampleLogits` as before.

`#else` (standard path): `DecodeLogits` + `SampleLogits`.

**`DecodeLogits` 2-arg overload:** Inside `#ifdef`, delegate to new 3-arg overload
with `skip_constraint_masking=false`.

**`DecodeLogits` 3-arg overload (when `LITERT_LM_ASYNC_CONSTRAINT_MASKING`):**
- When `HasConstraintDecoder()`:
  - If `!skip_constraint_masking && !token.empty() && last_run_is_decode`:
    call `UpdateConstraintState` with current token IDs.
  - Create `mask_thread_pool_` if null.
  - Call `decode_params.GetConstraintDecoder()->StartPrecomputeMask(*mask_thread_pool_)`.
- Call `DecodeInternal` (GPU forward pass — bitmap runs in parallel).
- Call `ConsumePendingOrAddProcessedToken`.
- If `HasConstraintDecoder() && !skip_constraint_masking`:
  - Apply precomputed mask (host-memory: direct; GPU: copy to float vector,
    mask, write back).
- `#else` (original path): update constraint state, `MaskLogits` as before.

**New `SampleToken` implementation:**
```cpp
absl::StatusOr<TensorBuffer>
LlmLiteRtCompiledModelExecutorBase::SampleToken(const TensorBuffer& logits) {
  if (sampler_ == nullptr) RETURN_IF_ERROR(InitializeSampler(logits_data_type_));
  int output_heads = llm_context_->runtime_config().output_heads.value_or(1);
  LITERT_ASSIGN_OR_RETURN(auto ids_tensor,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Int32, Layout({output_heads})),
          output_heads * sizeof(int32_t)));
  RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  return std::move(ids_tensor);
}
```

### Verification

```bash
# Default build (no async masking) must still pass all tests
bazel test --config=linux_x86_64 //runtime/components/constrained_decoding/...

# Build with async masking enabled
bazel build --config=linux_x86_64 \
  --define=async_constraint_masking=true \
  //runtime/...

# GPU integration test (requires GPU hardware)
bazel build --config=linux_x86_64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  --define=async_constraint_masking=true \
  //runtime/engine:litert_lm_main
```

Expected: clean build in both `--define=async_constraint_masking=true` and default modes. No regressions in constrained decoding tests.

---

## Change 3: NPU Structured Output (Constrained Decoding) + Dynamic Prefill Size

### Status

**✅ Done.**

- **3a (NPU constrained decoding):** `ApplyConstrainedGreedySampling` helper added; `Decode()` no longer hard-fails with `UnimplementedError`.
- **3b (Dynamic prefill):** `kPrefillSignature`/`kPrefillSize` replaced with `kPrefillSignaturePrefix`/`kInputPosTensorName`; `AllocateTransformerBuffers`, `CreateLlmInferenceContextWithBufferSharing`, and `WarmupInference` now take `absl::string_view prefill_signature`; both factory functions use `GetPrefillRunnerSetFromModel` dynamically.

### What it does

#### 3a. Constrained Decoding on NPU
Removes the hard `return absl::UnimplementedError("Constrained decoding is not
supported on NPU.")` stub in `LlmLiteRtNpuCompiledModelExecutor::Decode()` and
replaces it with a real implementation via greedy sampling on CPU-copied logits.

#### 3b. Dynamic Prefill Signature Discovery
Replaces the hardcoded `kPrefillSignature = "prefill_128"` / `kPrefillSize = 128`
constants in `CreateForGemma3` and `CreateForGemma3n` with dynamic discovery
using the existing `GetPrefillRunnerSetFromModel()` utility. This means NPU
models can now have any prefill length (e.g., 64, 256, 512, etc.).

### Files to modify

#### `runtime/executor/llm_litert_npu_compiled_model_executor.cc`

**Constants:** Replace:
```cpp
constexpr char kPrefillSignature[] = "prefill_128";
constexpr int kPrefillSize = 128;
```
with:
```cpp
constexpr absl::string_view kPrefillSignaturePrefix = "prefill";
constexpr absl::string_view kInputPosTensorName = "input_pos";
```

**New helper function `ApplyConstrainedGreedySampling`:**
```cpp
absl::StatusOr<int> ApplyConstrainedGreedySampling(
    const TensorBuffer& decoded_logits,
    ConstrainedDecoder* constraint_decoder);
```
Implementation:
1. Get `RankedTensorType` from `decoded_logits`.
2. Copy logits to `std::vector<float>`:
   - If `Float32`: use `CopyFromTensorBuffer<float>`.
   - If `Int16`: use `CopyFromTensorBuffer<int16_t>`, convert each to `float`.
   - Otherwise: return `InvalidArgumentError`.
3. Call `constraint_decoder->MaskLogits(absl::MakeSpan(logits), dims)`.
4. Find argmax and return it.

**`HasPerLayerEmbedder` fix:** Replace the hardcoded `kPrefillSignature` lookup
with a loop over all signatures that match `kPrefillSignaturePrefix`, checking
each for the `kPerLayerEmbedderTensor` input name.

**`AllocateTransformerBuffers`:** Add `absl::string_view prefill_signature`
parameter. Replace all occurrences of `kPrefillSignature` with the parameter.
Update both the `FindSignature()` call and all `CreateInputBuffer()` /
`CreateOutputBuffer()` calls.

**`CreateLlmInferenceContextWithBufferSharing`:** Add `absl::string_view
prefill_signature` parameter. Replace `kPrefillSignature` in
`GetInputTensorType()` and `CreateInputBuffer()` calls.

**`WarmupInference`:** Add `absl::string_view prefill_signature` parameter.
Replace `LlmSignatures::kPrefillLlm` with the parameter in the `Run()` call.

**`PrefillInternal`:** Replace `kPrefillSignature` with `prefill_signature`
in `llm_compiled_model_.Run()`.

**`Decode`:**
1. Remove the early-return guard `if (decode_params.HasConstraintDecoder()) return UnimplementedError`.
2. Add constrained state update before `DecodeInternal`:
   ```cpp
   if (decode_params.HasConstraintDecoder() && latency_stats_.decode_num_tokens > 0) {
     int last_token_id = pending_input_token[0]->id();
     RETURN_IF_ERROR(decode_params.GetConstraintDecoder()->UpdateConstraintState(
         absl::MakeSpan(&last_token_id, 1)));
   }
   ```
3. Replace `ApplyGreedySampling` with conditional:
   ```cpp
   int max_index;
   if (decode_params.HasConstraintDecoder()) {
     ASSIGN_OR_RETURN(max_index, ApplyConstrainedGreedySampling(
         decoded_logits, decode_params.GetConstraintDecoder()));
   } else {
     ASSIGN_OR_RETURN(max_index, ApplyGreedySampling(decoded_logits));
   }
   ```

**`Prefill`:** Fix stats: replace `latency_stats_.prefill_num_tokens += kPrefillSize`
with `+= prefill_length` (the actual chunk length).

**`CreateForGemma3` and `CreateForGemma3n`:** In both factory functions:
1. After `CompiledModel::Create(...)`, add:
   ```cpp
   ASSIGN_OR_RETURN(auto prefill_runner_set,
       GetPrefillRunnerSetFromModel(*transformer_model,
           kPrefillSignaturePrefix, kInputPosTensorName));
   RET_CHECK(!prefill_runner_set.empty()) << "No prefill signatures found";
   const std::string prefill_signature = prefill_runner_set.begin()->second;
   ```
2. Pass `prefill_signature` to `AllocateTransformerBuffers`,
   `CreateLlmInferenceContextWithBufferSharing`, and `WarmupInference`.
3. Remove the old hardcoded `SortedPrefillSignatureMap` construction
   (`prefill_runner_set[kPrefillSize] = kPrefillSignature`).

#### `runtime/executor/llm_litert_npu_compiled_model_executor.h`
Add `absl::string_view prefill_signature` parameter to the declarations of:
- `AllocateTransformerBuffers`
- `CreateLlmInferenceContextWithBufferSharing`
- `WarmupInference`

### Verification

```bash
# Build NPU executor
bazel build --config=android_arm64 \
  //runtime/executor:llm_litert_npu_compiled_model_executor

# Verify NPU executor compiles cleanly for all supported Android targets
bazel build --config=android_arm64 //runtime/executor/...
bazel build --config=android_x86_64 //runtime/executor/...
```

Functional verification requires NPU hardware (QNN/HTP). Check that:
1. `llm_litert_npu_compiled_model_executor.cc` no longer has the `UnimplementedError` constraint guard.
2. A model with non-128 prefill length (e.g., `prefill_256`) is selected correctly.

---

## Change 4: Soft Degradation for Non-SentencePiece Tokenizers

### Status

**✅ Done.** Both `gemma3_data_processor.cc` and `function_gemma_data_processor.cc` now log a warning and disable constrained decoding instead of returning an error when the tokenizer is non-SentencePiece or the constraint provider fails to initialize.

### What it does
`Gemma3DataProcessor::Create()` and `FunctionGemmaDataProcessor::Create()` both
previously hard-failed with `absl::InvalidArgumentError` when constrained
decoding was requested but the tokenizer was not SentencePiece (e.g., HuggingFace
BPE tokenizer). They also hard-failed if `LiteRtLmGemmaModelConstraintProvider_Create`
returned null.

Both hard errors are replaced with `ABSL_LOG(WARNING)` + graceful continuation
without the Gemma grammar constraint provider. The model still runs, just without
per-token tool-call constraints.

### Files to modify

#### `runtime/conversation/model_data_processor/gemma3_data_processor.cc`
Add `#include "absl/log/absl_log.h"`.

In `Gemma3DataProcessor::Create()`, replace the constraint provider setup block:
```cpp
// BEFORE (hard fail):
if (tokenizer->GetTokenizerType() != TokenizerType::kSentencePiece) {
  return absl::InvalidArgumentError("Constrained decoding is only supported...");
}
auto sp_tokenizer = reinterpret_cast<const SentencePieceTokenizer*>(tokenizer);
...
if (provider == nullptr) {
  return absl::InternalError("Failed to create GemmaModelConstraintProvider.");
}
constraint_provider.reset(provider);

// AFTER (soft degrade):
if (tokenizer->GetTokenizerType() != TokenizerType::kSentencePiece) {
  ABSL_LOG(WARNING) << "GemmaModelConstraintProvider disabled: constrained "
                       "tool constraints require SentencePiece tokenizer.";
} else {
  auto sp_tokenizer = reinterpret_cast<const SentencePieceTokenizer*>(tokenizer);
  ...
  if (provider == nullptr) {
    ABSL_LOG(WARNING) << "GemmaModelConstraintProvider unavailable for this "
                         "model; continuing without Gemma tool constraints.";
  } else {
    constraint_provider.reset(provider);
  }
}
```

#### `runtime/conversation/model_data_processor/function_gemma_data_processor.cc`
Apply the identical change as above in `FunctionGemmaDataProcessor::Create()`.
Also add `#include "absl/log/absl_log.h"`.

### Verification

```bash
# Build the conversation layer
bazel build --config=linux_x86_64 //runtime/conversation/...

# Run any existing data processor tests
bazel test --config=linux_x86_64 //runtime/components/...
```

Functional verification: run with a model that uses a HuggingFace BPE tokenizer + `--add_constraint=true`. Previously this crashed before prefill; after the fix it should print a warning and continue without tool-call constraints.

---

## Change 5: Robustness / Plumbing Fixes

### Status

| Sub-change | Status | Notes |
|------------|--------|-------|
| 5a — Optional LlmMetadata | ✅ Done | `GetLlmMetadata()` returns `std::nullopt` when section absent instead of crashing |
| 5b — Fallback input tensor names | ✅ Done | `litert_compiled_model_executor_utils.cc` falls back to alternate input tensor names |
| 5c — `.gitignore` additions | ✅ Done | `/build/` present; `/target/` not needed |

Full diff in `stash@{0}`.

### 5a. Optional LlmMetadata in model file

**Problem:** `LitertLmLoader::GetLlmMetadata()` called `.value()` on an empty
optional when the model had no LlmMetadata section, causing a crash. The callers
then had no way to distinguish "not found" from a real error.

**Files to modify:**

`runtime/util/litert_lm_loader.h` — In `GetLlmMetadata()`, replace:
```cpp
return GetSectionBuffer(BufferKey(...)).value();
```
with:
```cpp
auto optional_section_buffer = GetSectionBuffer(BufferKey(...));
if (optional_section_buffer.has_value()) {
  return optional_section_buffer.value();
}
ABSL_LOG(WARNING) << "LlmMetadata not found. Skipping.";
return litert::BufferRef<uint8_t>();
```

`runtime/components/model_resources_litert_lm.cc` — In `GetLlmMetadata()`,
after calling `litert_lm_loader_->GetLlmMetadata()`, add a size check before
parsing:
```cpp
auto buffer_ref = litert_lm_loader_->GetLlmMetadata();
if (buffer_ref.Size() == 0) {
  return absl::NotFoundError("LlmMetadata not found in the model.");
}
```

### 5b. Improved Prefill Signature Input Tensor Detection

**Problem:** `GetPrefillRunnerSetFromModel()` only handled models whose prefill
signature input was named exactly `input_positions` (the `input_positions_name`
parameter). Models using other naming conventions or embedding inputs were silently
skipped.

**File to modify:** `runtime/executor/litert_compiled_model_executor_utils.cc`

In `GetPrefillRunnerSetFromModel()`:

1. If `signature.InputTensor(input_positions_name)` fails (returns falsy), try
   fallback names in order: `"embeddings"`, `"tokens"`, `"token_ids"`.
2. If no input tensor is found after all fallbacks, `continue` (skip signature).
3. Support **rank-3** tensors (`[batch, seq_len, hidden_size]` — embedding-based
   models): use `Dimensions()[1]` as the sequence length, same as rank-2.
4. Use `->RankedTensorType()` (pointer dereference) since the tensor is now
   obtained via the optional/pointer API.

```cpp
constexpr std::array<absl::string_view, 3> kFallbackInputNames = {
    "embeddings", "tokens", "token_ids"};

auto input_sequence_tensor = signature.InputTensor(input_positions_name);
if (!input_sequence_tensor) {
  for (auto fallback_name : kFallbackInputNames) {
    input_sequence_tensor = signature.InputTensor(fallback_name);
    if (input_sequence_tensor) break;
  }
}
if (!input_sequence_tensor) continue;

LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                        input_sequence_tensor->RankedTensorType());
if (ranked_tensor_type.Layout().Rank() == 2) {
  // [batch_size, max_seq_len]
  prefill_runner_set[ranked_tensor_type.Layout().Dimensions()[1]] = signature_key;
} else if (ranked_tensor_type.Layout().Rank() == 3) {
  // [batch_size, max_seq_len, hidden_size]
  prefill_runner_set[ranked_tensor_type.Layout().Dimensions()[1]] = signature_key;
} else if (ranked_tensor_type.Layout().Rank() == 1) {
  // [max_seq_len]
  prefill_runner_set[ranked_tensor_type.Layout().Dimensions()[0]] = signature_key;
}
```

### 5c. `.gitignore` additions

**Status: ✅ Done.** `/build/` has been added to `.gitignore`. `/target/` is not needed because the VS Code CMake extension is configured to use `build/` via `.vscode/settings.json`.

### Verification

```bash
# 5a: Build model loading with a model lacking LlmMetadata section
bazel build --config=linux_x86_64 //runtime/util:litert_lm_loader
bazel build --config=linux_x86_64 //runtime/components:model_resources_litert_lm

# 5b: Build and run the executor utils test (if one exists)
bazel test --config=linux_x86_64 //runtime/executor/...

# 5c: Verify gitignore
cat .gitignore
```

For 5a: functional test requires a `.litertlm` model file that has no `LlmMetadataProto` section — the engine should log a warning and continue rather than crash.

For 5b: test with a model whose prefill signature input is named `embeddings` or `tokens` rather than `input_positions`.

---

## Re-implementation Order

When applying these changes to a newer upstream revision, the recommended order
is:

1. **Change 5c** — `.gitignore` (trivial, no conflicts expected)
2. **Change 5a** — Optional LlmMetadata (2 files, self-contained)
3. **Change 5b** — Prefill signature detection (1 file, low conflict risk)
4. **Change 4** — Soft degradation for non-SP tokenizers (2 files, low conflict risk)
5. **Change 3** — NPU constrained decoding + dynamic prefill (2 files, medium risk)
6. **Change 2** — Speculative GPU constrained decoding (7 files, highest complexity)
7. **Change 1** — Simple chat demo (2 new files, no conflict risk)

After each change, verify compilation with:
```bash
bazel build --config=linux_x86_64 //runtime/... //examples/...
# For NPU/Android:
bazel build --config=android_arm64 //runtime/... //examples/...
# With speculative decoding enabled:
bazel build --config=linux_x86_64 --define=async_constraint_masking=true //runtime/...
```
