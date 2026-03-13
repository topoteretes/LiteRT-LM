#!/bin/bash
# Benchmark script for LiteRT-LM knowledge graph extraction on Android.
# Builds/pushes the binary and runtime libs, then runs NPU structured-output
# extraction for four text pieces.
#
# Usage: ./benchmark_android.sh [--skip-build] [--skip-push]

set -euo pipefail

export PATH="/home/dmytro/Android/Sdk/platform-tools:$PATH"
export ANDROID_HOME="/home/dmytro/Android/Sdk"

MODEL_LOCAL="${MODEL_LOCAL:-$HOME/.litert-lm/models/gemma3-1b-it-int4.litertlm}"
MODEL_LOCAL_NPU="${MODEL_LOCAL_NPU:-$HOME/.litert-lm/models/gemma3-1b-it-int4-qualcomm.litertlm}"
PREBUILT_DIR="prebuilt/android_arm64"
DEVICE_DIR="/data/local/tmp"
MODEL_DEVICE="$DEVICE_DIR/model.litertlm"
MODEL_DEVICE_NPU="$DEVICE_DIR/model_npu.litertlm"
NPU_DISPATCH_DIR="${NPU_DISPATCH_DIR:-$DEVICE_DIR}"

find_first_file() {
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

find_qairt_file() {
  local rel_path="$1"
  local hit
  hit=$(find "$HOME/.cache/bazel" -type f -path "*/external/qairt/lib/$rel_path" 2>/dev/null | head -n 1 || true)
  if [[ -n "$hit" ]]; then
    echo "$hit"
    return 0
  fi
  return 1
}

DISPATCH_SO="${DISPATCH_SO:-$(find_first_file \
  "$PWD/bazel-bin/external/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so" \
  "$PWD/bazel-out/arm64-v8a-opt/bin/external/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so" \
  "$HOME/.cache/bazel/_bazel_dmytro/4aa95d11f0e0f122dfcdd727d11fcb01/execroot/litert_lm/bazel-out/arm64-v8a-opt/bin/external/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so")}"

QNN_HTP_SO="${QNN_HTP_SO:-$(find_qairt_file 'aarch64-android/libQnnHtp.so')}"
QNN_PREPARE_SO="${QNN_PREPARE_SO:-$(find_qairt_file 'aarch64-android/libQnnHtpPrepare.so')}"
QNN_SYSTEM_SO="${QNN_SYSTEM_SO:-$(find_qairt_file 'aarch64-android/libQnnSystem.so')}"
QNN_HTP_SKEL_SO="${QNN_HTP_SKEL_SO:-$(find_qairt_file 'hexagon-v75/unsigned/libQnnHtpV75Skel.so')}"
QNN_HTP_STUB_SO="${QNN_HTP_STUB_SO:-$(find_qairt_file 'aarch64-android/libQnnHtpV75Stub.so')}"

SKIP_BUILD=false
SKIP_PUSH=false
for arg in "$@"; do
  case $arg in
    --skip-build) SKIP_BUILD=true ;;
    --skip-push) SKIP_PUSH=true ;;
  esac
done

# --- 1. Check device ---
echo "Checking device..."
DEVICE=$(adb devices | grep -w "device" | head -1 | cut -f1)
if [ -z "$DEVICE" ]; then
  echo "ERROR: No Android device found. Connect a device and retry."
  exit 1
fi
echo "Device: $DEVICE"
echo "CPU/GPU model: $MODEL_LOCAL"
echo "NPU model:     $MODEL_LOCAL_NPU"
echo "Dispatch .so:  ${DISPATCH_SO:-<not found>}"

if [ ! -f "$MODEL_LOCAL" ]; then
  echo "ERROR: CPU/GPU model not found: $MODEL_LOCAL"
  exit 1
fi
if [ ! -f "$MODEL_LOCAL_NPU" ]; then
  echo "ERROR: NPU model not found: $MODEL_LOCAL_NPU"
  exit 1
fi

for f in "$DISPATCH_SO" "$QNN_HTP_SO" "$QNN_PREPARE_SO" "$QNN_SYSTEM_SO" "$QNN_HTP_SKEL_SO" "$QNN_HTP_STUB_SO"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo "ERROR: Missing required NPU runtime library: ${f:-<empty>}"
    echo "Hint: build //runtime/engine:litert_lm_main with --config=android_arm64 first."
    exit 1
  fi
done

# --- 2. Build ---
if [ "$SKIP_BUILD" = false ]; then
  echo ""
  echo "Building sync binary (no async flag)..."
  bazel build --config=android_arm64 //examples/simple_chat:simple_chat 2>&1 | tail -3
  cp -f bazel-bin/examples/simple_chat/simple_chat /tmp/simple_chat_sync

  echo ""
  echo "Building speculative binary (async_constraint_masking=true)..."
  bazel build --config=android_arm64 --define=async_constraint_masking=true \
    //examples/simple_chat:simple_chat 2>&1 | tail -3
  cp -f bazel-bin/examples/simple_chat/simple_chat /tmp/simple_chat_spec
else
  echo "Skipping build (--skip-build)"
fi

# --- 3. Push to device ---
if [ "$SKIP_PUSH" = false ]; then
  echo ""
  echo "Pushing binaries and libs to device..."

  # Clean old binaries
  adb shell "rm -f $DEVICE_DIR/simple_chat_sync $DEVICE_DIR/simple_chat_spec" 2>/dev/null || true

  # Push binaries
  adb push /tmp/simple_chat_sync "$DEVICE_DIR/simple_chat_sync"
  adb push /tmp/simple_chat_spec "$DEVICE_DIR/simple_chat_spec"
  adb shell "chmod +x $DEVICE_DIR/simple_chat_sync $DEVICE_DIR/simple_chat_spec"

  # Push GPU shared libs
  for f in "$PREBUILT_DIR"/*.so; do
    adb push "$f" "$DEVICE_DIR/"
  done

  # Push Qualcomm dispatch + QNN runtime libs for NPU.
  adb push "$DISPATCH_SO" "$DEVICE_DIR/libLiteRtDispatch_Qualcomm.so"
  adb push "$QNN_HTP_SO" "$DEVICE_DIR/libQnnHtp.so"
  adb push "$QNN_PREPARE_SO" "$DEVICE_DIR/libQnnHtpPrepare.so"
  adb push "$QNN_SYSTEM_SO" "$DEVICE_DIR/libQnnSystem.so"
  adb push "$QNN_HTP_SKEL_SO" "$DEVICE_DIR/libQnnHtpV75Skel.so"
  adb push "$QNN_HTP_STUB_SO" "$DEVICE_DIR/libQnnHtpV75Stub.so"

  # Push CPU/GPU model if not already on device
  if ! adb shell "test -f $MODEL_DEVICE" 2>/dev/null; then
    echo "Pushing CPU/GPU model (this may take a moment)..."
    adb push "$MODEL_LOCAL" "$MODEL_DEVICE"
  else
    echo "CPU/GPU model already on device, skipping."
  fi

  # Push NPU model if not already on device
  if ! adb shell "test -f $MODEL_DEVICE_NPU" 2>/dev/null; then
    echo "Pushing NPU model (this may take a moment)..."
    adb push "$MODEL_LOCAL_NPU" "$MODEL_DEVICE_NPU"
  else
    echo "NPU model already on device, skipping."
  fi
else
  echo "Skipping push (--skip-push)"
fi

# --- 4. Prepare knowledge extraction texts ---
KG_TEXT_1_HOST="/tmp/kg_text_1.txt"
KG_TEXT_2_HOST="/tmp/kg_text_2.txt"
KG_TEXT_3_HOST="/tmp/kg_text_3.txt"
KG_TEXT_4_HOST="/tmp/kg_text_4.txt"

cat > "$KG_TEXT_1_HOST" <<'EOF'
J. Robert Oppenheimer was the scientific director of the Manhattan Project's Los Alamos Laboratory.
He coordinated theoretical and experimental teams that designed and tested the first atomic bombs.
Oppenheimer worked with U.S. Army leadership and many physicists who had fled Europe.
EOF

cat > "$KG_TEXT_2_HOST" <<'EOF'
General Leslie Groves directed the Manhattan Engineer District for the U.S. Army Corps of Engineers.
Groves oversaw budget, logistics, security, and construction across major project sites.
He selected Oppenheimer to lead the scientific work at Los Alamos.
EOF

cat > "$KG_TEXT_3_HOST" <<'EOF'
Key Manhattan Project locations included Los Alamos in New Mexico, Oak Ridge in Tennessee, and Hanford in Washington.
Oak Ridge developed uranium enrichment processes, while Hanford produced plutonium.
The project integrated universities, government agencies, and industrial contractors.
EOF

cat > "$KG_TEXT_4_HOST" <<'EOF'
The Manhattan Project involved the U.S. Army Corps of Engineers, the Office of Scientific Research and Development,
and research groups from institutions such as the University of California and the University of Chicago.
Scientists Enrico Fermi, Niels Bohr, and Richard Feynman were associated with project efforts.
EOF

if [ "$SKIP_PUSH" = false ]; then
  adb push "$KG_TEXT_1_HOST" "$DEVICE_DIR/kg_text_1.txt"
  adb push "$KG_TEXT_2_HOST" "$DEVICE_DIR/kg_text_2.txt"
  adb push "$KG_TEXT_3_HOST" "$DEVICE_DIR/kg_text_3.txt"
  adb push "$KG_TEXT_4_HOST" "$DEVICE_DIR/kg_text_4.txt"
fi

# --- 5. Run benchmarks ---
echo ""
echo "================================================================"
echo "  BENCHMARK: Knowledge Graph Extraction (NPU Structured Output)"
echo "  Device: $DEVICE"
echo "================================================================"

RESULT_LABELS=()
RESULT_PREFILL=()
RESULT_DECODE=()
RESULT_TOTAL=()
RESULT_JSON=()

run_benchmark() {
  local label="$1"
  local run_id="$2"
  local text_file="$3"
  local binary="${4:-simple_chat_sync}"
  local backend="${5:-npu}"
  local model="${6:-$MODEL_DEVICE_NPU}"
  local extra_args="${7:-}"
  local outfile="/tmp/bench_${run_id}.txt"

  echo ""
  echo "--- $label ---"
  echo "  Cooling down (5s)..."
  sleep 5

  local npu_args=""
  if [[ "$backend" == "npu" ]]; then
    npu_args="--litert_dispatch_lib_dir=$NPU_DISPATCH_DIR"
  fi

  # shellcheck disable=SC2086
  adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./$binary \
    --model_path=$model --backend=$backend \
    --add_constraint=true \
    $npu_args $extra_args \
    --input_text_file=$text_file" \
    > "$outfile" 2>&1

  local prefill=$(grep "Prefill Speed:" "$outfile" | head -1 | grep -oP '[\d.]+(?= tokens/sec)')
  local decode=$(grep "Decode Speed:" "$outfile" | head -1 | grep -oP '[\d.]+(?= tokens/sec)')
  local total=$(grep "Total Time:" "$outfile" | head -1 | grep -oP '[\d.]+(?= s)')
  local json=$(grep "JSON validation:" "$outfile" | head -1)
  local json_status="N/A"
  if [[ -n "${json:-}" ]]; then
    json_status=${json#JSON validation: }
  fi

  echo "  Prefill: ${prefill:-N/A} tok/s"
  echo "  Decode:  ${decode:-N/A} tok/s"
  echo "  Total:   ${total:-N/A} s"
  echo "  $json"

  RESULT_LABELS+=("$label")
  RESULT_PREFILL+=("${prefill:-N/A}")
  RESULT_DECODE+=("${decode:-N/A}")
  RESULT_TOTAL+=("${total:-N/A}")
  RESULT_JSON+=("$json_status")
}

# --- NPU ---
run_benchmark "NPU text 1 (oppenheimer)"   "npu_kg_text_1"    "$DEVICE_DIR/kg_text_1.txt" simple_chat_sync npu "$MODEL_DEVICE_NPU"
run_benchmark "NPU text 2 (groves)"        "npu_kg_text_2"    "$DEVICE_DIR/kg_text_2.txt" simple_chat_sync npu "$MODEL_DEVICE_NPU"
run_benchmark "NPU text 3 (laboratories)"  "npu_kg_text_3"    "$DEVICE_DIR/kg_text_3.txt" simple_chat_sync npu "$MODEL_DEVICE_NPU"
run_benchmark "NPU text 4 (organizations)" "npu_kg_text_4"    "$DEVICE_DIR/kg_text_4.txt" simple_chat_sync npu "$MODEL_DEVICE_NPU"

# --- CPU ---
run_benchmark "CPU text 1 (oppenheimer)"   "cpu_kg_text_1"    "$DEVICE_DIR/kg_text_1.txt" simple_chat_sync cpu "$MODEL_DEVICE"
run_benchmark "CPU text 2 (groves)"        "cpu_kg_text_2"    "$DEVICE_DIR/kg_text_2.txt" simple_chat_sync cpu "$MODEL_DEVICE"
run_benchmark "CPU text 3 (laboratories)"  "cpu_kg_text_3"    "$DEVICE_DIR/kg_text_3.txt" simple_chat_sync cpu "$MODEL_DEVICE"
run_benchmark "CPU text 4 (organizations)" "cpu_kg_text_4"    "$DEVICE_DIR/kg_text_4.txt" simple_chat_sync cpu "$MODEL_DEVICE"

# --- GPU (sync) ---
run_benchmark "GPU text 1 (oppenheimer)"   "gpu_kg_text_1"    "$DEVICE_DIR/kg_text_1.txt" simple_chat_sync gpu "$MODEL_DEVICE"
run_benchmark "GPU text 2 (groves)"        "gpu_kg_text_2"    "$DEVICE_DIR/kg_text_2.txt" simple_chat_sync gpu "$MODEL_DEVICE"
run_benchmark "GPU text 3 (laboratories)"  "gpu_kg_text_3"    "$DEVICE_DIR/kg_text_3.txt" simple_chat_sync gpu "$MODEL_DEVICE"
run_benchmark "GPU text 4 (organizations)" "gpu_kg_text_4"    "$DEVICE_DIR/kg_text_4.txt" simple_chat_sync gpu "$MODEL_DEVICE"

# --- GPU+spec (async constraint masking) ---
run_benchmark "GPU+spec text 1 (oppenheimer)"   "gpuspec_kg_text_1" "$DEVICE_DIR/kg_text_1.txt" simple_chat_spec gpu "$MODEL_DEVICE"
run_benchmark "GPU+spec text 2 (groves)"        "gpuspec_kg_text_2" "$DEVICE_DIR/kg_text_2.txt" simple_chat_spec gpu "$MODEL_DEVICE"
run_benchmark "GPU+spec text 3 (laboratories)"  "gpuspec_kg_text_3" "$DEVICE_DIR/kg_text_3.txt" simple_chat_spec gpu "$MODEL_DEVICE"
run_benchmark "GPU+spec text 4 (organizations)" "gpuspec_kg_text_4" "$DEVICE_DIR/kg_text_4.txt" simple_chat_spec gpu "$MODEL_DEVICE"

echo ""
echo "================================================================"
echo "  SUMMARY TABLE"
echo "================================================================"
printf "| %-34s | %-14s | %-14s | %-10s | %-32s |\n" "Run" "Prefill tok/s" "Decode tok/s" "Total (s)" "JSON validation"
printf "|-%-34s-|-%-14s-|-%-14s-|-%-10s-|-%-32s-|\n" "----------------------------------" "--------------" "--------------" "----------" "--------------------------------"
for i in "${!RESULT_LABELS[@]}"; do
  printf "| %-34s | %-14s | %-14s | %-10s | %-32s |\n" \
    "${RESULT_LABELS[$i]}" "${RESULT_PREFILL[$i]}" "${RESULT_DECODE[$i]}" "${RESULT_TOTAL[$i]}" "${RESULT_JSON[$i]}"
done

echo ""
echo "================================================================"
echo "  DONE"
echo "================================================================"
