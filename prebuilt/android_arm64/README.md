# Prebuilt Android AArch64 Shared Libraries

Everything in this directory is Git-LFS. A **cargo** checkout of this repo does
not smudge LFS, so the copies there are 132-byte pointer files that link and
then fail at runtime — always package from a host checkout that has run
`git lfs pull`.

## liblitert_lm_c.so

**Built from source**, not an upstream prebuilt, and not interchangeable with
one. Two things about this particular link matter:

1. **The version script.** `c/litert_lm_c.lds` exports the `LiteRt*` symbols as
   GLOBAL dynamic symbols. Without them the GPU TopK sampler's `dlopen` cannot
   resolve `LiteRtCreateEnvironment` / `LiteRtGpuEnvironmentCreate`, the runtime
   silently falls back to the CPU sampler, and decode loses ~40% (see below).
   Check with:

   ```bash
   readelf --dyn-syms -W liblitert_lm_c.so | grep -c ' LiteRt'   # expect 400+
   ```

2. **16 KB pages.** Linked with `-Wl,-z,max-page-size=16384`, which Android 15+
   devices require. Check with:

   ```bash
   readelf -lW liblitert_lm_c.so | awk '/LOAD/{print $NF}'       # expect 0x4000
   ```

To rebuild it:

```bash
bazel build --config=android_arm64 \
  --linkopt=-Wl,-z,max-page-size=16384 \
  --host_linkopt=-Wl,-z,max-page-size=16384 \
  //c:liblitert_lm_c.so
```

Bazel sanitises `PATH`, so the NDK's `ld.lld` is invisible to *host* actions and
the build dies in `collect2: cannot find 'ld'`. `bfd` is not a workaround — it
has no `--start-lib`. Expose lld through **both** `--action_env` and
`--host_action_env`. The cognee-android repo's `scripts/build-litert.sh` does
all of this, caches on the source revision, and re-checks both properties above.

## libLiteRtTopKWebGpuSampler.so

**Modified** the same way as the OpenCL sampler below, and for the same reason.
Which of the two the runtime dlopens depends on the delegate it picked, and
inside an Android *app* process that is WebGPU, never OpenCL: the runtime's
`dlopen` of `libOpenCL.so` goes through `libvndksupport.so`, which is not in the
NDK public library list, so the linker namespace refuses it. (A
`/data/local/tmp` binary has no such restriction and does get OpenCL — which is
why the OpenCL patch mattered on the benchmark first.) Both ship.


## libLiteRtTopKOpenClSampler.so

**Modified** (2026-03-19) from the upstream prebuilt to add `liblitert_lm_c.so`
as a NEEDED ELF dependency.

### Why

The upstream prebuilt `libLiteRtTopKOpenClSampler.so` uses `R_AARCH64_GLOB_DAT`
relocations for LiteRT runtime symbols (`LiteRtCreateEnvironment`,
`LiteRtGpuEnvironmentCreate`, etc.). These relocations must be resolved at
`dlopen()` load time (even with `RTLD_LAZY`).

When loaded from a **statically-linked C++ binary**, all LiteRt symbols are
globally visible — the sampler loads and the GPU TopK sampler works (~6ms/token).

When loaded from a **dynamically-linked Rust binary** that uses `liblitert_lm_c.so`,
Android's linker namespace isolation prevents the sampler from seeing those
symbols, causing `dlopen` to fail with `cannot locate symbol "LiteRtCreateEnvironment"`.
The runtime then falls back to the CPU TopK sampler (~23ms/token), causing a
~40% performance regression.

### Fix

Two complementary changes:

1. **`vendor/LiteRT-LM/c/litert_lm_c.lds`** — exports `LiteRt*` symbols from
   `liblitert_lm_c.so` as GLOBAL dynamic symbols (visible in `.dynsym`).

2. **ELF binary patch on this file** — the prebuilt SO's `DT_SONAME` entry
   (which was `"libLiteRtTopKOpenClSampler.so"`) was repurposed as a
   `DT_NEEDED` entry pointing to `"liblitert_lm_c.so"`.  
   The SONAME string in the `.dynstr` section (file offset `0x3c84`) was
   overwritten with `"liblitert_lm_c.so\0"` (padded with nulls to fill the
   original 30-byte slot).

Together these ensure the Android linker loads `liblitert_lm_c.so` as a
dependency before resolving the sampler's `R_AARCH64_GLOB_DAT` relocations,
resolving all symbol lookup failures.

### Result

| | Before | After |
|--|---|---|
| GPU sampler loads | ❌ (CPU fallback) | ✅ |
| SampleToken (steady) | 17–29ms | ~7ms |
| Decode speed | ~16–17 tok/s | ~24 tok/s |

### Reproducing the patch

The patch script is at `tools/patch_sampler_so.py` in this repo.
Run it if the upstream prebuilt is ever refreshed:

```bash
python3 tools/patch_sampler_so.py
```
