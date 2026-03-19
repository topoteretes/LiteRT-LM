# Prebuilt Android AArch64 Shared Libraries

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
