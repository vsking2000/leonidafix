Claimed in the coordination issue as: SHADER-CLAMP-NAN-F64-PK

Depends on: none

Conflicts with: #1112 — both rewrite the `clamp_mods_f64` docstring and the `v_add_f64` output-modifiers row of `docs/design.md`; whichever lands second rebases. (`tests/test_f64_output_modifiers.py`, `tests/test_f64_transcendentals.py` and `tests/test_f64_diff.py` merge without a textual conflict, and they pass on a trial merge of the two.)

### What and why

Since #642, `Gen.clamp_mods` (the f32/f16 VOP3 `clamp`) has replaced a NaN with +0 before `FClamp`, following DX10_CLAMP=1, the game's MODE. GLSL.std.450 leaves `FClamp` undefined for a NaN operand. Two paths still used a bare `FClamp`, and `docs/design.md` listed them as open ("Still a bare `FClamp`: the f64 clamp (`clamp_mods_f64`) and the packed `v_pk_*` clamp"):

- `Gen.clamp_mods_f64`: `v_add_f64`, `v_trunc/ceil/floor/rndne_f64`, `v_rcp/rsq/sqrt_f64`, `v_cvt_f64_i32/u32`, `v_frexp_mant_f64`, `v_div_fixup_f64`, `v_div_fmas_f64` and the other f64 ops with `clamp`.
- The packed `v_pk_add/mul/fma/min/max_f16 … clamp`, per half.

Both now select +0 for a NaN before `FClamp`, the same way `clamp_mods` does (omod first, then the NaN select, then `FClamp`). The rule is the RDNA2 ISA's MODE.DX10_CLAMP ("clamp NaN to zero" in the vector ALU), applied as it already is to the measured f32 and f16 ops. The f64 and packed forms themselves have not been measured on hardware, and `docs/design.md` and the test docstring say so. The sign of a -0 result is still left to the driver, as for f32.

`tools/f64_diff.py`'s `v_add_f64` reference passed a NaN through `clamp`, although its `HW_MODE` has `dx10_clamp: 1`. It now gives +0 too. Its `fclamp-nan-zero` known-divergence entry no longer described a divergence; it is now `f64-clamp` and says what is true: f64 `clamp` programs stay excluded by default because `FClamp` leaves the sign of a -0 result to the driver (the CPU interpreter keeps -0 where the reference gives +0), and the `--known` case is a NaN, which the reference and the translator agree on. `docs/shader_validation.md` says the same.

### Tested

- `tests/semantic/test_vop3_clamp_nan.py` gets `ClampNanF64Tests` and `ClampNanPackedF16Tests`: NaN operands (quiet, negative, signalling), invalid operations (inf + -inf, inf * 0 in one half), the other half kept, no change without `clamp`, clamp controls, `v_pk_min/max_f16` with NaN halves (a half is a NaN only when both operands are), and `v_frexp_mant_f64` and `v_div_fixup_f64` (NaN denominator) as other callers of `clamp_mods_f64`. `v_cvt_f64_i32/u32` cannot produce a NaN, so they are not tested for one. On `main`, 29 of these cases error with `UnsupportedConstruct: GLSL.std.450 FClamp with these operands (undefined)` (27 subtests and the two invalid-operation tests); with this change they pass (21 tests, all OK).
- The SPIR-V data-flow tests that pinned the bare `FClamp` now expect the `OpIsNan`/`OpSelect` of +0.0 in front of it: `tests/test_f64_output_modifiers.py`, `tests/test_f64_transcendentals.py` (including `spirv-val`) and `tests/semantic/test_int32_to_f64.py`.
- `tests/test_f64_diff.py` (33 tests) covers the reference clamp for NaN, -0, +-inf and in-range values, and the `f64-clamp` known case (skipped by default, a match with `--known`, +0). `python -I tools/f64_diff.py --ops v_add_f64 --programs 0 --no-sweep --known --all-failures`: 4 match, 0 mismatch.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 470 files, 6,174 tests. The only failure is `test_int32_to_f64.test_constant_folded_mrt_words`, where `spirv-opt -O` (SPIRV-Tools v2025.1, Ubuntu) crashes with SIGSEGV. It fails the same way on `main` in this environment.
- `python -m pytest` on the 31 `tests/*.py` files that `import pytest`, minus `test_translate.py[ps_sdwa_f16_const.asm]` (needs LLVM 21+ `llvm-mc`; this machine has LLVM 18, and it fails the same way on `main`): 1,786 passed, 14 skipped. The other `tests/*.py` files (everything but `tests/semantic`) pass too: 9,651 passed, 246 skipped.
- `spirv-val --target-env vulkan1.2` passes on a translated `v_pk_fma_f16 … clamp` and `v_add_f64 … mul:2 clamp`.
- `python -m compileall -q make_game.py tools tests`, `scripts/check_no_game_data.py` on the changed files and `git diff --check` pass. A trial merge with current `main` is clean.
- No shader corpus was available, so there are no before/after translation counts. The change only adds an `OpIsNan` and an `OpSelect` in front of an existing `FClamp`, on paths that already translated.

GPUs tested: n/a (CPU interpreter only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
