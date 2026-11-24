Claimed in the coordination issue as: SHADER-F64-VALU-OPS

Depends on: none in code. The f64 results here follow the same omod rule as #1112 (clamp applies, the omod is ignored), but both PRs touch `tools/rdna2spirv.py` (the end of the f64 `v_add_f64`/`v_rcp_f64` branch, where the `F64_VALU_OPS` dispatch sits) and `docs/design.md`, so they conflict textually; whichever lands second rebases.

### What and why

The translator refused seven binary64 VALU ops by name (`Unsupported`): `v_mul_f64`, `v_fma_f64`, `v_min_f64`, `v_max_f64`, `v_cvt_f64_f32`, `v_cvt_i32_f64` and `v_cvt_u32_f64`. They now translate, in one new method, `Gen.f64_valu`:

- `v_mul_f64`: `OpFMul %double`.
- `v_fma_f64`: one fused multiply-add through the existing `Gen.emit_fma64` (one rounding; `--fma khr` gives `OpFmaKHR`, as for `v_div_fmas_f64`).
- `v_min_f64` / `v_max_f64`: `NMin`/`NMax`, then an `OpSelect` of S0 when S1 is a NaN. This is the rule measured for the f32 and f16 forms ("Float min/max/med3 with NaN": one NaN gives the other operand, two give S0). The f64 forms themselves are not measured.
- `v_cvt_f64_f32`: `OpFConvert`, which is exact. Under the game's MODE (f32 denormals flushed), a denormal source reads as a zero of its sign (`Gen.flush_f32`). That is by analogy with the f32 denormal-input rule measured for `v_floor/v_ceil/v_fract_f32`; it is not measured for this conversion.
- `v_cvt_i32_f64` / `v_cvt_u32_f64`: truncate toward zero, saturate out-of-range values and infinities, and give 0 for a NaN, as the f32 forms do (the saturation and NaN wording for the f64 opcodes is taken from the ISA description; not GPU-checked). A NaN is replaced with 0, then `FClamp` to the exact double bounds keeps `OpConvertFToS`/`OpConvertFToU` in range.

Source `abs`/`neg` apply to the double (to the f32 for `v_cvt_f64_f32`); for `v_min_f64`/`v_max_f64` they are moves of the sign bit in the high word, so a signalling NaN is not quieted before the compare, as `src_f_sign_mods` does for the f32/f16 forms. Output modifiers: only `clamp` reaches the f64 results (`clamp_mods_f64`); `mul:2`/`mul:4`/`div:2` are accepted and ignored, because an f64 result ignores the omod while f64 denormals are kept (the game's MODE), the per-result-precision gate of `tools/f64_omod_probe.py` measured on gfx1036 for `v_cvt_f64_i32/u32`, `v_frexp_mant_f64` and `v_div_fixup_f64` (the base's `v_cvt_f64_i32` path already drops it). It is not measured for these five ops. The integer results take no output modifier, and one is refused. `v_mul_f64`/`v_fma_f64` are refused in a program that writes MODE.FP_ROUND, since only `v_add_f64` models the directed f64 rounding modes.

Supporting changes:
- `tools/shader_ir.py` flags hex literal sources of the six f64-source ops as f64 literals (bits 63:32), like the existing f64 ops.
- `tools/f64_diff.py --list-ops` no longer lists the six now-translated ops in its "Excluded (not translated by main)" line.
- `tests/semantic/spirv_mini_interp.py` gets `OpFConvert` (binary32 <-> binary64).
- `docs/design.md` gets a contract row, "Binary64 multiply, FMA, min/max and conversions", and the "f64 literal sources" row lists the new ops.
- `tests/semantic/test_valu_cvt_int_f64.py` asserted that the two integer conversions are refused, and said it should be replaced by a run through the translator once they were implemented. It now is that run.

### Tested

- New `tests/semantic/test_valu_f64_ops.py` (21 tests) and the rewritten `tests/semantic/test_valu_cvt_int_f64.py` (8 tests) run each op through the translator and the CPU interpreter against hand-written expected words. They cover: products and overflow; the fused `a*a - (1 + 2^-29)` = 2^-60 case, which a two-rounding FMA gives as 0; NaN, signalling-NaN and signed-zero min/max cases (the -0 < +0 order follows the measured f32 rule and is not measured for f64), and neg/abs on a NaN S0; exact widening; denormal flush; truncation, saturation and NaN -> 0 for both integer conversions; source modifiers; hex literal, inline-float, inline-integer and SGPR-pair sources; that `mul:2`/`mul:4`/`div:2` scale nothing on each of the five f64-result ops, alone and with `clamp`; and the refusals.
- `spirv-val --target-env vulkan1.2` passes on a translated program for each of the seven ops, with an omod and clamp present on the f64-result ones.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 471 files, 6,192 tests. The only failure is `test_int32_to_f64.test_constant_folded_mrt_words`, where `spirv-opt -O` (SPIRV-Tools v2025.1, Ubuntu) crashes with SIGSEGV. It fails the same way on `main` in this environment.
- The pytest files (those that `import pytest`): 1,786 passed, 14 skipped, no failures. `test_translate.py[ps_sdwa_f16_const.asm]` is deselected: it needs LLVM 21+ `llvm-mc`, this machine has LLVM 18, and it fails the same way on `main`. The 39 standalone `tests/test_*.py` that mention f64, `shader_ir` or `design.md` (including `test_design_contract_rows.py`, `test_docs_design_rows.py` and `test_f64_diff.py`) pass; the others were not run.
- `python -m compileall -q make_game.py tools tests` and `scripts/check_no_game_data.py` on the changed files pass.
- No shader corpus was available, so there are no before/after counts. None of these ops is known to be in the game's listings (the README reports all 26,503 shaders translating), so this extends the translator's coverage of the ISA rather than unblocking a shader.

GPUs tested: n/a (CPU interpreter only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
