Claimed in the coordination issue as: SHADER-FRACT-F64

Depends on: none

Conflicts with: #1374 (`tools/shader_ir.py`: both append op names to the f64 literal-source tuple; `tools/f64_diff.py`: both shorten the same "Excluded (not translated by main)" line), the `v_trig_preop_f64` pull request from the same fork, branch `shader-trig-preop` (the same `shader_ir.py` tuple), and #1112 (`tools/rdna2spirv.py` and `docs/design.md`; when checked it also conflicted with `main` on its own). All are one-line overlaps with no code dependency; whichever lands second rebases. #1366 (the f64 and packed clamp of a NaN) touches `clamp_mods_f64`, which this uses; no textual conflict.

### What and why

`v_fract_f64` (VOP1 opcode 62, VOP3 opcode 446) was refused by name. It is now `D.f64 = S0.f64 - floor(S0.f64)`: an `OpFSub` of the source and GLSL.std.450 `Floor` on the double, with the edges selected from the operand's words so the driver does not decide them. The rule and what each part rests on are in the new `docs/design.md` row. Summary:

| input | result | source |
|---|---|---|
| finite normal | `x - floor(x)`, rounded once to nearest even | the ISA pseudocode `D0.f64 = S0.f64 + -floor(S0.f64)` (read as it stands in the RDNA3, RDNA4 and CDNA3 documents, identical; **the RDNA2 document was not read**; the text has no cap, inf or NaN rule); rounding inferred from `v_fract_f32`; LLVM's constant fold gives the same words |
| a tiny negative source (1 - tiny rounds to 1.0) | `0x3fefffffffffffff`, the largest double below 1 | `v_fract_f32` (`0x3f7fffff`) and `v_fract_f16` (`0x3bff`), measured on gfx1036; LLVM's constant fold (`fmin(x - floor(x), nextDown(1.0))`) gives the same word. LLVM's `min(V_FRACT_F64(x), 0x3fefffffffffffff)` in its f64 `floor` lowering is for GFX6 (SI) only ("V_FRACT is buggy on SI"), so it is no evidence about gfx10 |
| -0, +0, an exact integer, \|x\| >= 2^52 | +0 | the formula; `v_fract_f32` of -0 is +0 (measured); LLVM's fold gives +0 for both zeros and for +-1. A zero source is selected from the operand's words, so it does not depend on the driver's `Floor(-0.0)` |
| positive subnormal | itself (binary64 subnormals are kept, `denorm_mode_16_64 = 3`) | measured for `v_floor_f64` and the other rounding ops; LLVM's fold gives the same word for 2.0e-308 |
| negative subnormal | `0x3fefffffffffffff` (1 - tiny) | follows from the two rows above; LLVM's fold gives the same word for -2.0e-308 |
| +-inf | `0xfff8000000000000`, the negative quiet NaN | `v_fract_f32` `0xffc00000` and `v_fract_f16` `0xfe00` (measured); `v_div_fixup_f64`'s invalid result. LLVM's fold gives `+qnan` here, a generic compile-time APFloat model, so the row follows the measured words |
| a NaN | unchanged, sign and payload, a signalling one too | `v_fract_f32` and the f64 rounding ops (measured, IEEE = 0); LLVM's fold gives `+qnan` and quiets a signalling one (same model) |
| `abs`, `neg`, literal, inline, SGPR pair | as for the f64 rounding ops | `Gen.f64_source_words` |
| `mul:2`/`mul:4`/`div:2` | ignored | a binary64 result ignores them under `denorm_mode_16_64 = 3`, the game's MODE (`tools/f64_omod_probe.py`; `v_cvt_f64_i32`, `v_frexp_mant_f64`) |
| `clamp` | applied through `clamp_mods_f64`; the fraction is in [0, 1), so only a NaN changes | |

**Refused:** a program that writes MODE.FP_ROUND (`Unsupported: v_fract_f64 under a directed MODE.FP_ROUND`): the result for a negative non-integer is rounded (`-0.1` gives `0x3feccccccccccccd` to nearest and `...ccc` toward zero), whether the instruction follows the mode is not measured, and only `v_add_f64` models it; the sibling f64 ops that round (`v_mul_f64`, `v_fma_f64` in the pull request that adds them) refuse the same way.

**Not measured on hardware for binary64** (nothing here ran on a GPU): the rounding of a negative source, the `0x3fefffffffffffff` cap, the word for +-inf, the quiet bit of a signalling NaN, and subnormal sources. LLVM's `min(V_FRACT, 0x3fefffffffffffff)` workaround is for GFX6 (SI) only, so it says nothing about gfx10; `v_fract_f32` and `v_fract_f16` on gfx1036 already stop below 1, which is what the rule assumes for the binary64 form on the same part. The inputs for a `tools/hw_oracle` run are in the row: -1e-20, -2^-54, +-5e-324, +-inf, `0x7ff0000000000001`, -0.1, 2^52 - 0.5.

Where this differs from the sibling ops on `main`: `v_trunc_f64`, `v_ceil_f64`, `v_rndne_f64` and `v_floor_f64` still pass `mul:2`/`mul:4`/`div:2` to `clamp_mods_f64` there, although the same hardware measurement says the hardware ignores them under the game's MODE. #1112 drops the omod for those four (and for `v_add_f64`, `v_rcp/rsq/sqrt_f64`). `v_fract_f64` follows the measurement and #1112, as `v_cvt_f64_i32/u32` and `v_frexp_mant_f64` already do on `main`, so after #1112 the five rounding-family ops behave alike; until then `v_fract_f64 ... mul:2` and `v_floor_f64 ... mul:2` differ. The module also does not use a shared helper with the sibling branch, to keep that branch (which #1112 edits) untouched.

`clamp` of a NaN is not asserted in the tests: `clamp_mods_f64` leaves it to `FClamp` until #1366 is on `main`. With #1366, `v_fract_f64 ... clamp` of +-inf or a NaN gives +0.

One existing file disagrees with the rule on two edges: `tests/semantic/test_valu_fract_f64.py` models the instruction in Python without the translator (it pins the positive NaN `0x7ff8000000000000` for +-inf and quiets a signalling NaN) and cites no measurement; `docs/design.md` had no `v_fract_f64` row before this. This pull request does not touch it.

Also: `tools/shader_ir.py` flags the literal of `v_fract_f64` as an f64 one, `tools/f64_diff.py` no longer lists it as "not translated by main" (it is not added to the 47 forms it generates; a follow-up can), and `tests/semantic/test_f64_literal_whole.py` runs the new op through its two-dword and wider-than-32-bit literal cases. `docs/isa_coverage.md` is a dated report and is not regenerated here. No shader corpus was available, so how many game shaders use `v_fract_f64` is not counted.

### Tested

- New `tests/semantic/test_fract_f64.py` (23 tests): hand-written expected words for positives, negatives, exact integers, 2^52 and larger, -0/+0, the tiny-negative cap (-2^-60, the tie at -2^-54, -2^-53, -2^-52), subnormals, +-inf, quiet and signalling NaNs with payloads, `abs`/`neg`/`-|x|`, a one-dword literal, a wider literal, inline float and integer constants, an SGPR pair, an overlapping destination pair, an inactive lane, `clamp` on finite sources, ignored omods (and no `OpFMul %double` or `FClamp` without `clamp`), the eight finite f64 inputs of LLVM's `fract.ll` as literals (the +-inf and NaN rows of that file are not pinned: they differ, as above), the refusal under MODE.FP_ROUND (and that the same program without the write translates), and refusals (32-bit destination or source, `_e32` with a modifier, `op_sel`, `_dpp`, `_sdwa`). A driver model that quiets a NaN, reads a subnormal as zero (`driver_that_quiets_and_flushes` of `test_f64_round.py`, from `cs_w32_round_f64`'s measurements) and returns the positive NaN for `inf - inf` leaves the words unchanged; a fourth change of that model, `Floor(-0.0)` returning +0, leaves a zero source at +0. Which tests would pass without the infinity select because of the interpreter's `inf - inf` NaN, and which one does not, is noted in the file. `spirv-val --target-env vulkan1.2` passes with and without `clamp`.
- Mutation check: removing the below-one cap, the subnormal selection, the NaN selection or the infinity selection, applying the omod, dropping `neg`, or using `Ceil` for `Floor` each fails the file (4, 3, 6, 2, 1, 2 and 31 failures); so does removing the zero selection (1 failure, the `Floor(-0.0)` model test) and the MODE.FP_ROUND refusal (1 failure).
- `python -m compileall -q make_game.py tools tests`: ok.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 477 files, 6,246 tests; the only failure is `test_int32_to_f64.test_constant_folded_mrt_words` (`spirv-opt -O`, SPIRV-Tools v2025.1, SIGSEGV; it fails the same way on `main` here).
- `python -m pytest` on the 31 `tests/*.py` files that `import pytest`: 1,794 passed, 14 skipped, 3 failed: `test_translate.py[cs_dpp_cmp.asm]` (twice) and `[ps_sdwa_f16_const.asm]`, which need a newer `llvm-mc` than 18 and fail the same way on `main`.
- Standalone: `tests/test_f64_diff.py`, `test_f64_output_modifiers.py`, `test_f64_transcendentals.py`, `test_f64_omod_probe.py`, `test_gpu_round_f64.py`, `test_design_contract_rows.py`, `test_docs_design_rows.py`, `test_shader_ir.py`, `test_semantic_mutants.py`, `test_semantic_translator_use.py`, `test_text_write_line_endings.py`, `test_isa_coverage.py`, `test_ci_registration.py`: all pass.
- `scripts/check_no_game_data.py` on the changed files and `git diff --check`: pass. A trial merge with current `main` is clean.

GPUs tested: n/a (CPU interpreter only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
