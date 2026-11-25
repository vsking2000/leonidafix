Claimed in the coordination issue as: SHADER-TRIG-PREOP

Depends on: none

### What and why

The translator refused `v_trig_preop_f64` by name. It now translates when S1 is an integer constant with `S1[4:0] <= 2`, following the ISA pseudocode:

- `shift = S1[4:0] * 53`, plus `exponent(S0) - 1077` when S0's biased exponent field is above 1077;
- the result is the 53 fraction bits of 2/pi after the first `shift` ones, read as an integer, times `2^(-53 - shift)`, times a further `2^128` when the field is at least 1968;
- only S0's exponent field is read, so source `abs`/`neg` are accepted as no-ops, and zero, denormal, inf and NaN S0 use their field as it is.

`Gen.trig_preop_f64` builds the result with 32-bit integer ops only (no `Float64`, no `Int64`):
- The 2/pi words sit in a Private constant array (`%preop_tab`, 36 words, emitted the way `%fmt_size_tab` is).
- `two_over_pi_words` derives them when translating, from exact integer bounds on pi (Machin's formula). Both bounds must give the same floor, so the bits are proven, not approximated.
- The 53-bit window is read from three words, normalised with `FindUMsb` and packed straight into the exponent and mantissa fields.
- In the translated domain every result is a normal double (checked over all 3 x 2048 segment/exponent cases), so nothing is rounded. MODE.FP_ROUND and the denormal mode make no difference.

Where each part comes from:
- **The ISA text:** as it reads in the RDNA3, RDNA4 and CDNA3 documents (identical in all three; read in the generated per-architecture pseudocode tables shipped with the tinygrad project; their upstream path is not given because it was not confirmed). **The RDNA2 document's wording was not read.**
- **The window expression:** the text writes it as `(1201'B(2.0 / PI)[1200 : 0] << shift) & 0x1fffffffffffff`, which taken literally would give the constant's lowest 53 bits. I read it as the 53 bits after the first `shift`, the only reading that agrees with:
  - the text's own `scale = -53 - shift`;
  - LLVM's constant folding of `llvm.amdgcn.trig.preop` (`llvm/lib/Target/AMDGPU/AMDGPUInstCombineIntrinsic.cpp`);
  - the tinygrad project's VOP3 test `TestTrigPreop`, whose emulator reads the same window. It is in tinygrad's `hw` suite but runs on its emulator unless `USE_HW=1`, the emulator implements the same pseudocode, and it compares with a tolerance, so it is a third derivation of the same window, not hardware evidence.
- **Bit-exact external check:** LLVM's constant-folding test, `llvm/test/Transforms/InstCombine/AMDGPU/amdgcn-intrinsics.ll` (the `trig_preop_constfold*` functions). These are compile-time folds with exact expected words, not measurements. The translator reproduces all 22 of them that have S1[4:0] <= 2 (checked through the CPU interpreter, and pinned as literals in the new test); the others expect 0 for an out-of-range segment and are refused here.
- **Refused, because the sources disagree:**
  - S1[4:0] >= 3, or an S1 that is not an integer constant. LLVM's folding returns 0 from a shift of 1120 ("outbound segment (hardware behavior)"), while the text's 1201-bit constant still has bits there. With S1[4:0] <= 2 the largest shift is 1076, where the two agree.
  - `clamp` and the omods, which are not measured on this op.
- **Not measured on any hardware:** everything on RDNA2, exact results anywhere, the exponent >= 1968 rescale, special and denormal S0, S1 bits above [4:0], and a 32-bit literal S0.

Supporting changes:
- `tools/shader_ir.py` flags hex literal sources of the op as f64 literals (bits 63:32), like the other f64 ops.
- `docs/design.md` gets a contract row with the sources above and the unmeasured list.

### Tested

- New `tests/semantic/test_trig_preop_f64.py` (21 tests). Its expected words come from a reference written in the test from the pseudocode, with 2/pi from a different series (Gauss's arctangent formula), checked against the published leading hex digits of 2/pi. It does not use the translator's helpers. It covers:
  - every exponent field (0..2047) for each of segments 0, 1 and 2, through the translator and the CPU interpreter. This includes both sides of 1077 and 1968, the largest finite S0, zero, denormal, inf and NaN;
  - sign and mantissa independence, including `abs`/`neg`;
  - SGPR-pair, inline-float, inline-integer and literal S0;
  - S1 values with bits above bit 4 (32..34, 64, 65, hex and negative spellings);
  - LLVM's 22 in-domain fold words as literals (`TrigPreopLlvmFoldWords`);
  - a destination equal to S0, one overlapping it, and two uses in one program;
  - both round modes;
  - inactive lanes in a wave32 compute run;
  - the translator's table against the independent derivation;
  - every refusal (segments 3..31, register/float S1, clamp/omod, operand shapes).

  Mutating the translator (the 1968 or 1077 thresholds, the exponent bias, the 53-bit segment step, the window extraction shift) fails it.
- `spirv-val --target-env vulkan1.2` passes on a translated program, which has no `Float64` or `Int64` capability.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 475 files, 6,215 tests. The only failure is `test_int32_to_f64.test_constant_folded_mrt_words`, where `spirv-opt -O` (SPIRV-Tools v2025.1) crashes with SIGSEGV; it fails the same way on `main` in this environment.
- The pytest files (the 31 test files that import pytest): 1,787 passed, 14 skipped, 4 failed. The same four fail on unmodified `main` (`6558d53`) in this environment: `test_translate.py[cs_dpp_cmp.asm]` (two tests) and `test_translate.py[ps_sdwa_f16_const.asm]`, which need a newer `llvm-mc` than this machine's LLVM 18, and `test_ci_registration.py::test_every_selected_file_exposes_a_test_to_its_runner`.
- `python -m compileall -q make_game.py tools tests`, `tests/test_text_write_line_endings.py`, `tests/test_shader_ir.py`, `tests/test_isa_coverage.py`, `tests/test_f64_diff.py` `tests/test_design_contract_rows.py`, `tests/test_docs_design_rows.py` and `scripts/check_no_game_data.py` on the changed files pass.
- No shader corpus was available, so there are no before/after counts. The op is not in the game's listings (the README reports all shaders translating, and this op was refused), so this extends the translator's coverage of the ISA rather than unblocking a shader.

GPUs tested: n/a (CPU interpreter only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
