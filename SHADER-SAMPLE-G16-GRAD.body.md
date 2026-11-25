Claimed in the coordination issue as: SHADER-SAMPLE-G16-GRAD

Depends on: `shader-sample-grad-forms` (SHADER-SAMPLE-GRAD-FORMS, PR #TBD) and `shader-sample-g16` (SHADER-SAMPLE-G16, PR #TBD). **#TBD: fill in both PR numbers.** This branch merges both on top of `main`, and its own change is the single commit after those merges. Rebase it once both land.

### What and why

SHADER-SAMPLE-G16 translated `image_sample_d_g16` and `image_sample_d_cl_g16` and left the six g16 forms with a compare and/or a texel offset refused until their f32-gradient siblings (SHADER-SAMPLE-GRAD-FORMS) were in. This translates those six: `image_sample_c_d_g16`, `image_sample_c_d_cl_g16`, `image_sample_d_o_g16`, `image_sample_d_cl_o_g16`, `image_sample_c_d_o_g16` and `image_sample_c_d_cl_o_g16`.

`Gen.g16_gradients` now accepts every `SAMPLE_GRAD_FORMS` base op. The `_o` texel-offset dword and then the z-compare come before the packed gradients, and both stay whole VGPRs under a16 and g16. The helper keeps those leading words, widens the packed gradient halves as before, and passes the coordinates through. The f32-gradient compare/offset path then does the rest unchanged, refusals included: compare on a 3D image, a 3D `tfe`/`lwe` gradient form, and the `_o` forms under `--texel-offset-fallback`.

**Sources**, which agree on every point:

1. **Which words stay whole, and where the packed gradients start.**
   - LLVM `SIISelLowering.cpp` `lowerImage` pushes the extra address arguments (`IntrinsicsAMDGPU.td` `AMDGPUSampleVariants`: offset, then zcompare) unchanged before the gradients. Its only special case is a 16-bit bias padded to a dword.
   - Mesa ACO `aco_select_nir.cpp` asserts a 32-bit comparator (`nir_tex_src_comparator`) and offset (`nir_tex_src_offset`), and emplaces offset, bias, compare, then the derivatives.
2. **Gradient packing.** Unchanged from SHADER-SAMPLE-G16: two halves per VGPR, low first, each axis padded.
   - LLVM: `packImage16bitOpsToDwords`.
   - Mesa ACO: one `emit_pack_v1` per ddx/ddy.
3. **What a16 does.** It packs only the coordinates and the clamp.
   - LLVM: `IsA16` packs from `CoordStart` on.
   - Mesa ACO: coordinates, LOD and clamp go through `emit_pack_v1`; the comparator and offset are asserted 32-bit.

**Still refused:** the eight `_cd*_g16` forms. A coarse derivative's LOD is per quad, which `Grad` cannot express.

**Not run on a GPU.** None of these opcodes is in the game's listings. Two existing tests listed `_c_d_g16` / `_d_o_g16` as refused:
- `test_sample_g16.test_other_g16_forms`
- `test_sample_grad_forms.test_still_refused_neighbours`

They now check `_cd` g16 forms instead.

`docs/design.md`:
- The "16-bit gradients" row adds the compare/offset part with its sources.
- The two sentences that listed these forms as refused now point to that row.

### Tested

- New `tests/semantic/test_sample_g16_grad.py` (9 tests). It traces every operand to its VGPR and half: offset bit-fields, compare, both gradients, coordinate, clamp. It compares them with positions written by hand from the order above, covering:
  - all six opcodes in 2D, in a pixel and in a compute shader;
  - a16 + g16 in 1D, 2D and 3D;
  - 1D and a 2D array;
  - 3D with a three-field offset;
  - a cube compare;
  - `d16`;
  - image-operand order;
  - all eight `_cd*_g16` refusals, compare on 3D, the offset-fallback refusal and a short address;
  - `spirv-val --target-env vulkan1.2 --allow-offset-texture-operand` on every case in both stages.
- Mutation checks — each of these makes the tests fail:
  - not keeping the leading words;
  - keeping only the offset word;
  - putting the gradients before the leading words;
  - not dispatching the g16 names.
- `test_sample_g16.py`, `test_sample_grad_forms.py`, `test_sample_c.py`, `test_sample_cl.py`: pass.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 479 files, 6,252 tests. The only failure is `test_int32_to_f64.test_constant_folded_mrt_words` (`spirv-opt -O` SIGSEGV, SPIRV-Tools v2025.1), which fails the same way on `main` here.
- The pytest files (`grep -l "import pytest" tests/*.py`): 1,794 passed, 14 skipped, 3 failed. The failures are `test_translate` `cs_dpp_cmp` ×2 and `ps_sdwa_f16_const`, which need a newer `llvm-mc` than LLVM 18; they fail the same way on `main`.
- These also pass: `python -m compileall -q make_game.py tools tests`, `tests/test_docs_table_rows.py`, `test_docs_design_rows.py`, `test_design_contract_rows.py`, `test_docs_cli_flags.py`, `test_text_write_line_endings.py`, `git diff --check`, and `scripts/check_no_game_data.py` on the changed files.

GPUs tested: none (CPU structure checks and spirv-val only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` plus the two base branches; rebase after they land.
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
