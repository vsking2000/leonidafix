Claimed in the coordination issue as: SHADER-SAMPLE-GRAD-FORMS

Depends on: none

### What and why

The translator refused six explicit-gradient sample opcodes by name: `image_sample_c_d`, `image_sample_c_d_cl`, `image_sample_d_o`, `image_sample_d_cl_o`, `image_sample_c_d_o` and `image_sample_c_d_cl_o`. `docs/design.md` listed them as "refused still". Each of their pieces is already a GPU-tested rule of a sibling form:
- the depth compare (`image_sample_c`, `image_sample_c_b`);
- the explicit gradients (`image_sample_d`, `image_sample_d_cl`);
- the `_o` texel offset;
- the min-LOD clamp.

This change combines those pieces, in `Gen.image_op`:
- **Address order:** offset, z-compare, dP/dx, dP/dy, coordinates, clamp.
- **Sample:** `OpImageSampleExplicitLod` (`_d` forms) or `OpImageSampleDrefExplicitLod` (`_c_d` forms, one result: f32 bits, or an f16 in the low half under `d16`), with `Grad`, then `Offset`, then `MinLod`. That is the order SPIR-V requires for image operands.
- **Stages:** explicit gradients need no quad, so the same sample is emitted in every stage.
- **`a16`:** the offset, compare and f32 gradients keep a whole VGPR each and only the coordinates and the clamp are packed. This is the same layout the "Packed sample addresses" row documents for `_d_cl`.

The address order comes from two independent compilers that emit these opcodes:
- LLVM: `IntrinsicsAMDGPU.td`, `AMDGPUSampleVariants`, puts offset then zcompare in the extra address arguments, and `SIISelLowering.cpp` `lowerImage` places the extra arguments, then the gradients, then the coordinates and the clamp.
- Mesa ACO: its texture selection (`aco_select_nir.cpp`) pushes offset, bias, compare, ddx, ddy, then the coordinates, and picks these six opcodes from the same flags.

Refused, each with a stated reason:
- `_c_d` forms on a 3D image: Vulkan has no depth compare there.
- 3D gradient forms with `tfe`/`lwe`, as for `image_sample_d`.
- The `_o` forms under `--texel-offset-fallback`: its coordinate shift needs the sampled mip level, which is not computed from explicit gradients.

Still refused by name: the coarse-derivative `_cd` forms (their LOD is per quad, which `Grad` cannot express) and the g16 forms.

`docs/design.md` gets a contract row with the sources and what is unmeasured. The two rows that listed these opcodes as refused now point to it.

Not run on a GPU. None of these opcodes is in the game's listings (they were refused by name); this extends ISA coverage.

### Tested

- New `tests/semantic/test_sample_grad_forms.py` (13 tests). It follows every SPIR-V sample operand (coordinate, compare, both gradients, offset bit-fields, clamp) back to the VGPR it is loaded from. It compares that with positions written by hand from the address order above, for:
  - all six opcodes in 2D, in a pixel and in a compute shader;
  - 2D array;
  - 1D;
  - 3D `_d_o` with three offset fields;
  - `a16` (whole compare and gradients, packed coordinates and clamp);
  - `d16`;
  - dmask;
  - the image-operand order.

  It also covers the refusals and `spirv-val --target-env vulkan1.2 --allow-offset-texture-operand` on every form, in both stages. Swapping the two gradients or dropping `_c_d` from the a16 word count fails it.
- `tests/semantic/test_sample_c.py` and `test_sample_cl.py`: pass.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 477 files, 6,236 tests; the only failure is `test_int32_to_f64.test_constant_folded_mrt_words` (`spirv-opt -O`, SPIRV-Tools v2025.1, SIGSEGV), which fails the same way on `main` here.
- The pytest files (`grep -l "import pytest" tests/*.py`): 1,794 passed, 14 skipped, 3 failed: `test_translate` `cs_dpp_cmp` (two tests) and `ps_sdwa_f16_const`, which need a newer `llvm-mc` than this machine's LLVM 18 and fail the same way on `main`.
- `python -m compileall -q make_game.py tools tests`, `tests/test_docs_table_rows.py`, `tests/test_text_write_line_endings.py`, `git diff --check` and `scripts/check_no_game_data.py` on the changed files: pass.

GPUs tested: none (CPU structure checks and spirv-val only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
