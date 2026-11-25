Claimed in the coordination issue as: SHADER-SAMPLE-C-L-FORMS

Depends on: none

Conflicts with: the unmerged branch `shader-sample-grad-forms` (SHADER-SAMPLE-GRAD-FORMS, no pull request yet): one textual conflict in `tools/rdna2spirv.py`, the tuple of sample kinds that accept an offset (this adds `"_c_l"`, that adds `"_d", "_c_d"`; the resolution is the union of both). `docs/design.md` merges without a conflict. Both refuse a compare sample on a 3D image, with their own message. No conflict with #110, #154 or #225 (checked with `git merge-tree`).

### What and why

`image_sample_c_l_o` was refused by name, and `image_sample_c_l` was refused outside scalar 2D by one message ("needs scalar 2D depth comparison") that covered the dimension, the component mask and the modifiers. Both are now the explicit-LOD compare path `image_sample_c_lz` already uses, with the LOD taken from the address: `OpImageSampleDrefExplicitLod` with `Lod`, plus `Offset` for `_o`.

What was refused and why, as found: the code comment says "Vulkan Dref returns one scalar. Other component masks require the depth view's swizzle", which is about `dmask`; the dimension and modifier limits had no stated reason beyond coverage. A Dref sample on a 1D, 2D array or cube image is valid Vulkan and `image_sample_c_lz` already emits it, so those follow. `dmask` other than X still has no source for the other channels and stays refused.

| | |
|---|---|
| Forms | `image_sample_c_l`, `image_sample_c_l_o`: 1D, 2D, 2D array, cube (a cube reads its 6-layer array view, as the other cube samples do); any stage; `a16`; `d16` |
| Address | `[offset,] z-compare, coordinates, LOD`; under `a16` the offset and the z-compare keep a dword each and the coordinates and the LOD pack in f16 pairs (2D: 3 VGPRs, 1D: 2, array and cube: 3, plus the offset dword for `_o`) |
| Offset | the other `_o` forms': signed 6-bit fields x [5:0], y [13:8]; a 1D image takes x only, as a scalar `Offset`; image operands `Lod\|Offset` |
| `--texel-offset-fallback` | the fold reads the mip level of the explicit LOD, rounded by the S#'s MIP_FILTER as for `image_sample_l_o` (before: `_c_l_o` did not exist; a naive `_lz`-style fold would read level 0) |
| Refused | 3D (Vulkan has no `Dref` form: `spirv-val` reports VUID-StandaloneSpirv-OpImage-04777); `dmask` other than `0x1`; `tfe`, `lwe`, `glc` and any other modifier; an address of another width |

Sources, which agree (cited in the `docs/design.md` row):
- LLVM AMDGPU `IntrinsicsAMDGPU.td`: `AMDGPUSampleVariant` (the extra address arguments are `{offset} {bias} {z-compare}`, then the coordinates, then the `lod` appended to them) and `AMDGPUSample_c_l_o` (the `_c_l` variant with the offset argument in front); `SIISelLowering.cpp` `lowerImage` (the extra arguments first, each a dword, then the gradients, then the coordinates up to `VAddrEnd`, which includes the lod, packed in pairs under a16).
- Mesa ACO `visit_tex`: `args` = offset, bias, compare, derivatives, then the coordinates, which were gathered with the lod and the clamp after them and packed by `emit_pack_v1`; the offset fields are `(offset[i] & 0x3f) << (8 * i)`; `image_sample_c_l` / `_c_l_o` is the opcode for a compare with a LOD.
- `llvm-mc` (gfx1013 and gfx1030) accepts each form at exactly these widths and rejects one register fewer (a third check of the dword counts, in the test).
- The repo's own rows for the siblings `image_sample_c_lz` / `_c_lz_o`, `image_sample_l_o` and `image_sample_c` (`docs/design.md`).

I found no disagreement between them on anything that changes a result.

Unmeasured: nothing here was run on a GPU. The siblings are GPU-checked in 2D only (`ps_sample_c_lz`, `ps_texoff`); array, cube and 1D compare samples, the `a16` packing of the LOD, and the dword counts are checked against the compilers and the assembler, not on hardware. What the channels after X of a compare sample hold is not known, which is why `dmask` stays refused.

### Found, not changed here (one topic)

`image_sample_c`, `_c_b`, `_c_cl`, `_c_lz` and their `_o` forms emit invalid SPIR-V for a 3D image instead of refusing it: `spirv-val` rejects all six (VUID-StandaloneSpirv-OpImage-04777). They also ignore `dmask` and `tfe`/`lwe`. Posted as a `finding:`.

### Tested

- New `tests/semantic/test_sample_c_l_forms.py` (12 tests): for 16 forms (c_l and c_l_o, in 1D, 2D, 2D array and cube, with and without a16), every operand of the emitted instruction is followed back to its VGPR (and f16 half) and compared with positions written by hand from the two compilers: the z-compare, each coordinate, the LOD and the offset fields (bit 8k, 6 bits, signed); the sampled-image type of each dimension; the image operand order `Lod` / `Lod|Offset`; the scalar offset of a 1D image; the d16 result as an f16 in the low half; explicit LOD in the vertex and compute stages; 13 refusals (3D, dmask 0x2/0x3, tfe, lwe, glc, a missing or extra address register, a16 widths) with their messages; `spirv-val --target-env vulkan1.2` on every form with and without d16 (the `_o` ones with `--allow-offset-texture-operand`, and without it under `--texel-offset-fallback`); `llvm-mc` accepting each form at its width and rejecting one register fewer; and under the fallback, that the level chain reads the LOD register (also under a16, as the low half of its dword), that a filter without mips reads level 0 and that `_lz` stays at level 0.
- `tests/test_texel_offsets.py`: two forms added to its offset-field and fallback tests, and the folded level of `image_sample_c_l_o` for seven MIP_FILTER / LOD cases (72 tests in the file).
- `tests/test_translate.py`: the refusal test of the unmodelled `image_sample_c_l` forms now matches "depth comparison" and has a `tfe` case.
- Mutations, each failing the new tests (or `test_texel_offsets.py`): the LOD read from the first address word (22 failures); the fold at level 0 for `_c_l` (2 + 1); `_c_l` removed from the kinds that take an offset (39 + 1); the 3D guard removed (3); the dmask check removed (4); the a16 width formula off by one (8 + 1); the z-compare packed under a16 (9); the offset dropped from the Dref sample (10 + 1).
- `python -m compileall -q make_game.py tools tests`: ok.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 477 files, 6,235 tests; the only failure is `test_int32_to_f64.test_constant_folded_mrt_words` (`spirv-opt -O`, SPIRV-Tools v2025.1, SIGSEGV; it fails the same way on `main` here).
- `python -m pytest` on the 31 `tests/*.py` files that `import pytest`: 1,808 passed, 14 skipped, 3 failed (`test_translate.py[cs_dpp_cmp.asm]` twice and `[ps_sdwa_f16_const.asm]`; they need a newer `llvm-mc` than 18 and fail the same way on `main`).
- Standalone: `tests/test_design_contract_rows.py`, `test_docs_design_rows.py`, `test_text_write_line_endings.py`, `test_ci_registration.py`, `test_semantic_translator_use.py`: pass.
- `scripts/check_no_game_data.py` on the changed files and `git diff --check`: pass. A trial merge with current `main` is clean.

GPUs tested: none

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
