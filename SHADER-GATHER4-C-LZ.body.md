Claimed in the coordination issue as: SHADER-GATHER4-C-LZ

Depends on: none

Conflicts with: none textually, checked with `git merge-tree` against `origin/shader-sample-grad-forms` (the same fork's sampling branch edits the neighbouring `image_op` dispatch list and the sampling rows of `docs/design.md`): this change dispatches its two ops from a separate `elif` placed before that list, so the two branches do not touch the same lines, and the new row sits after "Typed 2D gathers". Whichever lands second should still re-run `git merge-tree`.

### What and why

`image_gather4_c_lz` and `image_gather4_c_lz_o` (the depth-compare gather at the base level) were refused by name. Each is now one `OpImageDrefGather %v4float` on the sampled image and comparison sampler `image_sample_c_lz` already uses, so the compare function is the sampler's (`runtime/src/gfx10_descriptors.h` builds `compareOp` from S# word 0 DEPTH_COMPARE_FUNC [14:12], the `VkCompareOp` values). The new code is `Gen.emit_gather4_c_lz`; the gather result, offset and `d16` handling are the ones `image_gather4_lz(_o)` has, the compare operand the ones `image_sample_c_lz` has.

| part | rule | source |
|---|---|---|
| address order | after the optional offset dword: z-compare (f32, also under `a16`), s, t, and the slice of an array or cube | LLVM `IntrinsicsAMDGPU.td` (`AMDGPUSampleVariant`: "{offset} {bias} {z-compare}", then the coordinates, no LOD for `_lz`; gather4 variants are `AMDGPUSampleVariantsNoGradients`); `llvm-mc -mcpu=gfx1030` (LLVM 18) takes 3 dwords for 2D, 4 for array and cube, +1 for `_o`, and 2, 3 and +1 under `a16` (z, then the coordinates packed in halves); Mesa ACO `visit_tex` pushes offset, bias, compare, derivatives, coordinates, with the compare whole under a16 and no LOD word for level zero; the existing depth-compare samples row |
| base level | `_lz` reads mip 0; `OpImageDrefGather` reads the base level | LLVM rewrites `image_gather4_c_l` with a constant LOD of zero or below to `IMAGE_GATHER4_C_LZ` (`AMDGPUInstCombineIntrinsic.cpp`, "Optimize _L to _LZ when _L is zero"; `MIMGInstructions.td`, `MIMGLZMapping`) |
| results | four comparisons, 0.0 or 1.0, in the gather order `image_gather4_lz` uses; `d16`: two registers, two comparisons each, low half first | LLVM `lowerImage` (`DMaskLanes = Gather4 ? 4 : popcount(dmask)`, `NumVDataDwords = (DMaskLanes + 1) / 2` for d16); `image_gather4_lz` (GPU-checked order) |
| `_o` | first dword: signed 6-bit x [5:0], y [13:8], as the `Offset` operand (`shaderImageGatherExtended`, not maintenance8); `--texel-offset-fallback` does not apply | the "Texel offsets" row, `image_gather4_lz_o`; Mesa ACO builds the dword as `(offset & 0x3F) << (8 * i)` |
| dimensions | 2D, 2D array, cube (through the 6-layer 2D-array view and `cube_coords`, as `image_gather4_lz`) | LLVM defines gather4 for 2D, Cube and 2DArray only |
| stages | any stage (a gather has no implicit LOD) | `OpImageDrefGather` is not an implicit-LOD operation |

**Refused**, each with a message: 1D and 3D; a `dmask` other than `0x1`; `tfe`, `lwe`, `unorm`, `glc` and any other modifier. The siblings `image_gather4_c_l` and the rest accept any single `dmask` bit but always compare channel 0, and `llvm-mc` accepts any one bit; a Dref gather compares the depth channel only, and Mesa's ACO sets `dmask = 1` for every shadow gather (`if (instr->is_shadow) dmask = 1; else dmask = 1 << instr->component`), so the new forms accept only `dmask:0x1` instead of silently ignoring the other bits. The status dword (`tfe`/`lwe`) and unnormalised coordinates are not implemented for this form.

**Where this differs from its siblings**: the explicit-LOD compare gathers (`image_gather4_c_l`, `_cl`, `_b` and their `_o` forms) emulate the gather with four `OpImageFetch` and decode the compare function from the S# in the shader, because core Vulkan has no gather with a LOD. For `_lz` that emulation is not needed, so the compare function comes from the sampler, as for `image_sample_c_lz`.

**Mesa ACO was read** (`visit_tex`, `nir_texop_tg4`, in two copies from a Mesa mirror, about 2021 and recent) and agrees on every point above: a shadow `tg4` at level zero selects exactly `image_gather4_c_lz` / `image_gather4_c_lz_o`, the address order, the offset packing, the whole-dword compare under a16 and dmask 1. Its half-texel / integer-coordinate gather workaround is for GFX8 and older integer results only and does not touch a compare gather on gfx10. **Not read:** the RDNA2 ISA text (the RDNA3, RDNA4 and CDNA3 pseudocode extractions have no MIMG instructions). **Not GPU-checked**: a Dref gather on a depth view, the cube form and the `d16` form have no hardware or driver measurement here; the pieces rest on the siblings' GPU checks (`ps_sample_c_lz` for the compare sampler, `ps_gather_wrap` for gather order). The op is not known to be in the game's listings (the README reports all 26,503 shaders translating, and these two were refused), so how many game shaders use it is not counted; no shader corpus was available.

`docs/design.md` has one new contract row, "`image_gather4_c_lz`, `image_gather4_c_lz_o`", with the sources above and what was not read or measured. No other row listed these two ops as refused.

### Tested

- New `tests/semantic/test_gather4_c_lz.py` (16 tests). Each form is translated and every operand of the emitted `OpImageDrefGather` is traced back to the VGPR (and half) it was read from, against positions written by hand from LLVM and the `llvm-mc` dword counts: 2D, 2D array, cube, `_o`, `a16` (z whole, coordinates packed), and the `Offset` fields (signed 6 bits at 0 and 8 of the first dword). It also checks that the four results land in components 0..3 of the gather result in order, the `d16` packing (two registers, integer packing), that the module has `OpImageDrefGather` and no `OpImageGather`/`OpImageFetch`/`OpImageQuery*`, `OpCapability ImageGatherExtended` only for `_o`, translation in the pixel, vertex and compute stages, that `--texel-offset-fallback` leaves the gather `Offset` alone, every refusal, and `spirv-val --target-env vulkan1.2` on every form (2D, array, cube; plain, `a16`, `d16`; `_o` and not; pixel and compute stage).
- Mutations that fail the file: reading the z-compare last (5 failures), dropping the `Offset` (3), reversing the result components (1), removing the `dmask` refusal (5), reading the offset's y field from bit 6 (2).
- `python -m compileall -q make_game.py tools tests`: ok.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 477 files, 6,239 tests; the only failure is `test_int32_to_f64.test_constant_folded_mrt_words` (`spirv-opt -O`, SPIRV-Tools v2025.1, SIGSEGV; it fails the same way on `main` here).
- `python -m pytest` on the `tests/*.py` files that `import pytest`: 1,794 passed, 14 skipped, 3 failed: `test_translate.py[cs_dpp_cmp.asm]` (twice) and `[ps_sdwa_f16_const.asm]`, which need a newer `llvm-mc` than 18 and fail the same way on `main`.
- Standalone: `tests/test_design_contract_rows.py`, `test_docs_design_rows.py`, `test_semantic_translator_use.py`, `test_sampler_source_census.py`, `test_isa_coverage.py`, `test_shader_ir.py`, `test_f64_diff.py`, `test_text_write_line_endings.py`: all pass; `test_image_gather4_variants.py` (which keeps these two ops out of `GATHER4_LOD_OPS`) passes unchanged.
- `scripts/check_no_game_data.py` on the changed files and `git diff --check`: pass. A trial merge with current `main` is clean.

GPUs tested: none (CPU translation and `spirv-val` only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
