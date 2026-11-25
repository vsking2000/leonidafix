Claimed in the coordination issue as: SHADER-SAMPLE-G16

Depends on: none

Conflicts with: SHADER-SAMPLE-GRAD-FORMS (pushed, open). Both edit the same "Not translated: ... the g16 forms" sentence of the min-LOD clamp row in `docs/design.md` and add rows next to each other. The code does not overlap: this change adds `Gen.g16_gradients` and two dispatch names, and that one extends the `_d` branch of `image_op`. Whichever lands second needs a trivial rebase. Follow-up once SHADER-SAMPLE-GRAD-FORMS lands: route the six compare/offset g16 forms (below) through the same `g16_gradients` helper.

### What and why

The translator refused all 16 `_g16` sample opcodes by name. This change translates the two whose f32-gradient sibling is translated on `main`: `image_sample_d_g16` and `image_sample_d_cl_g16`. They are the same sample as `image_sample_d` / `image_sample_d_cl`, except that dP/dx and dP/dy are f16, packed two halves per VGPR, low half first, with each axis padded to a whole VGPR:

| Dim | Gradient words |
|---|---|
| 1D | (dx/dh, pad), (dx/dv, pad) |
| 2D, 2D array, cube | (dx/dh, dy/dh), (dx/dv, dy/dv) |
| 3D | (dx/dh, dy/dh), (dz/dh, pad), (dx/dv, dy/dv), (dz/dv, pad) |

The coordinates and the clamp follow. They are f32 words, or packed halves when the instruction also has `a16`: g16 and a16 are independent. `Gen.g16_gradients` widens each gradient half with `UnpackHalf2x16` (the a16 coordinates' widening) and drops the pad halves. The existing f32-gradient path then does the rest unchanged, a16 included.

**Sources** — they agree on every point implemented here:
- **LLVM** `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`:
  - `lowerImage` sets IsG16 from the gradient operand type and IsA16 from the coordinate type. On a G16 target the two are independent, and the `_g16` opcode comes from `getMIMGG16MappingInfo`.
  - `packImage16bitOpsToDwords` packs the gradients; its comment and its code give the 1D, 2D and 3D layout in the table above.
- **Mesa ACO** `src/amd/compiler/instruction_selection/aco_select_nir.cpp`:
  - `a16` and `g16` are set from the bit sizes of the coordinate and of ddx.
  - ddx and ddy go through one `emit_pack_v1` call each. In `aco_isel_helpers.cpp`, that function pairs halves low first and pads an odd one.
  - `separate_g16` selects `image_sample_d_g16` / `image_sample_d_cl_g16` from GFX10 on.

`llvm-mc` 18 encodes `image_sample_d_g16` for both gfx1013 and gfx1030.

PR #633 (MIMG decoder) appears in the `g16` search. It changes only `tools/rdna2_decode.py`'s address-VGPR counts, not the translation. Its g16 count — each derivative axis rounded up to an even number of halves — matches this layout, so this change neither depends on it nor contradicts it.

**Still refused:**
- the six compare/offset g16 forms (`image_sample_c_d_g16`, `_c_d_cl_g16`, `_d_o_g16`, `_d_cl_o_g16`, `_c_d_o_g16`, `_c_d_cl_o_g16`), until SHADER-SAMPLE-GRAD-FORMS translates their f32 siblings;
- the eight `_cd*_g16` forms, because a coarse derivative has no exact Vulkan form.

**Unmeasured:** this has not been run on a GPU, and none of these opcodes is in the game's listings. The f16-to-f32 widening of a NaN or denormal gradient follows `UnpackHalf2x16`, exactly as the a16 coordinates do; on hardware it is unmeasured for both, since the "Packed sample addresses" row only GPU-checks coordinate selection.

`docs/design.md` gets a "16-bit gradients" row saying the same, and the clamp row's "the g16 forms" now names the two exceptions.

### Tested

- New `tests/semantic/test_sample_g16.py` (7 tests). It follows every gradient, coordinate and clamp operand back to the VGPR and the half it is read from, and compares the result with positions written by hand from the layout above:
  - dims: 1D, 2D, 2D array, cube, 3D;
  - with and without the clamp;
  - with and without a16 (a16 + g16 together in 1D, 2D and 3D);
  - in a pixel shader and in a compute shader.

  It also checks the f32 form's register shift, dmask, the refusals of the other 14 forms, a short address, and `spirv-val --target-env vulkan1.2` on every case.
- Mutation checks: each of these makes the tests fail — one VGPR per gradient component instead of packing, reading the high half first, keeping the pad, and mis-counting the words that follow.
- `tests/semantic/test_sample_c.py` and `test_sample_cl.py`: pass.
- `python -I scripts/run_semantic_tests.py tests/semantic`: 477 files, 6,230 tests; the only failure is `test_int32_to_f64.test_constant_folded_mrt_words` (`spirv-opt -O`, SPIRV-Tools v2025.1, SIGSEGV), the same as on `main` here.
- The pytest files (`grep -l "import pytest" tests/*.py`): 1,794 passed, 14 skipped, 3 failed. The 3 failures are `test_translate` `cs_dpp_cmp` (two tests) and `ps_sdwa_f16_const`, which need a newer `llvm-mc` than LLVM 18; they fail the same way on `main`.
- `python -m compileall -q make_game.py tools tests`, `tests/test_text_write_line_endings.py`, `tests/test_docs_table_rows.py`, `tests/test_docs_cli_flags.py`, `git diff --check` and `scripts/check_no_game_data.py` on the changed files: pass.

GPUs tested: none (CPU structure checks and spirv-val only, Linux)

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] A new translator rule is in the contract table of `docs/design.md`, with how it was confirmed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
