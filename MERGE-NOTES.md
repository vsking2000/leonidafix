# all-fixes-2: merge notes

Base: upstream/main `b4b7756e`. The branches were merged in this order, each with `--no-ff` and fetched just before
merging. Merge commits carry the session trailers.

| # | Branch | Head merged | Merge commit | Conflict |
|---|---|---|---|---|
| 1 | origin/shader-clamp-nan-f64-pk | d2377d88 | 610d9d91 | none |
| 2 | origin/shader-f64-valu-ops | 1b0442fc | eb300555 | none (docs/design.md, tools/f64_diff.py and tools/rdna2spirv.py auto-merged) |
| 3 | origin/shader-trig-preop | 78b87beb | a5ff577d | tools/shader_ir.py (with #2) |
| 4 | origin/shader-div-scale-probe | a2209ec3 | 16399f0d | none |
| 5 | origin/shader-fract-f64 | 3c359c50 | 56a35008 | tools/shader_ir.py (with #2 and #3), tools/f64_diff.py (with #2) |
| 6 | origin/shader-sample-grad-forms | a768a650 | 8e7a83df | none |
| 7 | origin/shader-sample-g16 | ece08d3b | 606773d7 | tools/rdna2spirv.py and docs/design.md (with #6) |

Tip of `all-fixes-2`: `606773d7`. Worktree:
`/tmp/claude-0/-home-user-leonidafix/4b42ddd9-d548-5497-842f-080a5b1709ef/scratchpad/wt-all-fixes-2`.

On its own, each of the seven branches merges cleanly into the current upstream/main (checked with
`git merge-tree --write-tree upstream/main origin/<branch>`).

## Conflicts between branches, with the resolved text

Each resolution keeps both sides' op names and rows. Whichever pull request lands second applies the resolved text.

### A. tools/shader_ir.py: the f64 literal-source tuple (`parse_line`, the `op.startswith((...))` at lines 228-231)

Branches: shader-f64-valu-ops adds `v_mul_f64`, `v_fma_f64`, `v_min_f64`, `v_max_f64`, `v_cvt_i32_f64` and
`v_cvt_u32_f64`; shader-trig-preop adds `v_trig_preop_f64` after `v_ldexp_f64`; shader-fract-f64 appends
`v_fract_f64`. Any two of the three conflict on the same two physical lines. Resolved, with all three:

```python
    if op.startswith(("v_add_f64", "v_cvt_f32_f64", "v_rcp_f64", "v_rsq_f64", "v_sqrt_f64",
                      "v_frexp_mant_f64", "v_frexp_exp_i32_f64", "v_ldexp_f64", "v_trig_preop_f64", "v_trunc_f64",
                      "v_ceil_f64", "v_rndne_f64", "v_floor_f64", "v_mul_f64", "v_fma_f64", "v_min_f64",
                      "v_max_f64", "v_cvt_i32_f64", "v_cvt_u32_f64", "v_fract_f64")):
```

For a pair, drop the names of the branch that has not landed.

### B. tools/f64_diff.py: the "Excluded (not translated by main)" string (line 774)

Branches: shader-f64-valu-ops removes its six ops, shader-fract-f64 removes `v_fract_f64`. Starting from upstream's
`v_mul_f64, v_min_f64, v_max_f64, v_fract_f64, v_cvt_f64_f32, v_ldexp_f64, v_mad_f64, v_cvt_i32_f64, v_cvt_u32_f64,
v_div_scale_f64`, the two removals together leave:

```python
        print("Excluded (not translated by main): v_ldexp_f64, v_mad_f64, v_div_scale_f64")
```

With only shader-f64-valu-ops: `v_fract_f64, v_ldexp_f64, v_mad_f64, v_div_scale_f64`. With only shader-fract-f64:
`v_mul_f64, v_min_f64, v_max_f64, v_cvt_f64_f32, v_ldexp_f64, v_mad_f64, v_cvt_i32_f64, v_cvt_u32_f64,
v_div_scale_f64`. `tests/test_f64_diff.py` passes on the result: it checks that `v_div_scale_f64` is still listed.
`v_ldexp_f64` is translated on main already and is still listed; neither branch touches it, so I left it.

### C. tools/rdna2spirv.py: the `image_op` dispatch list (`Gen.insn`, line ~8018 on all-fixes-2)

Branches: shader-sample-grad-forms adds `or op in self.SAMPLE_GRAD_FORMS`; shader-sample-g16 adds
`"image_sample_d_g16", "image_sample_d_cl_g16"`. Resolved:

```python
                    "image_sample_d", "image_sample_d_cl", "image_sample_d_g16", "image_sample_d_cl_g16",
                    "image_load", "image_gather4_lz", "image_gather4_l",
                    "image_gather4_lz_o", "image_gather4_l_o") or op in self.SAMPLE_GRAD_FORMS or \
                op in self.GATHER4_LOD_OPS:
```

The other changes both branches make to `image_op` auto-merged: the `_g16` strip at its top, `g16_gradients`, and
the `_c_d` handling.

### D. docs/design.md: the sample rows after "Packed sample addresses" (lines 306-309 on all-fixes-2)

Branches: shader-sample-grad-forms edits a sentence in the "Depth-compare samples with implicit LOD" row and in the
"Min-LOD clamp forms" row, and adds the "Explicit-gradient samples with compare or offset" row after the latter.
shader-sample-g16 adds the "16-bit gradients" row before "Depth-compare ..." and edits the same "Not translated:"
sentence of the "Min-LOD clamp forms" row. Resolved order:
1. "16-bit gradients ..." (from g16, unchanged);
2. "Depth-compare samples with implicit LOD ..." (grad-forms' version);
3. "Min-LOD clamp forms ..." (grad-forms' version, with the shared sentence merged as below);
4. "Explicit-gradient samples with compare or offset ..." (from grad-forms, unchanged).

The shared sentence in the "Min-LOD clamp forms" row:
- upstream: ``Not translated: `_c_d`, `_c_cd`, `_d_cl` with compare, the g16 forms.``
- grad-forms: ``Not translated: `_c_cd` and the g16 forms (`_c_d_cl` and `_d_cl_o` are the "Explicit-gradient samples with compare or offset" row).``
- g16: ``Not translated: `_c_d`, `_c_cd`, `_d_cl` with compare, the g16 forms other than `_d_g16` and `_d_cl_g16` (the "16-bit gradients" row).``
- resolved: ``Not translated: `_c_cd` and the g16 forms other than `_d_g16` and `_d_cl_g16` (the "16-bit gradients" row; `_c_d_cl` and `_d_cl_o` are the "Explicit-gradient samples with compare or offset" row).``

### Pairs with no conflict

These touch the same files without overlapping lines (auto-merged, checked by the tests below):
- clamp-nan-f64-pk with f64-valu-ops: docs/design.md, tools/f64_diff.py, tools/rdna2spirv.py;
- div-scale-probe with everything (it touches only docs/shader_validation.md and new files);
- fract-f64 with clamp-nan-f64-pk;
- the sample branches with the f64 branches.

## Open PR #1112 (`pr1112`, head `b30c7289`)

#1112 already conflicts with the current upstream/main on its own (`git merge-tree upstream/main pr1112`: docs/design.md
and tools/rdna2spirv.py), so it needs a rebase whatever happens to these branches. A trial merge of `pr1112` into
all-fixes-2 (aborted afterwards, nothing committed) conflicts in four places:

1. **tools/rdna2spirv.py, the `clamp_mods_f64` docstring (line ~7081): textual only.** shader-clamp-nan-f64-pk
   (commit 211662ff) documents NaN -> +0 under DX10_CLAMP before `FClamp`. #1112's sentence "a NaN result ... left
   to the driver (the hardware, with DX10_CLAMP set, gives +0 for NaN)" describes the old bare `FClamp`, and its own
   parenthesis records the +0 that the clamp branch implements. #1112's `clamp_mods_f64` body is that bare `FClamp`
   (`git show pr1112:tools/rdna2spirv.py`, lines 6811-6827). Resolution: the clamp branch's docstring and code, plus
   #1112's sentence that the f64 VALU arithmetic and rounding ops hand only `clamp`, because their omod does nothing
   while f64 denormals are kept.
2. **tools/rdna2spirv.py, the f64 rounding ops (line ~8782).** #1112 adds
   `result = self.clamp_mods_f64(result, [m for m in mods if m == "clamp"])`. The other side is upstream/main's own
   commit 26d010b4 (milk4sale), not one of the seven branches. Keep both.
3. **tools/rdna2spirv.py, the `v_add_f64` group (line ~8957): textual only.** #1112 adds a measured comment that the
   omod is dropped. The other side is shader-f64-valu-ops' `elif op in self.F64_VALU_OPS: self.f64_valu(op, a,
   mods)`. That branch already ignores the omod: since its follow-up commit a44423b8, both `dst_f64` calls in
   `f64_valu` pass `[m for m in mods if m == "clamp"]` (`git show origin/shader-f64-valu-ops:tools/rdna2spirv.py`,
   lines 6880 and 6916), which agrees with #1112's measurement. Resolution: keep #1112's comment and the
   `F64_VALU_OPS` branch.
4. **docs/design.md, the f64 rows (line ~389): textual.** These are #1112's rewritten "`v_add_f64` output modifiers"
   and rounding-op rows. On our side the rows come from shader-clamp-nan-f64-pk (the `v_add_f64` row's NaN -> +0
   clamp), shader-f64-valu-ops (the new "Binary64 multiply, FMA, min/max and conversions" row) and shader-fract-f64
   (the new `v_fract_f64` row). Keep every row. For the `v_add_f64` row, use the clamp branch's NaN -> +0 text plus
   #1112's measured omod text, as in point 1.

Branches that do not touch #1112's regions: trig-preop, div-scale-probe, sample-grad-forms, sample-g16. (`merge-tree`
against pr1112 reports conflicts for them too, but only because pr1112's base is older than main's; those conflicts
are #1112 against main, not against the branch.)

## Checks on all-fixes-2 (606773d7)

- `python3 -m compileall -q make_game.py tools tests`: rc=0.
- Every semantic test file the seven branches add or change passes (`test_f64_literal_whole`, `test_fract_f64`,
  `test_pk_f16_clamp`, `test_sample_g16`, `test_sample_grad_forms`, `test_trig_preop_f64`, `test_valu_cvt_int_f64`,
  `test_valu_f64_ops`, `test_vop3_clamp_nan`), except the known `test_int32_to_f64.test_constant_folded_mrt_words`
  (`spirv-opt -O` SIGSEGV in this environment, also on main).
- `python3 -I scripts/run_semantic_tests.py tests/semantic`:
  `481 files, 6323 tests, 1 files failed in 146 s with 4 jobs: test_int32_to_f64`. An earlier run overlapped the
  #1112 trial merge, whose conflict markers broke the tree, and is discarded.
- pytest on the files that import pytest: `3 failed, 1794 passed, 14 skipped`. The three are
  `test_translate.py` `cs_dpp_cmp` x2 and `ps_sdwa_f16_const`, which need a newer `llvm-mc` than LLVM 18.
- `tests/test_design_contract_rows.py`, `test_docs_design_rows.py`, `test_docs_table_rows.py`, `test_f64_diff.py`,
  `test_shader_ir.py`, `test_div_scale_probe.py`, `test_text_write_line_endings.py`, `test_f64_output_modifiers.py`,
  `test_f64_transcendentals.py`: OK. `test_ci_registration.py` (pytest): 16 passed.
- `spirv-val` runs inside those tests: the sample, trig_preop, f64 and clamp tests that call it passed.
