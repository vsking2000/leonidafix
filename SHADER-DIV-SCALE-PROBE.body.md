Claimed in the coordination issue as: SHADER-DIV-SCALE-PROBE

Depends on: none

### What and why

The translator refuses `v_div_scale_f32` and `v_div_scale_f64`. The public descriptions of them disagree at the edges (see the `finding:` on v_div_scale in the coordination issue):
- the ISA pseudocode (RDNA3/RDNA4/CDNA3 wording; the RDNA2 document's wording was not read);
- the D = S0 pass-through LLVM's divide lowering relies on (`SIISelLowering.cpp`, `LowerFDIV32`/`LowerFDIV64`);
- the rule tinygrad's RDNA3 emulator runs after fixes drawn from its hardware tests (`test/mockgpu/amd/emu.py`, `test/amd/hw/test_vop3.py`).

Read literally, the older f32 reading of the text makes LLVM's divide return +inf for 1.0 / 2^127. So the rule has to come from a measurement. This adds the tool for it and no rule.

`tools/div_scale_probe.py` follows `tools/f64_omod_probe.py` on `tools/hw_oracle`: raw one-lane kernels, the kernel's MODE set from its descriptor (the same five MODEs, `game_i0_d3` first), wave32 and wave64, and the same CLI, JSON report and exit codes (77 without an oracle or GFX10 device).

Per width it measures:
- 48 numerator/denominator pairs, each run as LLVM's denominator call `(d, d, n)` and numerator call `(n, d, n)`:
  - special operands: zero, -0, inf, quiet and signalling NaN;
  - denormals;
  - exponent-field gaps 95/96/97 [767/768/769];
  - denominators just below, at and above 2^126 [2^1022], and 2^127 [2^1023];
  - quotients below, at and above 2^-126 [2^-1022];
  - numerator fields 22/23/24 [52/53/54];
  - ordinary operands.
- 10 rows with S0 = S1 = S2.

Each of those 106 rows runs with VCC and with an SGPR mask destination (s10 / s[10:11]), preset to all ones. The probe records D, the lane's flag and the whole mask word. There are also 30 modifier rows (`neg` on each source; `clamp`, `mul:2`, `div:2`). VOP3B has no `abs`. LLVM's whole divide sequence runs on every pair, and the final quotient is compared with the correctly rounded IEEE quotient and with that quotient flushed.

Each row is printed next to the prediction of three candidate readings, written in the tool from the public sources and labelled as candidates:
- `isa_newer`: the text as worded, where the f32 reciprocal test is made in binary64 and so never fires;
- `isa_recip_width`: the reciprocal tested in the operand's width;
- `rdna3_emu`: tinygrad's rule as read from its source.

A run therefore shows directly which reading, if any, RDNA2 follows. It also shows:
- what special and denormal operands do in each MODE;
- the exact thresholds;
- the modifiers;
- whether the SGPR destination and wave64 match VCC and wave32.

Exit 0 means the run completed and the waves and destinations agree, whichever candidate matched. Exit 1 means they disagree.

`docs/shader_validation.md` gets a section for the tool, next to `f64_omod_probe`'s. It says plainly that the tool has not been run on hardware and what a run should settle. Nothing goes into the contract table of `docs/design.md`, because nothing is measured yet.

### Tested

- New `tests/test_div_scale_probe.py` (25 tests, standard library, registered by the `tests/test_*.py` glob like its siblings). It covers:
  - the case matrix against the docstring (48 pairs, 106 rows, 30 modifier rows, and each group's count);
  - every boundary exactly: gaps 95/96/97 and 767/768/769; fields 22/23/24 and 52/53/54; the neighbours of 2^126 and 2^1022 one unit in the last place away; quotients below, at and above 2^-126 and 2^-1022;
  - the exact rounding helper against `struct`;
  - the candidates reproducing the known disagreements: the zero-operand flag; 2^-126 / 1.0; S0 = S2 = 1.0 with S1 = 2^127; a denormal S1; and the f64 tiny-numerator quirk of the emulator rule. The 1.0 / 2^127 divide, run through the translator's gfx1036-measured `v_div_fmas_f32`/`v_div_fixup_f32` on the CPU interpreter, gives +inf under the f32-width reading and 0 under the newer wording;
  - the kernel shapes and word counts;
  - all 60 generated kernels assembling with `llvm-mc` 18 for gfx1036 in both wave sizes (skipped when `llvm-mc` is missing, as in the sibling tests);
  - hand-placed words landing in the right rows;
  - a fake oracle that executes the kernels' `v_div_scale` lines with one candidate: summary, JSON with LF line endings, `--table`, wave and destination differences exiting 1, a harness error exiting 2, an absent or non-GFX10 oracle exiting 77, CLI errors.

  Mutating the gap constant, the emulator rule's zero flag or quotient test, the wave64 mask-word count, or the reciprocal width fails it.
- `python -m compileall -q make_game.py tools tests`, `git diff --check`, `scripts/check_no_game_data.py` on the changed files: pass.
- `tests/test_text_write_line_endings.py`, `tests/test_docs_cli_flags.py`, `tests/test_docs_table_rows.py`, `tests/test_report_paths.py`, `tests/test_report_path_case.py`, `tests/test_fuzzer_work_isolation.py`, `tests/test_hw_oracle.py`, `tests/test_f64_omod_probe.py`: pass.
- The pytest files (`grep -l "import pytest" tests/*.py`, 31 files): 1,792 passed, 14 skipped, 4 failed. The 4 failures are the same as on unmodified `main`:
  - `test_translate` `cs_dpp_cmp` twice and `ps_sdwa_f16_const`: `llvm-mc` 18 is too old;
  - `test_ci_registration`, for the upstream file `tests/semantic/test_sub_co_same_operand.py`. The new test file is not in its list.

GPUs tested: none — the tool itself has not been run on hardware

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] No translator rule added, so no contract-table row; the tool's documentation says it is unmeasured.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
