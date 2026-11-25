Claimed in the coordination issue as: SHADER-DIV-SCALE-PROBE

Depends on: none

### What and why

The translator refuses `v_div_scale_f32` and `v_div_scale_f64`. The public descriptions of them disagree at the edges (see the `finding:` on v_div_scale in the coordination issue):
- the ISA pseudocode (RDNA3/RDNA4/CDNA wording; the RDNA2 document's wording was not read);
- the D = S0 pass-through LLVM's divide lowering relies on (`SIISelLowering.cpp`, `LowerFDIV32`/`LowerFDIV64`);
- the rule tinygrad's RDNA3 emulator runs after fixes drawn from its hardware tests (`test/mockgpu/amd/emu.py`, `test/amd/hw/test_vop3.py`).

Read literally, the older f32 reading of the text makes LLVM's divide return +inf for 1.0 / 2^127. So the rule has to come from a measurement. This adds the tool for it and no rule.

`tools/div_scale_probe.py` follows `tools/f64_omod_probe.py` on `tools/hw_oracle`: raw one-lane kernels, the kernel's MODE set from its descriptor (the same five MODEs, `game_i0_d3` first), wave32 and wave64, and the same CLI, JSON report and exit codes (77 without an oracle or GFX10 device).

Per width it measures:
- 49 numerator/denominator pairs, each run as LLVM's denominator call `(d, d, n)` and numerator call `(n, d, n)`:
  - special operands: zero, -0, inf, quiet and signalling NaN;
  - denormals, including the smallest denormal under a mid-range numerator (2^-60 [2^-284], exponent gap 67 [739], below 96 [768]), which reaches the denormal-denominator branch and not the gap branch;
  - exponent-field gaps 95/96/97 [767/768/769];
  - denominators just below, at and above 2^126 [2^1022], and 2^127 [2^1023];
  - quotients below, at and above 2^-126 [2^-1022];
  - numerator fields 22/23/24 [52/53/54];
  - ordinary operands.
- 10 rows with S0 = S1 = S2.

Each of those 108 rows runs with VCC and with an SGPR mask destination (s10 / s[10:11]), preset to all ones. D is preset to a sentinel too, one that no operand of the matrix and no candidate prediction equals, so the probe tells a D that is written from a D that is not (`d_unwritten`: counted in the summary, marked in `--json` and `--table`). The probe records D, the lane's flag and the whole mask word (`mask_lo`, and `mask_hi` in wave64, in `--json`). There are also 30 modifier rows (`neg` on each source; `clamp`, `mul:2`, `div:2`), with the same presets. VOP3B has no `abs`. LLVM's whole divide sequence runs on every pair, and the final quotient is compared with the correctly rounded IEEE quotient and with that quotient flushed.

Each row is printed next to the prediction of three candidate readings, written in the tool from the public sources and labelled as candidates:
- `isa_newer`: the text as worded, where the f32 reciprocal test is made in binary64 and so never fires;
- `isa_recip_width`: the same text with the reciprocal tested in the operand's width (the public emulators of the older instruction sets classify with "not normal", which also differs for zero/inf/NaN reciprocals and for zero quotients; no candidate models that, and the matrix has rows that tell it apart);
- `rdna3_emu`: tinygrad's rule as read from its source.

"D = S0 where no branch assigns it" is an assumption applied identically in all three candidates (for ordinary operands and inside every taken branch that assigns nothing); it is what LLVM relies on and it is not measured. One public emulator initialises D differently (yoshou's amdgpu-sim starts from `D = S0 * S2 / S1`; gem5's f64 model assigns D only in the branches that scale); neither is a candidate. A run shows which reading, if any, RDNA2 follows. It also shows what special and denormal operands do in each MODE, the exact thresholds, the modifiers, whether the SGPR destination and wave64 match VCC and wave32, and whether D is written at all.

What it cannot measure: the f32 divide sequence omits the `s_denorm_mode` switch LLVM wraps around its f32 FMAs (the kernels do not add one), so its verdicts are comparable with LLVM's lowering only in `i0_d3_f32keep`; the printed tallies of the other f32 MODEs say so (and `llvm_comparable` is false in the JSON). `abs` on a source cannot be encoded. A run without `--json` ends with a line saying that the summary is not the measurement; the file written by `--json` is what to post.

Exit 0 means the run completed and the waves and destinations agree, whichever candidate matched. Exit 1 means they disagree.

`docs/shader_validation.md` gets a section for the tool, next to `f64_omod_probe`'s. It says plainly that the tool has not been run on hardware and what a run should settle. Nothing goes into the contract table of `docs/design.md`, because nothing is measured yet.

### Tested

- New `tests/test_div_scale_probe.py` (36 tests, standard library, registered by the `tests/test_*.py` glob like its siblings). It covers:
  - the case matrix against the docstring (49 pairs, 108 rows, 30 modifier rows, and each group's count);
  - every boundary exactly: gaps 95/96/97 and 767/768/769; fields 22/23/24 and 52/53/54; the neighbours of 2^126 and 2^1022 one unit in the last place away; quotients below, at and above 2^-126 and 2^-1022; the pair that isolates the denormal-denominator branch;
  - the exact rounding helper against `struct`;
  - the candidates reproducing the known disagreements: the zero-operand flag; 2^-126 / 1.0; S0 = S2 = 1.0 with S1 = 2^127; a denormal S1; and the f64 tiny-numerator quirk of the emulator rule. The 1.0 / 2^127 divide, run through the translator's gfx1036-measured `v_div_fmas_f32`/`v_div_fixup_f32` on the CPU interpreter, gives +inf under the f32-width reading and 0 under the newer wording;
  - the kernel shapes and word counts, and that every `v_div_scale` of the plain and modifier kernels follows the mask preset and the D sentinel;
  - the divide sequence for both widths and both wave sizes against a table of LLVM's `LowerFDIV32`/`LowerFDIV64` instruction and operand order written by hand, and that the sequence computes the correctly rounded quotient of ordinary operands on exact arithmetic;
  - all 60 generated kernels assembling with `llvm-mc` 18 for gfx1036 in both wave sizes (skipped when `llvm-mc` is missing, as in the sibling tests);
  - hand-placed words landing in the right rows, and a flushed denormal quotient classified `cr_ftz`;
  - a fake oracle that executes the kernels' `v_div_scale` lines with one candidate: summary, JSON with LF line endings, `--table` and its legend, an unwritten D reported, the not-comparable labels of the f32 sequence, wave and destination differences exiting 1, a harness error exiting 2, an absent or non-GFX10 oracle exiting 77, CLI errors.

  Mutating the gap constant, the emulator rule's zero flag or quotient test, the wave64 mask-word count, the reciprocal width, the order or operands of the divide sequence, the mask preset, the D sentinel, the unwritten check or the flush classification fails it (checked one by one).
- `python -m compileall -q make_game.py tools tests`, `git diff --check`, `scripts/check_no_game_data.py` on the changed files: pass.
- `tests/test_text_write_line_endings.py`, `tests/test_docs_cli_flags.py`, `tests/test_docs_table_rows.py`, `tests/test_report_paths.py`, `tests/test_report_path_case.py`, `tests/test_fuzzer_work_isolation.py`, `tests/test_hw_oracle.py`, `tests/test_f64_omod_probe.py`, `tests/test_ci_registration.py`: pass.
- The pytest files (`grep -l "import pytest" tests/*.py`, 31 files): 1,794 passed, 14 skipped, 3 failed. The 3 failures are the same as on unmodified `main` and need a newer `llvm-mc` than 18: `test_translate` `cs_dpp_cmp` twice and `ps_sdwa_f16_const`.

GPUs tested: none — the tool itself has not been run on hardware

AI-assisted: yes

### Checklist

- [x] Based on current `main` and merges cleanly (a conflicting pull request is labeled `needs rebase`).
- [x] One topic; follow-up work goes in a new pull request.
- [x] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [x] No translator rule added, so no contract-table row; the tool's documentation says it is unmeasured.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Y7fbZYVY7EtLknWBZUfSLf
