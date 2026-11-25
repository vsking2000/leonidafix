"""Unit tests for tools/div_scale_probe.py (case matrix, candidate readings, kernels, simulated oracle, CLI).

Standard library only; no GPU and no game files. The probe has not been run on hardware: these tests check what can
be checked without it. The case matrix and its boundaries must be what the docstring says. The generated kernels must
assemble (with llvm-mc where it is installed). The candidate references must reproduce the contradictions they were
written to tell apart. And the words of a run must land in the right rows (a simulated oracle that executes the
kernels' v_div_scale lines with one candidate, and hand-placed words).
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import div_scale_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402

F32, F64 = pr.FMTS["f32"], pr.FMTS["f64"]
_REG = re.compile(r"^(-?)v(?:(\d+)|\[(\d+):(\d+)\])$")


def regs_of(token):
    """(negated, [register numbers]) of a VGPR operand such as v4, -v[2:3]."""
    m = _REG.match(token)
    if not m:
        raise ValueError(token)
    if m.group(2):
        return bool(m.group(1)), [int(m.group(2))]
    return bool(m.group(1)), list(range(int(m.group(3)), int(m.group(4)) + 1))


class FakeOracle:
    """Simulated hw_oracle backend: executes the probe kernels' seeds, presets, v_div_scale, copies and stores.

    v_div_scale follows the candidate `reading`, with neg source modifiers applied and clamp/omod ignored, and writes
    the flag to bit 0 of the mask with every other bit cleared. v_div_fixup writes the correctly rounded quotient of
    its denominator and numerator registers. The remaining arithmetic of the divide is not executed."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, reading="isa_recip_width", arch=None, wave64_flips=False, sgpr_flips=False, fail=None):
        self.reading, self.wave64_flips, self.sgpr_flips, self.fail = reading, wave64_flips, sgpr_flips, fail
        self.arch = arch or self.arch
        self.closed = False

    def run(self, kernels):
        if self.fail:
            return [self.fail for _ in kernels]
        return [self.execute(k) for k in kernels]

    def execute(self, kernel):
        v, s, words = {}, {}, [0xDEAD0000] * kernel.words
        pairs = {"vcc": ("vcc_lo", "vcc_hi"), "s[10:11]": ("s10", "s11"), "s[12:13]": ("s12", "s13")}
        for line in kernel.lines:
            op, _, rest = line.partition(" ")
            args = [a.strip() for a in rest.split(",")]
            if op == "v_mov_b32_e32":
                src = args[1]
                v[int(args[0][1:])] = s.get(src, 0) if not src[0].isdigit() and not src.startswith("-") \
                    else int(src, 0) & 0xFFFFFFFF
            elif op in ("s_mov_b32", "s_mov_b64"):
                for name in pairs.get(args[0], (args[0],)):
                    s[name] = 0xFFFFFFFF
            elif op.startswith("v_div_scale_"):
                f = pr.FMTS[op[-3:]]
                last = args[4].split()
                args[4] = last[0]
                ops = []
                for tok in args[2:5]:
                    negated, rr = regs_of(tok)
                    bits = sum(v.get(r, 0) << (32 * i) for i, r in enumerate(rr))
                    ops.append(pr.neg(f, bits) if negated else bits)
                d, flag = pr.READINGS[self.reading](f, *ops)
                if d == pr.NAN:
                    d = (f.emask << f.mant) | (1 << (f.mant - 1))
                if self.wave64_flips and kernel.wave == 64:
                    flag ^= 1
                if self.sgpr_flips and args[1].startswith("s1"):
                    flag ^= 1
                for i, r in enumerate(regs_of(args[0])[1]):
                    v[r] = (d >> (32 * i)) & 0xFFFFFFFF
                names = pairs.get(args[1], (args[1],))
                for i, name in enumerate(names):
                    s[name] = flag if i == 0 else 0
            elif op.startswith("v_div_fixup_"):
                f = pr.FMTS[op[-3:]]
                den = sum(v.get(r, 0) << (32 * i) for i, r in enumerate(regs_of(args[2])[1]))
                num = sum(v.get(r, 0) << (32 * i) for i, r in enumerate(regs_of(args[3])[1]))
                q = pr.divide(f, num, den)
                if q == pr.NAN:
                    q = (f.emask << f.mant) | (1 << (f.mant - 1))
                for i, r in enumerate(regs_of(args[0])[1]):
                    v[r] = (q >> (32 * i)) & 0xFFFFFFFF
            elif op == "global_store_dword":
                words[v[20] // 4] = v[int(args[1][1:])]
        return words

    def close(self):
        self.closed = True


def run_main(argv, oracle):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle):
        with redirect_stdout(out), redirect_stderr(err):
            code = pr.main(argv)
    return code, out.getvalue(), err.getvalue()


class CaseMatrixTests(unittest.TestCase):
    def test_counts_are_the_docstring_counts(self):
        doc = pr.__doc__
        for f in (F32, F64):
            self.assertEqual(len(pr.pairs(f)), 48)
            self.assertEqual(len(pr.same_values(f)), 10)
            self.assertEqual(len(pr.scale_rows(f)), 106)
            self.assertEqual(len(pr.mod_rows(f)), 30)
            self.assertEqual(len({name for name, _, _ in pr.pairs(f)}), 48, "case names must be unique")
        for text in ("That is 48 pairs", "10 cases with\nS0 = S1 = S2", "106 rows per width", "30 rows per width",
                     "11 special pairs", "5 denormal pairs", "6 exponent-gap pairs", "8 reciprocal pairs",
                     "5 quotient pairs", "6 tiny-numerator pairs", "7 ordinary pairs"):
            self.assertIn(text, doc)
        groups = [11, 5, 6, 8, 5, 6, 7]
        self.assertEqual(sum(groups), 48)

    def test_exponent_gap_boundaries(self):
        for f, want in ((F32, [95, 96, 97, 95, 96, 97]), (F64, [767, 768, 769, 767, 768, 769])):
            gaps = [pr.exp_field(f, n) - pr.exp_field(f, d) for name, n, d in pr.pairs(f) if name.startswith("gap ")]
            self.assertEqual(gaps, want)

    def test_tiny_numerator_boundaries(self):
        for f, want in ((F32, [22, 22, 23, 23, 24, 24]), (F64, [52, 52, 53, 53, 54, 54])):
            fields = [pr.exp_field(f, n) for name, n, _ in pr.pairs(f) if name.startswith("field ")]
            self.assertEqual(fields, want)
            for name, n, _ in pr.pairs(f):
                if name.startswith("field "):
                    self.assertEqual(pr.value(f, n), Fraction(2) ** (pr.exp_field(f, n) - f.bias))

    def test_reciprocal_boundaries(self):
        for f, k in ((F32, 126), (F64, 1022)):
            dens = [d for name, n, d in pr.pairs(f) if name.startswith("1 / ") and ("2^" in name)
                    and not name.startswith("1 / -")]
            edge = Fraction(2) ** k
            self.assertEqual([pr.value(f, d) for d in dens][1], edge)
            below, at, above, top = (pr.value(f, d) for d in dens)
            self.assertLess(below, edge)
            self.assertGreater(above, edge)
            self.assertEqual(top, edge * 2)
            # neighbours, one unit in the last place away
            self.assertEqual(edge - below, edge * Fraction(1, 2 ** (f.mant + 1)))
            self.assertEqual(above - edge, edge * Fraction(1, 2 ** f.mant))
            # 1 / 2^k is the smallest normal; just above 2^k the reciprocal is denormal
            self.assertFalse(pr.is_denorm(f, pr.divide(f, pr.pow2(f, 0), dens[1])))
            self.assertTrue(pr.is_denorm(f, pr.divide(f, pr.pow2(f, 0), dens[2])))

    def test_quotient_boundaries(self):
        for f, k in ((F32, -126), (F64, -1022)):
            named = {name: (n, d) for name, n, d in pr.pairs(f)}
            mn = Fraction(2) ** k
            q = {name: pr.value(f, n) / pr.value(f, d) for name, (n, d) in named.items()
                 if pr.value(f, d) != 0 and not pr.is_nan(f, n) and not pr.is_inf(f, n)
                 and not pr.is_nan(f, d) and not pr.is_inf(f, d)}
            self.assertLess(q[f"below 2^{k + 1} / 2"], mn)
            self.assertEqual(q[f"2^{k + 1} / 2"], mn)
            self.assertGreater(q[f"above 2^{k + 1} / 2"], mn)
            self.assertEqual(q[f"2^{k} / 1"], mn)
            self.assertEqual(q[f"2^{k} / 2^30"], mn / 2 ** 30)

    def test_round_to_and_value_agree_with_struct(self):
        for x in (1.0, -2.5, 3e-5, 1e10, 2.0 ** -149, 2.0 ** -130, 3.4e38):
            self.assertEqual(pr.round_to(F32, Fraction(x)), pr.from_float(F32, x))
        for x in (1.0, -2.5, 3e-5, 1e300, 2.0 ** -1074, 2.0 ** -1050):
            self.assertEqual(pr.round_to(F64, Fraction(x)), pr.from_float(F64, x))
            self.assertEqual(pr.value(F64, pr.from_float(F64, x)), Fraction(x))
        self.assertEqual(pr.round_to(F32, Fraction(2) ** 200), 0x7F800000)


class CandidateTests(unittest.TestCase):
    """The contradictions docs/design.md and the coordination finding name, as the three references give them."""

    def test_zero_operand_flag(self):
        one = pr.pow2(F32, 0)
        for f in (F32, F64):
            o = pr.pow2(f, 0)
            self.assertEqual(pr.isa_newer(f, o, 0, o), (pr.NAN, 0))
            self.assertEqual(pr.isa_recip_width(f, o, 0, o), (pr.NAN, 0))
            self.assertEqual(pr.rdna3_emu(f, o, 0, o), (pr.NAN, 1))
        self.assertEqual(pr.isa_newer(F32, 0, one, 0), (pr.NAN, 0))

    def test_smallest_normal_over_one(self):
        f, n, one = F32, pr.pow2(F32, -126), pr.pow2(F32, 0)
        self.assertEqual(pr.isa_newer(f, n, one, n), (pr.pow2(f, -62), 0))
        self.assertEqual(pr.isa_recip_width(f, n, one, n), (pr.pow2(f, -62), 0))
        self.assertEqual(pr.rdna3_emu(f, n, one, n), (pr.pow2(f, -62), 1))
        # The denominator call: the text scales 1.0 too (both scaled, flag clear).
        self.assertEqual(pr.isa_newer(f, one, one, n), (pr.pow2(f, 64), 0))

    def test_one_over_2_127(self):
        f, one, d = F32, pr.pow2(F32, 0), pr.pow2(F32, 127)
        self.assertEqual(pr.isa_newer(f, one, d, one), (pr.pow2(f, 64), 1))
        self.assertEqual(pr.isa_recip_width(f, one, d, one), (one, 1))
        self.assertEqual(pr.rdna3_emu(f, one, d, one), (one, 1))
        # Denominator call under the f32-width reading: 2^127 * 2^64 overflows.
        self.assertEqual(pr.isa_recip_width(f, d, d, one), (0x7F800000, 1))
        self.assertEqual(pr.isa_newer(f, d, d, one), (d, 1))

    def test_one_over_2_127_divide_is_inf_under_the_f32_width_reading(self):
        """The LLVM f32 divide fed each reading's v_div_scale results, through the translator's v_div_fmas_f32 and
        v_div_fixup_f32 (gfx1036-measured rules) on the CPU interpreter: +inf for the f32-width reading, 0 for the
        newer wording (2^-127 flushed under the game MODE)."""
        sys.path.insert(0, str(ROOT / "tests" / "semantic"))
        try:
            import support
        finally:
            sys.path.remove(str(ROOT / "tests" / "semantic"))
        f, one, d = F32, pr.pow2(F32, 0), pr.pow2(F32, 127)
        results = {}
        for c in ("isa_newer", "isa_recip_width"):
            den_s, _ = pr.READINGS[c](f, d, d, one)
            num_s, flag = pr.READINGS[c](f, one, d, one)
            lines = [f"v_mov_b32 v2, {d:#x}", f"v_mov_b32 v4, {one:#x}", f"v_mov_b32 v6, {den_s:#x}",
                     f"v_mov_b32 v7, {num_s:#x}", f"s_mov_b32 vcc_lo, {flag}"]
            lines += [ln for ln in pr.sequence_lines(f, 32) if not ln.startswith("v_div_scale")]
            lines += ["v_mov_b32 v0, v8", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
            results[c] = support.mrt0_words(support.translate_lines("ps_div_scale_seq", lines))[0]
        self.assertEqual(results["isa_recip_width"], 0x7F800000)
        self.assertEqual(results["isa_newer"], 0x00000000)

    def test_denormal_denominator(self):
        f, one, dmin = F32, pr.pow2(F32, 0), 1
        self.assertEqual(pr.rdna3_emu(f, dmin, dmin, one), (pr.NAN, 1))
        self.assertEqual(pr.isa_newer(f, dmin, dmin, one), (pr.ldexp(f, dmin, 64), 1))

    def test_rdna3_emu_f64_tiny_numerator_keeps_the_text(self):
        f, one = F64, pr.pow2(F64, 0)
        n = pr.pow2(f, 53 - 1023)
        self.assertEqual(pr.rdna3_emu(f, n, one, n), (pr.ldexp(f, n, 128), 0))
        n32 = pr.pow2(F32, 23 - 127)
        self.assertEqual(pr.rdna3_emu(F32, n32, pr.pow2(F32, 0), n32)[1], 1)

    def test_ordinary_operands_pass_through(self):
        for f in (F32, F64):
            for c in pr.CANDIDATES:
                for n, d in ((3, 7), (1, 4), (-5.5, 1.25)):
                    nb, db = pr.round_to(f, Fraction(n)), pr.round_to(f, Fraction(d))
                    self.assertEqual(pr.READINGS[c](f, db, db, nb), (db, 0), (f.name, c, n, d))
                    self.assertEqual(pr.READINGS[c](f, nb, db, nb), (nb, 0), (f.name, c, n, d))

    def test_self_test(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(pr.main(["--self-test"]), 0)
        self.assertIn("self-test ok", out.getvalue())


class KernelTests(unittest.TestCase):
    def test_kernel_shapes(self):
        kernels, meta = pr.build_kernels(asm, list(pr.MODES), [32, 64], list(pr.WIDTHS))
        self.assertEqual(len(kernels), 2 * 5 * 2 * 3)
        self.assertEqual(meta[0], (32, "game_i0_d3", "f32", "scale"))
        for k, (wave, mode, width, section) in zip(kernels, meta):
            self.assertTrue(k.raw)
            self.assertEqual(k.threads, 1)
            self.assertEqual(k.mode, pr.MODES[mode])
            self.assertEqual(k.lines[-2:], ["s_waitcnt vmcnt(0)", "s_endpgm"])
            stores = sum(1 for ln in k.lines if ln.startswith("global_store_dword"))
            self.assertEqual(stores, k.words)
            f = pr.FMTS[width]
            per_row = f.words + (1 if wave == 32 else 2)
            want = {"scale": 2 * 106 * per_row, "mods": 30 * per_row, "seq": 48 * f.words}[section]
            self.assertEqual(k.words, want)
            self.assertLessEqual(k.words, 16384)

    def test_mask_destinations(self):
        lines, _ = pr.scale_kernel_lines(F32, 64)
        text = "\n".join(lines)
        self.assertIn("v_div_scale_f32 v8, vcc, v2, v2, v4", text)
        self.assertIn("v_div_scale_f32 v8, s[10:11], v4, v2, v4", text)
        self.assertIn("v_mov_b32_e32 v21, vcc_hi", text)
        lines, _ = pr.scale_kernel_lines(F64, 32)
        text = "\n".join(lines)
        self.assertIn("v_div_scale_f64 v[8:9], vcc_lo, v[2:3], v[2:3], v[4:5]", text)
        self.assertIn("v_div_scale_f64 v[8:9], s10, v[4:5], v[2:3], v[4:5]", text)
        self.assertNotIn("|", text)          # VOP3B has no abs

    def test_kernels_assemble_when_llvm_mc_available(self):
        llvm_mc = os.environ.get("LLVM_MC") or shutil.which("llvm-mc")
        if not llvm_mc or not os.path.isfile(llvm_mc):
            self.skipTest("llvm-mc not found")
        for wave in (32, 64):
            kernels, _ = pr.build_kernels(asm, list(pr.MODES), [wave], list(pr.WIDTHS))
            text, _, _ = asm.module_text(kernels, "gfx1036", wave=wave)
            flags = ["-mattr=+wavefrontsize64"] if wave == 64 else []
            res = subprocess.run(
                [llvm_mc, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1036", *flags, "-filetype=obj", "-o", os.devnull],
                input=text, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr[:800])


class ParsingTests(unittest.TestCase):
    def test_hand_placed_words_land_in_their_rows(self):
        meta = [(32, "game_i0_d3", "f32", "scale"), (64, "game_i0_d3", "f64", "scale"),
                (32, "game_i0_d3", "f32", "seq")]
        f32_words = [0] * (2 * 106 * 2)
        f32_words[0:2] = [0x12345678, 0xFFFFFFFE]        # row 0: VCC, "1 / +0", den call; flag 0, other lanes set
        f32_words[106 * 2:106 * 2 + 2] = [0x0BADF00D, 1]  # first SGPR row
        f64_words = [0] * (2 * 106 * 4)
        f64_words[4:8] = [0x11111111, 0x22222222, 1, 0x80000000]   # row 1: "1 / +0", num call, wave64
        seq_words = [0] * 48
        seq_words[47] = 0xBF800000
        rows, seq_rows, summary = pr.evaluate(meta, [f32_words, f64_words, {"words": seq_words}])
        r0 = rows[0]
        self.assertEqual((r0["case"], r0["call"], r0["dst"], r0["d"], r0["mask"]),
                         ("1 / +0", "den", "vcc", 0x12345678, [0xFFFFFFFE]))
        r_sgpr = rows[106]
        self.assertEqual((r_sgpr["dst"], r_sgpr["case"], r_sgpr["d"], r_sgpr["mask"]),
                         ("sgpr", "1 / +0", 0x0BADF00D, [1]))
        r64 = [r for r in rows if r["width"] == "f64"][1]
        self.assertEqual((r64["call"], r64["d"], r64["mask"]), ("num", 0x2222222211111111, [1, 0x80000000]))
        self.assertEqual((seq_rows[-1]["case"], seq_rows[-1]["q"]), (pr.pairs(F32)[-1][0], 0xBF800000))
        # 1 / -2^127 correctly rounded is -2^-127, a denormal: -1.0 is no reading's answer.
        self.assertEqual(seq_rows[-1]["verdict"], "other")
        self.assertIn("game_i0_d3 f32", summary["candidates"])

    def test_harness_string_result_is_an_error(self):
        with self.assertRaises(RuntimeError):
            pr.evaluate([(32, "game_i0_d3", "f32", "seq")], ["assembler rejected: x"])


class ProbeCliTests(unittest.TestCase):
    def test_device_following_one_reading(self):
        with tempfile.TemporaryDirectory() as work:
            report = os.path.join(work, "nested", "probe.json")
            oracle = FakeOracle(reading="isa_recip_width")
            code, out, err = run_main(["--json", report], oracle)
            self.assertEqual((code, err), (0, ""), out)
            self.assertTrue(oracle.closed)
            self.assertIn("wave_diffs=0, dst_diffs=0", out)
            for mode in pr.MODES:
                for width in pr.WIDTHS:
                    line = next(ln for ln in out.splitlines() if ln.startswith(f"{mode} {width}: "))
                    m = re.search(r"isa_recip_width (\d+)/(\d+)", line)
                    self.assertEqual(m.group(1), m.group(2), line)
            # the f32 readings part ways; for f64 the two text readings are the same rule
            f32_line = next(ln for ln in out.splitlines() if ln.startswith("game_i0_d3 f32: "))
            m = re.search(r"isa_newer (\d+)/(\d+)", f32_line)
            self.assertLess(int(m.group(1)), int(m.group(2)))
            f64_line = next(ln for ln in out.splitlines() if ln.startswith("game_i0_d3 f64: "))
            m = re.search(r"isa_newer (\d+)/(\d+)", f64_line)
            self.assertEqual(m.group(1), m.group(2))
            self.assertIn("isa_newer MISMATCH game_i0_d3 f32", out)
            with open(report, "rb") as fh:
                raw = fh.read()
            self.assertNotIn(b"\r\n", raw)
            data = json.loads(raw.decode("utf-8"))
            self.assertEqual(data["arch"], "gfx1036")
            self.assertEqual(data["modes"], list(pr.MODES))
            self.assertEqual(data["candidates"], list(pr.CANDIDATES))
            self.assertEqual(len(data["rows"]), 2 * 5 * 2 * (2 * 106 + 30))
            self.assertEqual(len(data["sequence"]), 2 * 5 * 2 * 48)
            row = data["rows"][0]
            self.assertEqual(set(row["candidates"]), set(pr.CANDIDATES))
            self.assertTrue(row["candidates"]["isa_recip_width"]["match"])
            seq = data["summary"]["sequence"]["game_i0_d3 f32"]
            self.assertEqual(seq["other"], 0)

    def test_table_prints_every_row(self):
        code, out, _ = run_main(["--table", "--waves", "32", "--modes", "game_i0_d3", "--widths", "f32"],
                                FakeOracle())
        self.assertEqual(code, 0)
        self.assertEqual(sum(1 for ln in out.splitlines() if ln.startswith("w32 game_i0_d3 f32 ")), 2 * 106 + 30)

    def test_wave_and_destination_differences_exit_1(self):
        code, out, _ = run_main(["--modes", "game_i0_d3"], FakeOracle(wave64_flips=True))
        self.assertEqual(code, 1)
        self.assertIn("WAVE DIFF", out)
        code, out, _ = run_main(["--modes", "game_i0_d3", "--waves", "32"], FakeOracle(sgpr_flips=True))
        self.assertEqual(code, 1)
        self.assertIn("DST DIFF", out)

    def test_harness_error_exits_2(self):
        code, out, _ = run_main(["--modes", "game_i0_d3"], FakeOracle(fail="assembler rejected: bad"))
        self.assertEqual(code, 2)
        self.assertIn("harness error", out)

    def test_unavailable_or_non_gfx10_oracle_exits_77(self):
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no device")):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(pr.main([]), 77)
            self.assertIn("hardware oracle unavailable", out.getvalue())
        code, out, _ = run_main([], FakeOracle(arch="gfx1100"))
        self.assertEqual((code, out.strip()), (77, "hardware oracle unavailable: gfx1100 is not GFX10"))

    def test_unknown_entries_are_cli_errors(self):
        for argv in (["--modes", "game_i0_d3,nope"], ["--widths", "f16"], ["--waves", "16"]):
            with self.assertRaises(SystemExit) as caught:
                with redirect_stderr(io.StringIO()):
                    pr.main(argv)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
