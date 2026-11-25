"""Semantic tests for RDNA2 v_trig_preop_f64 (SHADER-TRIG-PREOP), written from the ISA pseudocode.

    shift = S1[4:0] * 53; if exponent(S0) > 1077 then shift += exponent(S0) - 1077
    result = the 53 fraction bits of 2/pi after the first `shift` bits, as an integer
    scale = -53 - shift; if exponent(S0) >= 1968 then scale += 128
    D = ldexp(result, scale)

The translator takes S1 only as an integer constant with S1[4:0] <= 2 (docs/design.md says why). Every expected
word here comes from `ref_trig_preop` below: 2/pi from its own integer series (Gauss's arctangent formula, not the
Machin formula the translator uses) and the pseudocode above, never from the translator's helpers. None of it was
measured on RDNA2 hardware.
"""
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402  (puts the tools directory on sys.path)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

FRACTION_BITS = 1201          # the ISA text's 1201'B(2.0 / PI)
SENT_LO, SENT_HI = 0x13572468, 0x24681357


def _gauss_pi_bounds(bits):
    """lo <= pi * 2**bits <= hi from pi/4 = 12 atan(1/18) + 8 atan(1/57) - 5 atan(1/239), in integers."""
    g = bits + 64
    one = 1 << g

    def atan_inv(x):
        total, power, k, terms = 0, one // x, 0, 0
        while power:
            total += (-1) ** k * (power // (2 * k + 1))
            power //= x * x
            k += 1
            terms += 1
        return total, terms + 2         # each floored term is off by < 1 unit, and so is the dropped tail

    (a18, e18), (a57, e57), (a239, e239) = atan_inv(18), atan_inv(57), atan_inv(239)
    approx, err = 4 * (12 * a18 + 8 * a57 - 5 * a239), 4 * (12 * e18 + 8 * e57 + 5 * e239)
    return (approx - err) >> 64, -((-(approx + err)) >> 64)


def _two_over_pi_fraction(bits):
    """floor(2/pi * 2**bits), proved by the two bounds giving the same floor."""
    g = bits + 64
    lo, hi = _gauss_pi_bounds(g)
    num = 1 << (bits + 1 + g)
    q = num // hi
    assert q == num // lo, "2/pi not determined"
    return q


TWO_OVER_PI = _two_over_pi_fraction(FRACTION_BITS)     # 0.{b_1200 ... b_0} of the ISA text, as an integer
# The leading hex digits of 2/pi as published (e.g. in the argument-reduction tables of fdlibm's __kernel_rem_pio2).
PUBLISHED_PREFIX = 0xA2F9836E4E441529FC2757D1F534DDC0DB6295993C439041FE5163AB


def ref_trig_preop(c, s0_bits):
    """The pseudocode on the binary64 bits of S0 and S1[4:0] = c; returns the 64-bit result."""
    e = (s0_bits >> 52) & 0x7FF
    shift = c * 53 + (e - 1077 if e > 1077 else 0)
    scale = -53 - shift + (128 if e >= 1968 else 0)
    window = (TWO_OVER_PI << shift >> (FRACTION_BITS - 53)) & ((1 << 53) - 1)
    value = Fraction(window) * Fraction(2) ** scale
    out = float(value)
    assert Fraction(out) == value, "the reference result is not exact"   # never happens for c <= 2 (test below)
    return struct.unpack("<Q", struct.pack("<d", out))[0]


def d2u(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def run(lines):
    spv = support.translate_lines("ps_trig_preop_f64", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    w = support.mrt0_words(spv)
    return w[0] | (w[1] << 32)


def program(s0_lo, s0_hi, s1, s0="v[4:5]", extra=()):
    return [f"v_mov_b32 v4, {s0_lo:#x}", f"v_mov_b32 v5, {s0_hi:#x}", *extra,
            f"v_trig_preop_f64 v[0:1], {s0}, {s1}"]


_TEMPLATES = {}


def eval_preop(c, s0_bits, s0="v[4:5]"):
    """Translate once per (segment, S0 spelling) and substitute S0's two words into the module's constants."""
    key = (c, s0)
    if key not in _TEMPLATES:
        _TEMPLATES[key] = support.translate_lines(
            "ps_trig_preop_f64", program(SENT_LO, SENT_HI, c, s0) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    spv = (_TEMPLATES[key].replace(f"OpConstant %uint {SENT_LO}\n", f"OpConstant %uint {s0_bits & 0xFFFFFFFF}\n")
           .replace(f"OpConstant %uint {SENT_HI}\n", f"OpConstant %uint {s0_bits >> 32}\n"))
    out = si.run_fragment(spv)["%out_mrt0"]
    return out[0] | (out[1] << 32)


class TwoOverPiTable(unittest.TestCase):
    def test_reference_matches_published_digits(self):
        n = PUBLISHED_PREFIX.bit_length()
        self.assertEqual(n, 224)
        self.assertEqual(TWO_OVER_PI >> (FRACTION_BITS - n), PUBLISHED_PREFIX)

    def test_translator_table_matches_independent_derivation(self):
        count = rdna2spirv.Gen.TRIG_PREOP_WORDS
        words = rdna2spirv.two_over_pi_words(count)
        self.assertEqual(len(words), count)
        table = 0
        for w in words:
            table = (table << 32) | w
        self.assertEqual(table, TWO_OVER_PI >> (FRACTION_BITS - 32 * count))

    def test_translated_domain_has_only_normal_results(self):
        # S1[4:0] <= 2 and any exponent field: the largest shift is 2 * 53 + 970 = 1076, which reads fraction bits
        # up to 1129 (inside the table's 1152), and no result is zero or below 2^-1022, so none is rounded.
        for c in range(3):
            for e in range(2048):
                r = ref_trig_preop(c, e << 52)
                self.assertTrue(0 < (r >> 52) < 0x7FF, (c, e, hex(r)))


class TrigPreopValues(unittest.TestCase):
    def test_every_exponent_of_every_translated_segment(self):
        # Covers both sides of 1077 and 1968, exponent 0 (zero and denormal S0), 2046 (the largest finite) and
        # 2047 (inf and NaN): the text reads the exponent field whatever the class.
        for c in range(3):
            for e in range(2048):
                bits = (e << 52) | 0x000ABCDE12345678
                self.assertEqual(eval_preop(c, bits), ref_trig_preop(c, bits), (c, e))

    def test_segment_zero_of_one_is_two_over_pi(self):
        # The top 53 bits of 2/pi, truncated: 0x145F306DC9C882 * 2^-53, an exponent field of 0x3FE.
        self.assertEqual(ref_trig_preop(0, d2u(1.0)), 0x3FE45F306DC9C882)
        self.assertEqual(run(program(0, 0x3FF00000, 0)), 0x3FE45F306DC9C882)

    def test_sign_and_mantissa_are_not_read(self):
        rng = random.Random(1201)
        for _ in range(60):
            e = rng.choice([0, 1, 1023, 1077, 1078, 1500, 1967, 1968, 2046, 2047, rng.randrange(2048)])
            c = rng.randrange(3)
            bits = (rng.getrandbits(1) << 63) | (e << 52) | rng.getrandbits(52)
            want = ref_trig_preop(c, bits & ~(1 << 63))
            self.assertEqual(eval_preop(c, bits), want, (c, hex(bits)))
            for spelling in ("-v[4:5]", "|v[4:5]|", "-|v[4:5]|"):
                self.assertEqual(eval_preop(c, bits, spelling), want, (c, hex(bits), spelling))

    def test_named_operands(self):
        cases = [
            (0x7FEFFFFF, 0xFFFFFFFF),   # the largest finite
            (0x7FF00000, 0),            # +inf
            (0xFFF00000, 0),            # -inf
            (0x7FF80000, 0x123),        # quiet NaN
            (0x7FF00000, 1),            # signalling NaN
            (0x000FFFFF, 0xFFFFFFFF),   # the largest denormal
            (0, 0), (0x80000000, 0),    # +-0
            (0x434FFFFF, 0xFFFFFFFF),   # exponent 1076, just below the 1077 step
            (0x43500000, 0),            # exponent 1077
            (0x43600000, 0),            # exponent 1078: shift + 1
            (0x7AFFFFFF, 0xFFFFFFFF),   # exponent 1967
            (0x7B000000, 0),            # exponent 1968: scale + 128
        ]
        for hi, lo in cases:
            for c in range(3):
                self.assertEqual(run(program(lo, hi, c)), ref_trig_preop(c, (hi << 32) | lo), (hex(hi), c))

    def test_s1_bits_above_four_are_not_read(self):
        # Inline integers 32..34 and 64, a literal with bit 5 set, and literals that are negative as an i32.
        for s1, c in (("32", 0), ("33", 1), ("34", 2), ("64", 0), ("65", 1), ("0x41", 1), ("0xffffffe2", 2),
                      ("0x80000000", 0)):
            self.assertEqual(run(program(0, 0x40900000, s1)), ref_trig_preop(c, d2u(1024.0)), s1)

    def test_s0_source_forms(self):
        sgpr = ["s_mov_b32 s4, 0", "s_mov_b32 s5, 0x47000000", "v_trig_preop_f64 v[0:1], s[4:5], 1"]
        self.assertEqual(run(sgpr), ref_trig_preop(1, 0x4700000000000000))
        # Inline constants are binary64 values; a 32-bit literal supplies bits 63:32 (docs/design.md, f64 literals).
        for s0, bits in (("1.0", d2u(1.0)), ("-4.0", d2u(-4.0)), ("0.5", d2u(0.5)), ("64", 64),
                         ("0x7ff00000", 0x7FF0000000000000), ("0x47e00000", 0x47E0000000000000)):
            for c in range(3):
                self.assertEqual(run([f"v_trig_preop_f64 v[0:1], {s0}, {c}"]), ref_trig_preop(c, bits), (s0, c))

    def test_round_mode_does_not_change_the_exact_result(self):
        for mode in (0, 3):     # FP_ROUND[3:2] (f16/f64): nearest even, toward zero
            for c in range(3):
                for hi in (0x3FF00000, 0x7FEFFFFF, 0x7B000000):
                    lines = program(0, hi, c, extra=[f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}"])
                    self.assertEqual(run(lines), ref_trig_preop(c, hi << 32), (mode, c, hex(hi)))

    def test_inactive_lanes_keep_their_destination(self):
        # Wave32 compute: lanes 0..9 run the op with exponent 1100 + 40 * lane, lanes 10..31 are masked off.
        fssm = bytearray(0x148)
        fssm[:4] = b"FSSM"
        struct.pack_into("<I", fssm, 0xD4, 0x20)
        struct.pack_into("<4I", fssm, 0x138, 0x8088, 32, 1, 1)
        lines = ["v_mul_u32_u24 v4, 40, v0", "v_add_nc_u32 v4, 0x44c, v4", "v_lshlrev_b32 v5, 20, v4",
                 "v_mov_b32 v4, 0", f"v_mov_b32 v6, {SENT_LO:#x}", f"v_mov_b32 v7, {SENT_HI:#x}",
                 "v_cmpx_gt_u32 10, v0", "v_trig_preop_f64 v[6:7], v[4:5], 2", "s_mov_b32 exec_lo, -1",
                 "v_lshlrev_b32 v9, 3, v0",
                 "buffer_store_dword v6, v9, s[0:3], 0 offen", "buffer_store_dword v7, v9, s[0:3], 0 offen offset:4",
                 "s_endpgm"]
        spv = support.translate_lines("cs_trig_preop_f64", lines, fssm=bytes(fssm))
        mem = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                    "%sg_lane": i},
                               push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
        for lane in range(32):
            lo, hi = mem[8 * lane], mem[8 * lane + 4]
            if lane < 10:
                want = ref_trig_preop(2, (1100 + 40 * lane) << 52)
                self.assertEqual((lo, hi), (want & 0xFFFFFFFF, want >> 32), f"lane {lane}")
            else:
                self.assertEqual((lo, hi), (SENT_LO, SENT_HI), f"inactive lane {lane}")


class TrigPreopLlvmFoldWords(unittest.TestCase):
    """The expected words of LLVM's constant-folding tests for llvm.amdgcn.trig.preop.f64, written out as literals.

    Origin: llvm/test/Transforms/InstCombine/AMDGPU/amdgcn-intrinsics.ll, the `trig_preop_constfold*` functions (the
    22 whose segment S1[4:0] is 0..2; the others return 0 for an out-of-range segment and are refused here). They are
    compile-time folds written by LLVM's AMDGPU developers with bit-exact CHECK words, not hardware measurements; the
    reference above does not produce these words. Rows are (LLVM function name without its prefix, S0 bits, the S1
    the LLVM test passes, expected result). The LLVM names `..._mantissa1__segment1` and `..._mantissaX__segment1` pass
    S1 = 0, not 1; the S1 column is what the test passes.
    """
    CASES = (
        ("inf_0", 0x7FF0000000000000, 0, 0x0B43DD63F5F2F8BD),
        ("ninf_0", 0xFFF0000000000000, 0, 0x0B43DD63F5F2F8BD),
        ("constfold", 0x407596F5C28F5C29, 2, 0x394A6EE06DB14ACC),
        ("constfold_neg32_segment", 0x407596F5C28F5C29, -32, 0x3FE45F306DC9C882),
        ("constfold_exponent0_mantissa0__segment0", 0x0000000000000000, 0, 0x3FE45F306DC9C882),
        ("constfold_exponent0_mantissa1__segment0", 0x000FFFFFFFFFFFFF, 0, 0x3FE45F306DC9C882),
        ("constfold_exponent0_mantissaX__segment0", 0x0004A7F09D5F47D4, 0, 0x3FE45F306DC9C882),
        ("constfold_exponent0_mantissa0__segment2", 0x0000000000000000, 2, 0x394A6EE06DB14ACC),
        ("constfold_exponent0_mantissa1__segment2", 0x000FFFFFFFFFFFFF, 2, 0x394A6EE06DB14ACC),
        ("constfold_exponent0_mantissaX__segment2", 0x00094A6EE06DB14A, 2, 0x394A6EE06DB14ACC),
        ("constfold_exponent1607_mantissa0__segment0", 0x6470000000000000, 0, 0x1EC8135A2FBF209C),
        ("constfold_exponent1607_mantissa1__segment1", 0x647FFFFFFFFFFFFF, 0, 0x1EC8135A2FBF209C),
        ("constfold_exponent1607_mantissaX__segment1", 0x6471B791D6398353, 0, 0x1EC8135A2FBF209C),
        ("constfold_exponent1607_mantissa0__segment2", 0x6470000000000000, 2, 0x181272117E2EF7E4),
        ("constfold_exponent1607_mantissa1__segment2", 0x647FFFFFFFFFFFFF, 2, 0x181272117E2EF7E4),
        ("constfold_exponent1607_mantissaX__segment2", 0x647272117E2EF7E4, 2, 0x181272117E2EF7E4),
        ("constfold_exponent1968_mantissa0__segment0", 0x7B00000000000000, 0, 0x10374F463F669E5F),
        ("constfold_exponent1968_mantissa1__segment0", 0x7B0FFFFFFFFFFFFF, 0, 0x10374F463F669E5F),
        ("constfold_exponent1968_mantissax__segment0", 0x7B074F463F669E5F, 0, 0x10374F463F669E5F),
        ("constfold_exponent1968_mantissa0__segment2", 0x7B00000000000000, 2, 0x098F2F8BD9E839CE),
        ("constfold_exponent1968_mantissa1__segment2", 0x7B0FFFFFFFFFFFFF, 2, 0x098F2F8BD9E839CE),
        ("constfold_exponent1968_mantissaX__segment2", 0x7B0A2F8BD9E839CE, 2, 0x098F2F8BD9E839CE),
    )

    def test_there_are_twenty_two_cases(self):
        self.assertEqual(len(self.CASES), 22)

    def test_translator_reproduces_every_in_domain_llvm_word(self):
        for name, s0, s1, want in self.CASES:
            with self.subTest(name=name):
                self.assertEqual(run(program(s0 & 0xFFFFFFFF, s0 >> 32, s1)), want)


class TrigPreopRegisterShapes(unittest.TestCase):
    S0 = 0x7FD0000000000000     # exponent field 2045: shift 968 + 53 * segment

    def export(self, lines, outs):
        spv = support.translate_lines("ps_trig_preop_f64", lines + [f"exp mrt0 {outs} done vm", "s_endpgm"])
        w = support.mrt0_words(spv)
        return [w[0] | (w[1] << 32), w[2] | (w[3] << 32)]

    def test_destination_is_the_same_pair_as_s0(self):
        # S0's high word is read before either destination word is written.
        lines = program(self.S0 & 0xFFFFFFFF, self.S0 >> 32, 1)[:2] + ["v_trig_preop_f64 v[4:5], v[4:5], 1"]
        self.assertEqual(self.export(lines, "v4, v5, v4, v5")[0], ref_trig_preop(1, self.S0))

    def test_destination_overlaps_s0(self):
        lines = program(self.S0 & 0xFFFFFFFF, self.S0 >> 32, 1)[:2] + ["v_trig_preop_f64 v[5:6], v[4:5], 1"]
        self.assertEqual(self.export(lines, "v5, v6, v5, v6")[0], ref_trig_preop(1, self.S0))

    def test_two_uses_in_one_program(self):
        lines = program(self.S0 & 0xFFFFFFFF, self.S0 >> 32, 0)[:2] + [
            "v_trig_preop_f64 v[0:1], v[4:5], 0", "v_trig_preop_f64 v[2:3], v[4:5], 2"]
        got = self.export(lines, "v0, v1, v2, v3")
        self.assertEqual(got, [ref_trig_preop(0, self.S0), ref_trig_preop(2, self.S0)])


class TrigPreopRefusals(unittest.TestCase):
    def refused(self, line, text=""):
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            run([line])
        self.assertIn(text, str(ctx.exception))

    def test_segments_three_to_thirty_one(self):
        # From a shift of 1120 the ISA text and LLVM's constant folding disagree (docs/design.md).
        for c in range(3, 32):
            self.refused(f"v_trig_preop_f64 v[0:1], v[4:5], {c}", "segment S1[4:0]")
        self.refused("v_trig_preop_f64 v[0:1], v[4:5], -1", "segment S1[4:0] = 31")
        self.refused("v_trig_preop_f64 v[0:1], v[4:5], 0x23", "segment S1[4:0] = 3")

    def test_register_or_float_segment(self):
        for s1 in ("v4", "s4", "1.0", "-v4"):
            self.refused(f"v_trig_preop_f64 v[0:1], v[4:5], {s1}", "not an integer constant")

    def test_modifiers(self):
        for mod in ("clamp", "mul:2", "mul:4", "div:2"):
            self.refused(f"v_trig_preop_f64 v[0:1], v[4:5], 0 {mod}", "modifiers")

    def test_operand_shapes(self):
        self.refused("v_trig_preop_f64 v0, v[4:5], 0", "destination")
        self.refused("v_trig_preop_f64 s[0:1], v[4:5], 0", "destination")
        self.refused("v_trig_preop_f64 v[0:1], v4, 0", "operand")


class TrigPreopModule(unittest.TestCase):
    def test_capabilities_and_spirv_val(self):
        spv = support.translate_lines("ps_trig_preop_f64", program(0, 0x3FF00000, 1)
                                      + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertNotIn("OpCapability Float64", spv)
        self.assertNotIn("OpCapability Int64", spv)
        self.assertIn("%preop_tab = OpVariable", spv)
        spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
        if not spirv_as or not spirv_val:
            self.skipTest("spirv-as / spirv-val not installed")
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            r = subprocess.run([spirv_as, "--target-env", "vulkan1.2", src, "-o", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
