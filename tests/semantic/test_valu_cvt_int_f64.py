"""v_cvt_i32_f64 and v_cvt_u32_f64 through the translator (SHADER-F64-VALU-OPS).

RDNA2 ISA, V_CVT_I32_F64 / V_CVT_U32_F64: D = (int)S0.d, truncating toward zero. "Out-of-range floating point values
(including infinity) saturate. NaN is converted to 0", as v_cvt_i32_f32/v_cvt_u32_f32 do (docs/design.md,
"Binary64 multiply, FMA, min/max and conversions"). v_cvt_f64_i32/v_cvt_f64_u32 are in test_int32_to_f64.py.

This file used to assert that the translator refuses both ops by name, so that implementing them would replace it
with a run through the translator against these rules; this is that run. Expected words are written by hand.
Standard library only; no game data.
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on the path)
import rdna2spirv  # noqa: E402

INF = float("inf")
QNAN = 0x7FF8000000000000
NEG_QNAN = 0xFFF8000000000000
SNAN = 0x7FF0000000000001


def run(op_line, x):
    """The word left in v2 after op_line with v[0:1] = x (a Python float, or an int taken as the binary64 word)."""
    w = x if isinstance(x, int) else struct.unpack("<Q", struct.pack("<d", x))[0]
    lines = [f"v_mov_b32 v0, {w & 0xFFFFFFFF:#x}", f"v_mov_b32 v1, {w >> 32:#x}", "v_mov_b32 v2, 0xdeadbeef",
             op_line, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"]
    return mrt0_words(translate_lines("ps_cvt_int_f64", lines))[0]


I32 = "v_cvt_i32_f64_e32 v2, v[0:1]"
U32 = "v_cvt_u32_f64_e32 v2, v[0:1]"


class CvtI32F64(unittest.TestCase):
    def test_truncates_toward_zero(self):
        for x, want in ((0.0, 0), (-0.0, 0), (1.9999, 1), (-1.9999, 0xFFFFFFFF), (-0.5, 0), (123456789.75, 123456789),
                        (-2147483648.0, 0x80000000), (2147483647.0, 0x7FFFFFFF), (2147483647.9, 0x7FFFFFFF),
                        (-2147483648.9, 0x80000000)):
            with self.subTest(x=x):
                self.assertEqual(run(I32, x), want)

    def test_saturates(self):
        for x, want in ((2147483648.0, 0x7FFFFFFF), (1e300, 0x7FFFFFFF), (INF, 0x7FFFFFFF),
                        (-2147483649.0, 0x80000000), (-1e300, 0x80000000), (-INF, 0x80000000)):
            with self.subTest(x=x):
                self.assertEqual(run(I32, x), want)

    def test_nan_gives_zero(self):
        for nan in (QNAN, NEG_QNAN, SNAN):
            with self.subTest(nan=hex(nan)):
                self.assertEqual(run(I32, nan), 0)

    def test_source_modifiers_and_literal(self):
        self.assertEqual(run("v_cvt_i32_f64_e64 v2, -|v[0:1]|", 7.5), 0xFFFFFFF9)
        self.assertEqual(run("v_cvt_i32_f64_e64 v2, -v[0:1]", -7.5), 7)
        # The literal dword is bits 63:32 of the f64: 0xc0080000 is -3.0.
        self.assertEqual(run("v_cvt_i32_f64_e32 v2, 0xc0080000", 0.0), 0xFFFFFFFD)


class CvtU32F64(unittest.TestCase):
    def test_truncates_and_saturates(self):
        for x, want in ((0.0, 0), (1.9999, 1), (4294967295.0, 0xFFFFFFFF), (4294967295.5, 0xFFFFFFFF),
                        (4294967296.0, 0xFFFFFFFF), (INF, 0xFFFFFFFF), (-0.5, 0), (-1.0, 0), (-INF, 0),
                        (3000000000.25, 3000000000)):
            with self.subTest(x=x):
                self.assertEqual(run(U32, x), want)

    def test_nan_gives_zero(self):
        for nan in (QNAN, NEG_QNAN, SNAN):
            with self.subTest(nan=hex(nan)):
                self.assertEqual(run(U32, nan), 0)

    def test_abs_source(self):
        self.assertEqual(run("v_cvt_u32_f64_e64 v2, |v[0:1]|", -12.5), 12)


class Refusals(unittest.TestCase):
    def test_output_modifiers_are_refused(self):
        # An integer result takes no clamp or omod here.
        for line in ("v_cvt_i32_f64_e64 v2, v[0:1] clamp", "v_cvt_u32_f64_e64 v2, v[0:1] mul:2"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_cvt_int_f64", ["v_mov_b32 v1, 5", line, "exp mrt0 v2, v2, v2, v2 done vm",
                                                       "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
