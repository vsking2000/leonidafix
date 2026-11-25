"""VOP3 `clamp` of a NaN result gives +0 (Gen.clamp_mods, Gen.clamp_mods_f64 and the packed f16 clamp).

The game's MODE has DX10_CLAMP 1, and with it the clamp bit turns a NaN result into +0: measured on RDNA2 (gfx1036)
for v_rcp_iflag_f32 (docs/shader_validation.md, "32-bit float VALU on the GPU": a NaN result +0 except with DX10_CLAMP
off, 169 programs) and for v_fmac_f16, v_fma_mixlo_f16 and v_fma_mix_f32 (SHADER-186, "f16 output modifiers, clamp and
v_fma_mix* operands on RDNA2").
GLSL.std.450 leaves FClamp undefined for a NaN operand, so the translator selects +0 for a NaN before FClamp, as the
div_fixup and cvt_pknorm clamps already did. The f64 forms (Gen.clamp_mods_f64) and the packed f16 forms (v_pk_add/mul/
fma/min/max_f16, per half) follow the same ISA rule, MODE.DX10_CLAMP; neither has been measured on hardware itself
(docs/design.md says so too), so ClampNanF64Tests and ClampNanPackedF16Tests check the translator against that rule only. Before that the shared helper emitted a bare FClamp, which the CPU
interpreter refuses (UnsupportedConstruct) and which a driver may return as the NaN.

Each program computes one value in v4 and exports it. The NaN comes from a NaN operand (quiet, negative, signalling)
or from an invalid operation (inf - inf, -inf * 0); controls without a NaN check that the clamp itself is unchanged.
Standard library only; no game data.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from support import mrt0_words, translate_lines  # noqa: E402

QNAN = 0x7FC00000
NEG_QNAN = 0xFFC00000
SNAN = 0x7F800001
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
ONE = 0x3F800000
TWO = 0x40000000
QUARTER = 0x3E800000
NEG_ONE = 0xBF800000
F16_QNAN = 0x7E00
F16_NEG_QNAN = 0xFE00
F16_ONE = 0x3C00
F16_HALF = 0x3800


def run(op_line, v1, v2, v3=0):
    """mrt0.x after `op_line` (which writes v4) with v1..v3 set to the given words."""
    spv = translate_lines("ps_clamp_nan", [
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"v_mov_b32 v3, {v3:#x}",
        "v_mov_b32 v4, 0xdeadbeef",
        op_line,
        "exp mrt0 v4, v4, v4, v4 done vm",
        "s_endpgm",
    ])
    return mrt0_words(spv)[0]


class ClampNanF32Tests(unittest.TestCase):
    def test_nan_operand_clamps_to_plus_zero(self):
        for nan in (QNAN, NEG_QNAN, SNAN):
            for line in ("v_add_f32_e64 v4, v1, v2 clamp",
                         "v_sub_f32_e64 v4, v1, v2 clamp",
                         "v_mul_f32_e64 v4, v1, v2 clamp",
                         "v_fma_f32 v4, v1, v2, v2 clamp",
                         "v_mul_f32_e64 v4, v1, v2 mul:2 clamp"):
                with self.subTest(op=line, nan=hex(nan)):
                    self.assertEqual(run(line, nan, ONE), 0)

    def test_invalid_operation_clamps_to_plus_zero(self):
        self.assertEqual(run("v_sub_f32_e64 v4, v1, v2 clamp", POS_INF, POS_INF), 0)       # inf - inf
        self.assertEqual(run("v_add_f32_e64 v4, v1, v2 clamp", POS_INF, NEG_INF), 0)       # inf + -inf
        self.assertEqual(run("v_mul_f32_e64 v4, v1, v2 clamp", NEG_INF, 0), 0)             # -inf * 0
        self.assertEqual(run("v_fma_f32 v4, v1, v2, v3 clamp", POS_INF, 0, ONE), 0)        # inf * 0 + 1

    def test_all_nan_min_max_clamps_to_plus_zero(self):
        # Without clamp an all-NaN min/max keeps src0's NaN (docs/design.md); the clamp then gives +0.
        for op in ("v_min_f32_e64", "v_max_f32_e64"):
            with self.subTest(op=op):
                self.assertEqual(run(f"{op} v4, v1, v2 clamp", QNAN, NEG_QNAN), 0)
        for op in ("v_min3_f32", "v_max3_f32"):
            with self.subTest(op=op):
                self.assertEqual(run(f"{op} v4, v1, v2, v3 clamp", QNAN, SNAN, NEG_QNAN), 0)

    def test_nan_without_clamp_is_still_nan(self):
        # The select is only on the clamp path: without clamp a NaN result stays a NaN.
        w = run("v_add_f32_e64 v4, v1, v2", QNAN, ONE)
        self.assertEqual(w & 0x7F800000, 0x7F800000)
        self.assertNotEqual(w & 0x007FFFFF, 0)

    def test_clamp_controls_unchanged(self):
        self.assertEqual(run("v_add_f32_e64 v4, v1, v2 clamp", TWO, TWO), ONE)
        self.assertEqual(run("v_add_f32_e64 v4, v1, v2 clamp", NEG_ONE, 0), 0)
        self.assertEqual(run("v_add_f32_e64 v4, v1, v2 clamp", QUARTER, 0), QUARTER)
        self.assertEqual(run("v_mul_f32_e64 v4, v1, v2 clamp", POS_INF, ONE), ONE)
        self.assertEqual(run("v_mul_f32_e64 v4, v1, v2 clamp", NEG_INF, ONE), 0)


    def test_legacy_ops_nan_clamps_to_plus_zero(self):
        # v_mul_legacy_f32 ... clamp is the most common clamp form in the game's listings (docs/design.md).
        self.assertEqual(run("v_mul_legacy_f32_e64 v4, v1, v2 clamp", QNAN, ONE), 0)
        self.assertEqual(run("v_mad_legacy_f32 v4, v1, v2, v3 clamp", QNAN, ONE, 0), 0)


# An f16 result written to the low half keeps the high half of v4 (0xdead); op_sel:[0,0,1] and v_fma_mixhi_f16 write
# the high half and keep the low one (0xbeef).
LO_ZERO = 0xDEAD0000
HI_ZERO = 0x0000BEEF


class ClampNanF16Tests(unittest.TestCase):
    def test_f16_nan_clamps_to_plus_zero(self):
        for nan in (F16_QNAN, F16_NEG_QNAN):
            for line in ("v_add_f16_e64 v4, v1, v2 clamp",
                         "v_mul_f16_e64 v4, v1, v2 clamp",
                         "v_fmac_f16_e64 v4, v1, v2 clamp",
                         "v_fma_f16 v4, v1, v2, v3 clamp"):
                with self.subTest(op=line, nan=hex(nan)):
                    self.assertEqual(run(line, nan, F16_ONE), LO_ZERO)

    def test_f16_high_half_nan_clamps_to_plus_zero(self):
        self.assertEqual(run("v_add_f16_e64 v4, v1, v2 op_sel:[0,0,1] clamp", F16_QNAN, F16_ONE), HI_ZERO)
        self.assertEqual(run("v_fma_mixhi_f16 v4, v1, v2, v3 clamp", QNAN, ONE, 0), HI_ZERO)

    def test_f16_all_nan_med3_clamps_to_plus_zero(self):
        self.assertEqual(run("v_med3_f16 v4, v1, v2, v3 clamp", F16_QNAN, F16_NEG_QNAN, 0x7E01), LO_ZERO)

    def test_conversions_nan_clamp_to_plus_zero(self):
        self.assertEqual(run("v_cvt_f32_f16_e64 v4, v1 clamp", F16_QNAN, 0), 0)
        self.assertEqual(run("v_cvt_f16_f32_e64 v4, v1 clamp", QNAN, 0), LO_ZERO)
        self.assertEqual(run("v_cvt_f16_f32_e64 v4, v1 mul:2 clamp", NEG_QNAN, 0), LO_ZERO)

    def test_mix_ops_nan_clamp_to_plus_zero(self):
        # SHADER-186 measured +0 on gfx1036 for these (docs/shader_validation.md).
        self.assertEqual(run("v_fma_mix_f32 v4, v1, v2, v3 op_sel_hi:[1,1,1] clamp", F16_QNAN, F16_ONE, 0), 0)
        self.assertEqual(run("v_fma_mixlo_f16 v4, v1, v2, v3 clamp", QNAN, ONE, 0), LO_ZERO)

    def test_f16_clamp_controls_unchanged(self):
        self.assertEqual(run("v_add_f16_e64 v4, v1, v2 clamp", F16_ONE, F16_ONE), 0xDEAD0000 | F16_ONE)
        self.assertEqual(run("v_add_f16_e64 v4, v1, v2 clamp", F16_HALF, 0), 0xDEAD0000 | F16_HALF)
        self.assertEqual(run("v_fma_mixlo_f16 v4, v1, v2, v3 clamp", TWO, ONE, 0), 0xDEAD0000 | F16_ONE)


# f64 values as (low word, high word).
F64_QNAN = (0x00000000, 0x7FF80000)
F64_NEG_QNAN = (0x00000000, 0xFFF80000)
F64_SNAN = (0x00000001, 0x7FF00000)
F64_POS_INF = (0x00000000, 0x7FF00000)
F64_ONE = (0x00000000, 0x3FF00000)
F64_TWO = (0x00000000, 0x40000000)
F64_QUARTER = (0x00000000, 0x3FD00000)
F64_NEG_ONE = (0x00000000, 0xBFF00000)


def run_f64(op_line, x, y=F64_ONE):
    """(low, high) of v[4:5] after `op_line` with v[0:1] = x and v[2:3] = y."""
    spv = translate_lines("ps_clamp_nan_f64", [
        f"v_mov_b32 v0, {x[0]:#x}",
        f"v_mov_b32 v1, {x[1]:#x}",
        f"v_mov_b32 v2, {y[0]:#x}",
        f"v_mov_b32 v3, {y[1]:#x}",
        "v_mov_b32 v4, 0xdeadbeef",
        "v_mov_b32 v5, 0xdeadbeef",
        op_line,
        "exp mrt0 v4, v5, v4, v5 done vm",
        "s_endpgm",
    ])
    return tuple(mrt0_words(spv)[:2])


class ClampNanF64Tests(unittest.TestCase):
    """Gen.clamp_mods_f64: the same DX10_CLAMP rule as the f32 helper. Until SHADER-CLAMP-NAN-F64-PK it emitted a
    bare FClamp, which the CPU interpreter refuses (UnsupportedConstruct) for a NaN operand."""

    def test_nan_operand_clamps_to_plus_zero(self):
        for nan in (F64_QNAN, F64_NEG_QNAN, F64_SNAN):
            for line in ("v_add_f64 v[4:5], v[0:1], v[2:3] clamp",
                         "v_add_f64 v[4:5], v[0:1], v[2:3] mul:2 clamp",
                         "v_trunc_f64_e64 v[4:5], v[0:1] clamp",
                         "v_rcp_f64_e64 v[4:5], v[0:1] clamp",
                         "v_sqrt_f64_e64 v[4:5], v[0:1] clamp"):
                with self.subTest(op=line, nan=nan):
                    self.assertEqual(run_f64(line, nan), (0, 0))

    def test_other_callers_of_clamp_mods_f64_give_plus_zero_for_a_nan(self):
        # v_frexp_mant_f64 passes a NaN operand through, and v_div_fixup_f64 does for a NaN denominator (S1), so
        # both hand a NaN to clamp_mods_f64. v_cvt_f64_i32/u32 cannot make a NaN, so they are not here.
        for nan in (F64_QNAN, F64_NEG_QNAN, F64_SNAN):
            with self.subTest(op="v_frexp_mant_f64", nan=nan):
                self.assertEqual(run_f64("v_frexp_mant_f64_e64 v[4:5], v[0:1] clamp", nan), (0, 0))
            with self.subTest(op="v_div_fixup_f64 S1", nan=nan):
                self.assertEqual(run_f64("v_div_fixup_f64 v[4:5], v[2:3], v[0:1], v[2:3] clamp", nan), (0, 0))
        # controls: 3.0 has the mantissa 0.75, and -3.0 has -0.75, which clamps to +0
        self.assertEqual(run_f64("v_frexp_mant_f64_e64 v[4:5], v[0:1] clamp", (0, 0x40080000)), (0, 0x3FE80000))
        self.assertEqual(run_f64("v_frexp_mant_f64_e64 v[4:5], v[0:1] clamp", (0, 0xC0080000)), (0, 0))

    def test_invalid_operation_clamps_to_plus_zero(self):
        # inf + -inf
        neg_inf = (0, 0xFFF00000)
        self.assertEqual(run_f64("v_add_f64 v[4:5], v[0:1], v[2:3] clamp", F64_POS_INF, neg_inf), (0, 0))

    def test_nan_without_clamp_is_still_nan(self):
        lo, hi = run_f64("v_add_f64 v[4:5], v[0:1], v[2:3]", F64_QNAN)
        self.assertEqual(hi & 0x7FF00000, 0x7FF00000)
        self.assertNotEqual((hi & 0x000FFFFF) | lo, 0)

    def test_clamp_controls_unchanged(self):
        self.assertEqual(run_f64("v_add_f64 v[4:5], v[0:1], v[2:3] clamp", F64_TWO, F64_TWO), F64_ONE)
        self.assertEqual(run_f64("v_add_f64 v[4:5], v[0:1], v[2:3] clamp", F64_NEG_ONE, (0, 0)), (0, 0))
        self.assertEqual(run_f64("v_add_f64 v[4:5], v[0:1], v[2:3] clamp", F64_QUARTER, (0, 0)), F64_QUARTER)
        self.assertEqual(run_f64("v_add_f64 v[4:5], v[0:1], v[2:3] clamp", F64_POS_INF, F64_ONE), F64_ONE)


F16_POS_INF = 0x7C00
F16_TWO = 0x4000


class ClampNanPackedF16Tests(unittest.TestCase):
    """The packed f16 clamp (v_pk_add/mul/fma_f16 ... clamp): each half as the scalar f16 ops, a NaN half +0.
    Until SHADER-CLAMP-NAN-F64-PK the vec2 went to FClamp with the NaN in it."""

    def test_nan_half_clamps_to_plus_zero_other_half_kept(self):
        for nan in (F16_QNAN, F16_NEG_QNAN):
            # x + 0, x * 1 and x * 1 + 0 keep the other half's 0.5.
            for line, y in (("v_pk_add_f16 v4, v1, v2 clamp", 0),
                            ("v_pk_mul_f16 v4, v1, v2 clamp", F16_ONE << 16 | F16_ONE),
                            ("v_pk_fma_f16 v4, v1, v2, v3 clamp", F16_ONE << 16 | F16_ONE)):
                with self.subTest(op=line, nan=hex(nan), half="lo"):
                    self.assertEqual(run(line, F16_HALF << 16 | nan, y), F16_HALF << 16)
                with self.subTest(op=line, nan=hex(nan), half="hi"):
                    self.assertEqual(run(line, nan << 16 | F16_HALF, y), F16_HALF)

    def test_min_max_nan_halves_clamp_to_plus_zero(self):
        # min/max return the other operand for one NaN, so a half is a NaN only when both operands are: +0 then.
        # Expected words are literals: (src0, src1) -> result. 0x7e00 is a quiet NaN, 0x3800 is 0.5, 0x8000 is -0,
        # 0xbc00 is -1.
        for line in ("v_pk_max_f16 v4, v1, v2 clamp", "v_pk_min_f16 v4, v1, v2 clamp"):
            for src, want, what in ((0x7E007E00, 0x00000000, "both halves NaN"),
                                    (0x38007E00, 0x38000000, "lo half NaN, hi half 0.5"),
                                    (0x7E003800, 0x00003800, "hi half NaN, lo half 0.5"),
                                    (0x80008000, 0x00000000, "-0 in both halves"),
                                    (0xBC00BC00, 0x00000000, "-1 in both halves")):
                with self.subTest(op=line, case=what):
                    self.assertEqual(run(line, src, src), want)
            # one NaN operand per half gives the other operand (0.5), which is in range
            with self.subTest(op=line, case="one NaN operand per half"):
                self.assertEqual(run(line, 0x38007E00, 0x7E003800), 0x38003800)

    def test_invalid_operation_half_clamps_to_plus_zero(self):
        # inf * 0 in the low half, 2 * 1 (clamped to 1) in the high half.
        self.assertEqual(run("v_pk_mul_f16 v4, v1, v2 clamp", F16_TWO << 16 | F16_POS_INF, F16_ONE << 16),
                         F16_ONE << 16)

    def test_packed_clamp_controls_unchanged(self):
        self.assertEqual(run("v_pk_add_f16 v4, v1, v2 clamp", F16_ONE << 16 | F16_HALF, F16_ONE << 16),
                         F16_ONE << 16 | F16_HALF)          # 1 + 1 clamps to 1; 0.5 + 0 stays 0.5
        self.assertEqual(run("v_pk_mul_f16 v4, v1, v2 clamp", 0xBC00 << 16 | F16_HALF, F16_ONE << 16 | F16_ONE),
                         F16_HALF)


if __name__ == "__main__":
    unittest.main()
