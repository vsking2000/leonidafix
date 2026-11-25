"""v_fract_f64 (SHADER-FRACT-F64): D.f64 = S0.f64 - floor(S0.f64) through the translator.

Every expected word below is written by hand from the rule in docs/design.md ("v_fract_f64"), not computed by the
code under test. The rule is the RDNA2 ISA's `D0.f64 = S0.f64 + -floor(S0.f64)` with the edges that were measured for
v_fract_f32 and v_fract_f16 on gfx1036: -0 and an exact integer give +0, a tiny negative source stops at the largest
double below 1, a NaN comes back unchanged and fract(+-inf) is the negative quiet NaN. None of these edges has been
measured for the binary64 form itself.

Not asserted here: `clamp` of a NaN (an infinity or a NaN source). Gen.clamp_mods_f64 leaves that to FClamp until the
DX10_CLAMP change to it (SHADER-CLAMP-NAN-F64-PK) is on main; the clamp cases below use finite sources only.
"""
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spirv_mini_interp
import support
from test_f64_round import driver_that_quiets_and_flushes

POS_ZERO, NEG_ZERO = 0x0000000000000000, 0x8000000000000000
BELOW_ONE = 0x3FEFFFFFFFFFFFFF           # the largest double below 1.0
NEG_QNAN = 0xFFF8000000000000            # fract(+-inf)
SIGN = 0x8000000000000000


def run_bits(instruction, bits, extra=()):
    """v[4:5] after `instruction`, with v[0:1] = `bits`, as one 64-bit pattern."""
    low, high = support.mrt0_words(support.translate_lines("ps_fract_f64_case", [
        f"v_mov_b32 v0, {bits & 0xFFFFFFFF:#x}", f"v_mov_b32 v1, {bits >> 32:#x}",
        "v_mov_b32 v4, 0x11223344", "v_mov_b32 v5, 0x55667788", *extra,
        instruction, "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm",
    ]))[:2]
    return low | high << 32


def run_const(instruction):
    """v[4:5] after `instruction` (which names its own source), as one 64-bit pattern."""
    low, high = support.mrt0_words(support.translate_lines("ps_fract_f64_const", [
        "v_mov_b32 v4, 0x11223344", "v_mov_b32 v5, 0x55667788",
        instruction, "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm",
    ]))[:2]
    return low | high << 32


FRACT = "v_fract_f64_e32 v[4:5], v[0:1]"


class FractF64Values(unittest.TestCase):
    def check(self, cases, instruction=FRACT):
        for source, want in cases:
            with self.subTest(source=hex(source)):
                self.assertEqual(run_bits(instruction, source), want, hex(run_bits(instruction, source)))

    def test_positive_values_give_their_fraction(self):
        self.check((
            (0x3FF4000000000000, 0x3FD0000000000000),      # 1.25 -> 0.25
            (0x400E000000000000, 0x3FE8000000000000),      # 3.75 -> 0.75
            (0x3FB999999999999A, 0x3FB999999999999A),      # 0.1 -> 0.1, a value below 1 is unchanged
            (0x3FEFFFFFFFFFFFFF, 0x3FEFFFFFFFFFFFFF),      # the largest double below 1
            (0x3FF0000000000001, 0x3CB0000000000000),      # 1 + 2^-52 -> 2^-52
            (0x432FFFFFFFFFFFFF, 0x3FE0000000000000),      # 2^52 - 0.5 -> 0.5, the last binade with a fraction
        ))

    def test_negative_values_give_one_minus_the_magnitude_fraction(self):
        self.check((
            (0xBFD0000000000000, 0x3FE8000000000000),      # -0.25 -> 0.75
            (0xBFF4000000000000, 0x3FE8000000000000),      # -1.25 -> 0.75
            (0xBFE8000000000000, 0x3FD0000000000000),      # -0.75 -> 0.25
            (0xBFB999999999999A, 0x3FECCCCCCCCCCCCD),      # -0.1 -> 0.9, 1 - 0.1 rounded once to nearest
            (0xBFF0000000000001, 0x3FEFFFFFFFFFFFFE),      # -(1 + 2^-52) -> 1 - 2^-52, exact
            (0xC32FFFFFFFFFFFFF, 0x3FE0000000000000),      # -(2^52 - 0.5) -> 0.5
        ))

    def test_exact_integers_and_large_values_give_plus_zero(self):
        self.check((
            (0x3FF0000000000000, POS_ZERO),                # 1.0
            (0xC008000000000000, POS_ZERO),                # -3.0
            (0x4330000000000000, POS_ZERO),                # 2^52: every double from here up is an integer
            (0xC330000000000000, POS_ZERO),                # -2^52
            (0x43B0000000000000, POS_ZERO),                # 2^60
            (0xC3B0000000000000, POS_ZERO),                # -2^60
            (0x7FEFFFFFFFFFFFFF, POS_ZERO),                # the largest finite double
            (0xFFEFFFFFFFFFFFFF, POS_ZERO),                # and its negative
        ))

    def test_signed_zero_gives_plus_zero(self):
        self.check(((POS_ZERO, POS_ZERO), (NEG_ZERO, POS_ZERO)))

    def test_a_tiny_negative_source_stops_below_one(self):
        # 1 - tiny rounds to 1.0; the hardware gives the largest double below 1 (v_fract_f32: 0x3f7fffff, v_fract_f16:
        # 0x3bff, both measured).
        self.check((
            (0xBC30000000000000, BELOW_ONE),               # -2^-60
            (0xBC90000000000000, BELOW_ONE),               # -2^-54: 1 - 2^-54 is a tie and rounds to even, 1.0
            (0xBCA0000000000000, BELOW_ONE),               # -2^-53: 1 - 2^-53 is exactly the largest below 1
            (0xBCB0000000000000, 0x3FEFFFFFFFFFFFFE),      # -2^-52: 1 - 2^-52
        ))

    def test_a_subnormal_keeps_its_value_or_stops_below_one(self):
        self.check((
            (0x0000000000000001, 0x0000000000000001),      # the smallest subnormal: its own fraction, not flushed
            (0x000FFFFFFFFFFFFF, 0x000FFFFFFFFFFFFF),      # the largest subnormal
            (0x0010000000000000, 0x0010000000000000),      # the smallest normal
            (0x8000000000000001, BELOW_ONE),               # -(the smallest subnormal): 1 - tiny
            (0x800FFFFFFFFFFFFF, BELOW_ONE),
            (0x8010000000000000, BELOW_ONE),               # -(the smallest normal)
        ))

    def test_infinities_give_the_negative_quiet_nan(self):
        self.check(((0x7FF0000000000000, NEG_QNAN), (0xFFF0000000000000, NEG_QNAN)))

    def test_a_nan_comes_back_unchanged(self):
        self.check((
            (0x7FF8000000000000, 0x7FF8000000000000),      # quiet
            (0xFFF8000000000000, 0xFFF8000000000000),      # quiet, negative
            (0x7FF0000000000001, 0x7FF0000000000001),      # signalling stays signalling (IEEE = 0)
            (0xFFF0000080000000, 0xFFF0000080000000),
            (0x7FF8DEADBEEF1234, 0x7FF8DEADBEEF1234),      # payload kept
            (0x7FF4000000000000, 0x7FF4000000000000),
        ))


class FractF64Operands(unittest.TestCase):
    def test_source_modifiers_apply_before_the_fraction(self):
        x = 0xBFF4000000000000                              # -1.25
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], -v[0:1]", x), 0x3FD0000000000000)    # 1.25 -> 0.25
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], |v[0:1]|", x), 0x3FD0000000000000)   # 1.25 -> 0.25
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], -|v[0:1]|", x), 0x3FE8000000000000)  # -1.25 -> 0.75
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], -|v[0:1]|", 0x3FF4000000000000), 0x3FE8000000000000)
        # a modifier works on the sign bit alone, so a NaN keeps its payload and an infinity still gives the NaN
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], -v[0:1]", 0x7FF0000000000001), 0xFFF0000000000001)
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], |v[0:1]|", 0xFFF0000080000000), 0x7FF0000080000000)
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], -v[0:1]", 0x7FF0000000000000), NEG_QNAN)
        # the smallest subnormal negated is a negative subnormal
        self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], -v[0:1]", 1), BELOW_ONE)

    def test_literal_inline_and_sgpr_pair_sources(self):
        # one literal dword is bits 63:32 of the constant, with a zero low word
        self.assertEqual(run_const("v_fract_f64_e32 v[4:5], 0x3ff40000"), 0x3FD0000000000000)         # 1.25
        # a literal wider than one dword is the whole constant (docs/design.md, "f64 literal sources")
        self.assertEqual(run_const("v_fract_f64_e32 v[4:5], 0x3ff4000000000001"), 0x3FD0000000000004)  # 1.25 + 2^-52
        self.assertEqual(run_const("v_fract_f64_e32 v[4:5], -0.5"), 0x3FE0000000000000)               # inline float
        self.assertEqual(run_const("v_fract_f64_e32 v[4:5], 4.0"), POS_ZERO)
        # an inline integer is sign-extended to 64 bits: 1 is the smallest subnormal, -1 a NaN
        self.assertEqual(run_const("v_fract_f64_e32 v[4:5], 1"), 0x0000000000000001)
        self.assertEqual(run_const("v_fract_f64_e32 v[4:5], -1"), 0xFFFFFFFFFFFFFFFF)
        for instruction, want in (("v_fract_f64_e32 v[4:5], s[2:3]", 0x3FD0000000000000),
                                  ("v_fract_f64_e64 v[4:5], -s[2:3]", 0x3FE8000000000000)):
            with self.subTest(instruction=instruction):
                self.assertEqual(run_bits(instruction, 0, extra=("s_mov_b32 s2, 0", "s_mov_b32 s3, 0x3ff40000")), want)

    def test_inactive_lane_keeps_both_destination_words(self):
        low, high = support.mrt0_words(support.translate_lines("ps_fract_f64_exec", [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3ff40000",
            "v_mov_b32 v4, 0x11223344", "v_mov_b32 v5, 0x55667788",
            "s_mov_b32 exec_lo, 0", FRACT, "s_mov_b32 exec_lo, -1",
            "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm",
        ]))[:2]
        self.assertEqual((low, high), (0x11223344, 0x55667788))

    def test_destination_pair_may_overlap_the_source_pair(self):
        low, high = support.mrt0_words(support.translate_lines("ps_fract_f64_overlap", [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0xbff40000",         # -1.25 in v[0:1]
            "v_fract_f64_e32 v[1:2], v[0:1]",
            "exp mrt0 v1, v2, v1, v2 done vm", "s_endpgm",
        ]))[:2]
        self.assertEqual(low | high << 32, 0x3FE8000000000000)     # 0.75

    def test_clamp_keeps_a_finite_fraction_and_the_cap(self):
        for source, want in ((0x3FF4000000000000, 0x3FD0000000000000), (0xBFF4000000000000, 0x3FE8000000000000),
                             (POS_ZERO, POS_ZERO), (NEG_ZERO, POS_ZERO), (0xBC30000000000000, BELOW_ONE),
                             (0x0000000000000001, 0x0000000000000001)):
            with self.subTest(source=hex(source)):
                self.assertEqual(run_bits("v_fract_f64_e64 v[4:5], v[0:1] clamp", source), want)

    def test_output_modifiers_scale_nothing(self):
        # A binary64 result ignores mul:2/mul:4/div:2 while denorm_mode_16_64 = 3, the game's MODE
        # (tools/f64_omod_probe.py), so the fraction of 1.25 stays 0.25 and is neither 0.5, 1.0 nor 0.125.
        x = 0x3FF4000000000000
        for mods in ("mul:2", "mul:4", "div:2", "mul:2 clamp", "clamp div:2"):
            with self.subTest(mods=mods):
                self.assertEqual(run_bits(f"v_fract_f64_e64 v[4:5], v[0:1] {mods}", x), 0x3FD0000000000000)
        spv = support.translate_lines("ps_fract_f64_omod", [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3ff40000", "v_fract_f64_e64 v[4:5], v[0:1] mul:4",
            "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])
        self.assertNotIn("OpFMul %double", spv)
        self.assertNotIn("FClamp", spv)

    def test_refused_in_a_program_that_writes_mode_fp_round(self):
        # x - floor(x) of a negative non-integer is rounded, and only v_add_f64 models the directed f64 modes: -0.1 gives
        # 0x3FECCCCCCCCCCCCD to nearest and 0x3FECCCCCCCCCCCCC toward zero, so the program is refused, not translated
        # to the nearest-even word.
        with self.assertRaises(support.rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_fract_f64_round_mode", [
                "s_round_mode 0xc", "v_mov_b32 v0, 0x9999999a", "v_mov_b32 v1, 0xbfb99999", FRACT,
                "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])
        self.assertIn("MODE.FP_ROUND", str(refused.exception))
        # the same program without the write translates
        self.assertEqual(run_bits(FRACT, 0xBFB999999999999A), 0x3FECCCCCCCCCCCCD)

    def test_rejects_forms_it_does_not_translate(self):
        for instruction in ("v_fract_f64 v4, v[0:1]",                       # a 32-bit destination
                            "v_fract_f64 v[4:5], v0",                       # a 32-bit source
                            "v_fract_f64_e32 v[4:5], -v[0:1]",              # a source modifier needs the VOP3 form
                            "v_fract_f64_e64 v[4:5], v[0:1] op_sel:[1,0]",
                            "v_fract_f64_dpp v[4:5], v[0:1] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                            "v_fract_f64_sdwa v[4:5], v[0:1] dst_sel:DWORD src0_sel:DWORD"):
            with self.subTest(instruction=instruction):
                with self.assertRaises(support.rdna2spirv.Unsupported):
                    support.translate_lines("ps_fract_f64_bad", [instruction, "s_endpgm"])


class FractF64LlvmFoldWords(unittest.TestCase):
    """The f64 words of LLVM's constant-folding test for llvm.amdgcn.fract.f64, as literals.

    Origin: llvm/test/Transforms/InstSimplify/ConstProp/AMDGPU/fract.ll, `test_f64` (the fold is in
    llvm/lib/Analysis/ConstantFolding.cpp: fmin(x - floor(x), nextDown(1.0)) in APFloat). A compile-time model, not a
    hardware measurement. The file's other three f64 inputs, +inf, -inf and +qnan, all fold to +qnan (0x7FF8000000000000)
    there; this translator gives the measured f32/f16 words instead (negative quiet NaN for an infinity, a NaN unchanged),
    so they are not pinned here. LLVM's comment calls 2.0e-308 the "min normal": it is a subnormal (0x000E61ACF033D1A4).
    """
    CASES = (
        ("+0.0", 0x0000000000000000, 0x0000000000000000),
        ("-0.0", 0x8000000000000000, 0x0000000000000000),
        ("+1.0", 0x3FF0000000000000, 0x0000000000000000),
        ("-1.0", 0xBFF0000000000000, 0x0000000000000000),
        ("+2.25", 0x4002000000000000, 0x3FD0000000000000),
        ("-6.125", 0xC018800000000000, 0x3FEC000000000000),
        ("+2.0e-308", 0x000E61ACF033D1A4, 0x000E61ACF033D1A4),
        ("-2.0e-308", 0x800E61ACF033D1A4, 0x3FEFFFFFFFFFFFFF),
    )

    def test_translator_reproduces_llvms_fold_words(self):
        for name, source, want in self.CASES:
            with self.subTest(name=name):
                self.assertEqual(run_bits(FRACT, source), want)


class FractF64SpecialSources(unittest.TestCase):
    """A NaN, an infinity and a subnormal are selected from the operand's words, not taken from the driver's Floor:
    under a driver that quiets a NaN and reads a subnormal as zero (the model of tests/semantic/test_f64_round.py,
    from what cs_w32_round_f64 measured on an RTX 4070 Ti and llvmpipe) the words are still the ones above."""

    def test_the_driver_model_changes_nothing(self):
        cases = ((0x7FF0000000000001, 0x7FF0000000000001), (0xFFF0000080000000, 0xFFF0000080000000),
                 (0x7FF0000000000000, NEG_QNAN), (0xFFF0000000000000, NEG_QNAN),
                 (0x0000000000000001, 0x0000000000000001), (0x000FFFFFFFFFFFFF, 0x000FFFFFFFFFFFFF),
                 (0x8000000000000001, BELOW_ONE), (0x800FFFFFFFFFFFFF, BELOW_ONE),
                 (0xBFF4000000000000, 0x3FE8000000000000), (0x3FF4000000000000, 0x3FD0000000000000))
        with driver_that_quiets_and_flushes() as calls:
            for source, want in cases:
                with self.subTest(source=hex(source)):
                    self.assertEqual(run_bits(FRACT, source), want)
        self.assertTrue(calls, "the driver's Floor still ran; its answer for those sources is what is not used")

    # Which tests would pass without the infinity select: on this interpreter inf - inf is x86's negative NaN
    # (spirv_mini_interp.ieee_nan), which is the expected word, so test_infinities_give_the_negative_quiet_nan and the
    # infinity rows of test_the_driver_model_changes_nothing would pass even if the translated code left an infinity to
    # the subtraction. Only the test below, which makes that NaN the positive one, fails without the select (checked by
    # mutation: two failures, both its sources).
    def test_an_infinity_gives_the_negative_nan_whatever_nan_the_subtraction_makes(self):
        # inf - inf is left to the hardware by the language: the interpreter returns x86's negative NaN, an ARM
        # CPU or a driver may return the positive one. The translated code does not depend on it.
        with patch.object(spirv_mini_interp, "ieee_nan", lambda r, *operands: math.nan if r != r else r):
            for source in (0x7FF0000000000000, 0xFFF0000000000000):
                with self.subTest(source=hex(source)):
                    self.assertEqual(run_bits(FRACT, source), NEG_QNAN)

    def test_a_zero_source_gives_plus_zero_whatever_floor_of_minus_zero_is(self):
        # Floor(-0.0) is a zero of either sign by the language. A driver that returns +0 makes -0 - (+0) = -0 unless the
        # zero is selected from the operand's words.
        class Math:
            def __getattr__(self, name):
                return getattr(math, name)

            @staticmethod
            def copysign(magnitude, sign):
                return 0.0 if magnitude == 0.0 else math.copysign(magnitude, sign)

        with patch.object(spirv_mini_interp, "math", Math()):
            for source in (POS_ZERO, NEG_ZERO):
                with self.subTest(source=hex(source)):
                    self.assertEqual(run_bits(FRACT, source), POS_ZERO)
            self.assertEqual(run_bits(FRACT, 0x3FF4000000000000), 0x3FD0000000000000)   # the model changes nothing else

    def test_the_selection_is_what_keeps_a_negative_subnormal_below_one(self):
        # Without it the driver model gives Floor(-subnormal) = -0 and a result of -subnormal; this is the check that
        # the model would catch a translation that left the subnormal to Floor.
        with driver_that_quiets_and_flushes():
            self.assertEqual(run_bits(FRACT, 0x8000000000000001), BELOW_ONE)


class FractF64Module(unittest.TestCase):
    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "SPIRV-Tools not installed")
    def test_spirv_validates(self):
        for mods in ("", " clamp"):
            with self.subTest(mods=mods):
                spv = support.translate_lines("ps_fract_f64_val", [
                    "v_mov_b32 v0, 0", "v_mov_b32 v1, 0xbff40000", f"v_fract_f64_e64 v[4:5], -|v[0:1]|{mods}",
                    "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])
                with tempfile.TemporaryDirectory() as directory:
                    asm = Path(directory) / "shader.spvasm"
                    binary = asm.with_suffix(".spv")
                    asm.write_text(spv, encoding="ascii")
                    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", str(asm), "-o", str(binary)], check=True)
                    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(binary)], check=True)

    def test_float64_is_declared(self):
        spv = support.translate_lines("ps_fract_f64_cap", [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3ff40000", FRACT, "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])
        self.assertIn("OpCapability Float64", spv)


if __name__ == "__main__":
    unittest.main()
