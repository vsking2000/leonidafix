"""An inline literal that does not fit in 32 bits on a binary64 VOP3 source is the whole constant.

LLVM's gfx1013 listing prints the two-dword literal of an f64 instruction as one `0x` value of 16 hex
digits, the same form `tools/f64_diff.py` writes. `docs/design.md` ("f64 literal sources") covers the
other encoding: a single literal dword supplies bits 63:32 of the constant with bits 31:0 zero, and an
inline integer is sign-extended. A literal wider than a 32-bit word is therefore the two words of one
double, not the high word of one.

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py and checks the
(low, high) dword pair the module builds for the source. The constants are exact: 0x3FF8000000000001
splits into the low word 0x00000001 and the high word 0x3FF80000, 0x40100000 is the high word of 4.0,
and 4 is 4.0 as an inline integer. The behaviour of a literal source is executed and compared against
the CPU reference in `tools/f64_diff.py` (its own `v_add_f64`/`v_rsq_f64`/`v_sqrt_f64` cases).
Standard library only.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF
ONE_HALF_UP = 0x3FF8000000000001          # 1.5 + 2^-52: not expressible as one literal dword
ONE_HALF_WORDS = (0x00000001, 0x3FF80000)  # (low word, high word) of ONE_HALF_UP
FOUR_HIGH = 0x40100000                    # one literal dword: bits 63:32 of 4.0
UNARY = ("v_sqrt_f64", "v_rsq_f64", "v_rcp_f64", "v_floor_f64", "v_ceil_f64", "v_trunc_f64",
         "v_rndne_f64", "v_fract_f64")


def constants(spv):
    const = {}
    for raw in spv.splitlines():
        lhs, sep, rhs = raw.strip().partition(" = OpConstant %uint ")
        if sep:
            const[lhs] = int(rhs)
    return const


def built_pairs(spv):
    """Every (low, high) dword pair the module builds for a binary64 value, in program order."""
    const, pairs = constants(spv), []
    for raw in spv.splitlines():
        lhs, sep, rhs = raw.strip().partition(" = OpCompositeConstruct %v2uint_d ")
        if sep:
            lo, hi = rhs.split()
            pairs.append((const[lo], const[hi]))
    return pairs


def source_words(op, src):
    """The word pair the translated `op v[4:5], src` builds for its source (its first such pair)."""
    spv = translate_lines("ps_f64_literal_whole",
                          [f"{op} v[4:5], {src}",
                           "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"])
    pairs = built_pairs(spv)
    if not pairs:
        raise AssertionError("the module builds no binary64 operand pair")
    return pairs[0]


def frexp_words(src):
    """(low, high) the translated `v_frexp_exp_i32_f64 v4, src` works on: the exponent field
    extraction names the high word, the fraction OR names the low word."""
    spv = translate_lines("ps_f64_literal_frexp",
                          [f"v_frexp_exp_i32_f64 v4, {src}",
                           "exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"])
    const, high, low = constants(spv), None, None
    for raw in spv.splitlines():
        lhs, sep, rhs = raw.strip().partition(" = ")
        if sep and rhs.startswith("OpBitFieldUExtract %uint ") and high is None:
            high = const[rhs.split()[2]]
        if sep and rhs.startswith("OpBitwiseOr %uint ") and low is None:
            low = const[rhs.split()[3]]
    if high is None or low is None:
        raise AssertionError("the module has no frexp exponent extraction and fraction OR")
    return low, high


class WholeLiteralTests(unittest.TestCase):
    def test_one_literal_dword_is_bits_63_32(self):
        # The 32-bit rule is unchanged: 0x40100000 is the high word and the low word is zero.
        self.assertEqual(source_words("v_sqrt_f64", f"0x{FOUR_HIGH:08x}"), (0, FOUR_HIGH))

    def test_a_printed_width_picks_the_rule_not_the_value(self):
        # 0x12345678 fits one dword, but the 16-digit spelling is the full-width form, so its value
        # is the whole constant: the low word here, not the high one (tools/f64_diff.py writes it).
        self.assertEqual(source_words("v_sqrt_f64", "0x12345678"), (0, 0x12345678))
        self.assertEqual(source_words("v_sqrt_f64", "0x0000000012345678"), (0x12345678, 0))
        self.assertEqual(frexp_words("0x0000000012345678"), (0x12345678, 0))

    def test_inline_integer_is_sign_extended(self):
        self.assertEqual(source_words("v_sqrt_f64", "4"), (4, 0))
        self.assertEqual(source_words("v_sqrt_f64", "-1"), (MASK32, MASK32))

    def test_two_dword_literal_is_the_whole_constant(self):
        # One literal dword would build (0, 1) here: the literal's low 32 bits as the high word.
        for op in UNARY:
            with self.subTest(op=op):
                self.assertEqual(source_words(op, f"0x{ONE_HALF_UP:016x}"), ONE_HALF_WORDS)
        self.assertEqual(frexp_words(f"0x{ONE_HALF_UP:016x}"), ONE_HALF_WORDS)

    def test_a_literal_wider_than_32_bits_translates(self):
        # 0x100000000 used to be refused as `Unsupported: v_ceil_f64 literal`; it is the double
        # 2^32, its two words 0x00000000 and 0x00000001 (tests/semantic/test_f64_round.py kept the refusal).
        for op in ("v_ceil_f64", "v_floor_f64", "v_trunc_f64", "v_rndne_f64", "v_fract_f64"):
            with self.subTest(op=op):
                self.assertEqual(source_words(op, "0x100000000"), (0, 1))


if __name__ == "__main__":
    unittest.main()
