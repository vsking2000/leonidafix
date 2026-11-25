"""The six g16 forms with a compare and/or a texel offset (SHADER-SAMPLE-G16-GRAD).

image_sample_c_d_g16, image_sample_c_d_cl_g16, image_sample_d_o_g16, image_sample_d_cl_o_g16, image_sample_c_d_o_g16
and image_sample_c_d_cl_o_g16. Standard library only (spirv-val is used when it is on PATH); no GPU, no LLVM, no game
files.

The expected positions are written by hand from the address order two independent compilers emit:
* LLVM `SIISelLowering.cpp` `lowerImage`. The extra address arguments come first, pushed whole: the offset, then
  zcompare (`IntrinsicsAMDGPU.td` `AMDGPUSampleVariants`); only a 16-bit bias is padded to a dword. Then the gradients,
  packed when IsG16 (`packImage16bitOpsToDwords`: two halves per VGPR, low first, each axis padded). Then the
  coordinates and the clamp, packed when IsA16.
* Mesa ACO `aco_select_nir.cpp`. The comparator and the offset are asserted to be 32-bit and emplaced whole: offset,
  bias, compare. Then ddx and ddy, packed by one `emit_pack_v1` call each when g16. Then the coordinates (packed with
  the clamp when a16).
So: offset word, compare word, ceil(n/2) VGPRs per gradient axis, then coordinates and clamp. The compare and the
offset stay whole under a16 and g16. The tests trace each SPIR-V operand back to the VGPR and half it is read from.
Not run on a GPU.
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support
import rdna2spirv

DESC = "s[8:15], s[16:19]"
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)
V = 30


def translate(name, line):
    fssm = CS_FSSM if name.startswith("cs_") else None
    body = [line, "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"] if name.startswith("ps_") else [line, "s_endpgm"]
    return support.translate_lines(name, body, fssm=fssm)


def the_sample(spv):
    (line,) = [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]
    return line


def defs(spv):
    return {m.group(1): (m.group(2), m.group(3).split()) for m in re.finditer(r"^\s*(%\w+) = (\w+) (.*)$", spv, re.M)}


def consts(spv):
    return {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}


def vgprs_behind(spv, value):
    """VGPRs a value is built from: k for a whole VGPR, (k, half) for a half widened by UnpackHalf2x16."""
    d, c = defs(spv), consts(spv)

    def walk(v):
        op, args = d[v]
        if op == "OpLoad":
            chain_op, chain = d[args[1]]
            assert chain_op == "OpAccessChain" and chain[1] == "%vgpr", (chain_op, chain)
            return [c[chain[2]]]
        if op == "OpBitcast":
            return walk(args[1])
        if op == "OpCompositeConstruct":
            return [x for a in args[1:] for x in walk(a)]
        if op == "OpCompositeExtract":
            inner = d[args[1]]
            if inner[0] == "OpExtInst" and inner[1][2] == "UnpackHalf2x16":
                (k,) = walk(inner[1][3])
                return [(k, int(args[2]))]
            return [walk(args[1])[int(args[2])]]
        raise AssertionError(f"{v}: {op} {args}")
    return walk(value)


def offset_source(spv, value):
    """(VGPR, [bit offsets]) the texel-offset operand is extracted from (6-bit OpBitFieldSExtract fields)."""
    d, c = defs(spv), consts(spv)
    op, args = d[value]
    parts = args[1:] if op == "OpCompositeConstruct" else [value]
    src, bits = None, []
    for p in parts:
        op, a = d[p]
        if op == "OpBitcast":
            op, a = d[a[1]]
        assert op == "OpBitFieldSExtract" and c[a[3]] == 6, (op, a)
        bits.append(c[a[2]])
        src = vgprs_behind(spv, a[1])[0]
    return src, bits


def operands(line):
    """(coordinate, compare or None, {image operand: [values]})."""
    toks = line.split()
    dref = "Dref" in toks[2]
    coord, ref = toks[5], (toks[6] if dref else None)
    rest = toks[7:] if dref else toks[6:]
    masks, vals, out = rest[0].split("|"), rest[1:], {}
    for m in masks:
        n = 2 if m == "Grad" else 1
        out[m], vals = vals[:n], vals[n:]
    assert not vals, line
    return coord, ref, out


def regs(n):
    return f"v[{V}:{V + n - 1}]"


H = lambda k, half: (V + k, half)    # noqa: E731
W = lambda k: V + k                  # noqa: E731

# (op, dim, a16) -> address VGPR count and the hand-written layout (offset, compare, dx, dy, coordinate, clamp)
CASES = {
    ("image_sample_c_d_g16", "2D", False): (5, dict(ref=W(0), dx=[H(1, 0), H(1, 1)], dy=[H(2, 0), H(2, 1)],
                                                    uv=[W(3), W(4)])),
    ("image_sample_c_d_cl_g16", "2D", False): (6, dict(ref=W(0), dx=[H(1, 0), H(1, 1)], dy=[H(2, 0), H(2, 1)],
                                                       uv=[W(3), W(4)], clamp=W(5))),
    ("image_sample_d_o_g16", "2D", False): (5, dict(off=W(0), dx=[H(1, 0), H(1, 1)], dy=[H(2, 0), H(2, 1)],
                                                    uv=[W(3), W(4)])),
    ("image_sample_d_cl_o_g16", "2D", False): (6, dict(off=W(0), dx=[H(1, 0), H(1, 1)], dy=[H(2, 0), H(2, 1)],
                                                       uv=[W(3), W(4)], clamp=W(5))),
    ("image_sample_c_d_o_g16", "2D", False): (6, dict(off=W(0), ref=W(1), dx=[H(2, 0), H(2, 1)],
                                                      dy=[H(3, 0), H(3, 1)], uv=[W(4), W(5)])),
    ("image_sample_c_d_cl_o_g16", "2D", False): (7, dict(off=W(0), ref=W(1), dx=[H(2, 0), H(2, 1)],
                                                         dy=[H(3, 0), H(3, 1)], uv=[W(4), W(5)], clamp=W(6))),
    # a16 and g16 together: the offset and the compare stay whole, the coordinates and the clamp are halves.
    ("image_sample_c_d_g16", "2D", True): (4, dict(ref=W(0), dx=[H(1, 0), H(1, 1)], dy=[H(2, 0), H(2, 1)],
                                                   uv=[H(3, 0), H(3, 1)])),
    ("image_sample_c_d_cl_o_g16", "2D", True): (6, dict(off=W(0), ref=W(1), dx=[H(2, 0), H(2, 1)],
                                                        dy=[H(3, 0), H(3, 1)], uv=[H(4, 0), H(4, 1)],
                                                        clamp=H(5, 0))),
    ("image_sample_d_cl_o_g16", "1D", True): (4, dict(off=W(0), dx=[H(1, 0)], dy=[H(2, 0)], uv=[H(3, 0)],
                                                      clamp=H(3, 1))),
    # other dims
    ("image_sample_c_d_o_g16", "1D", False): (5, dict(off=W(0), ref=W(1), dx=[H(2, 0)], dy=[H(3, 0)], uv=[W(4)])),
    ("image_sample_c_d_cl_g16", "2D_ARRAY", False): (7, dict(ref=W(0), dx=[H(1, 0), H(1, 1)],
                                                             dy=[H(2, 0), H(2, 1)], uv=[W(3), W(4), W(5)],
                                                             clamp=W(6))),
    ("image_sample_d_cl_o_g16", "3D", False): (9, dict(off=W(0), dx=[H(1, 0), H(1, 1), H(2, 0)],
                                                       dy=[H(3, 0), H(3, 1), H(4, 0)], uv=[W(5), W(6), W(7)],
                                                       clamp=W(8))),
    ("image_sample_d_o_g16", "3D", True): (7, dict(off=W(0), dx=[H(1, 0), H(1, 1), H(2, 0)],
                                                   dy=[H(3, 0), H(3, 1), H(4, 0)], uv=[H(5, 0), H(5, 1), H(6, 0)])),
}
OFFSET_BITS = {"1D": [0], "2D": [0, 8], "2D_ARRAY": [0, 8], "3D": [0, 8, 16]}


def line_for(op, dim, a16, n):
    dst, dmask = ("v20", "0x1") if "_c_" in op else ("v[20:23]", "0xf")
    return f"{op} {dst}, {regs(n)}, {DESC} dmask:{dmask} dim:SQ_RSRC_IMG_{dim}{' a16' if a16 else ''}"


class G16GradLayout(unittest.TestCase):
    def check(self, stage, op, dim, a16):
        n, want = CASES[(op, dim, a16)]
        spv = translate(stage, line_for(op, dim, a16, n))
        line = the_sample(spv)
        coord, ref, ops = operands(line)
        self.assertEqual("Dref" in line, "ref" in want, line)
        self.assertIn("ExplicitLod", line)
        self.assertEqual(vgprs_behind(spv, ops["Grad"][0]), want["dx"])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][1]), want["dy"])
        self.assertEqual(vgprs_behind(spv, coord), want["uv"])
        if "ref" in want:
            self.assertEqual(vgprs_behind(spv, ref), [want["ref"]])
        if "off" in want:
            self.assertEqual(offset_source(spv, ops["Offset"][0]), (want["off"], OFFSET_BITS[dim]))
        else:
            self.assertNotIn("Offset", ops)
        if "clamp" in want:
            self.assertEqual(vgprs_behind(spv, ops["MinLod"][0]), [want["clamp"]])
        else:
            self.assertNotIn("MinLod", ops)
        masks = line.split()[7 if "Dref" in line else 6].split("|")
        self.assertEqual(masks, [m for m in ("Grad", "Offset", "MinLod") if m in masks])

    def test_every_case_in_both_stages(self):
        for (op, dim, a16) in CASES:
            for stage in ("ps_g16g", "cs_g16g"):
                with self.subTest(op=op, dim=dim, a16=a16, stage=stage):
                    self.check(stage, op, dim, a16)

    def test_all_six_opcodes_are_covered(self):
        self.assertEqual({op for op, _, _ in CASES},
                         {f"{op}_g16" for op in rdna2spirv.Gen.SAMPLE_GRAD_FORMS})

    def test_cube_compare(self):
        spv = translate("ps_cube", f"image_sample_c_d_g16 v20, {regs(6)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_CUBE")
        _coord, ref, ops = operands(the_sample(spv))
        self.assertEqual(vgprs_behind(spv, ref), [W(0)])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][0]), [H(1, 0), H(1, 1)])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][1]), [H(2, 0), H(2, 1)])

    def test_d16_compare(self):
        spv = translate("ps_d16", f"image_sample_c_d_o_g16 v20, {regs(6)}, {DESC} dmask:0x1 "
                                  "dim:SQ_RSRC_IMG_2D d16")
        res = the_sample(spv).split()[0]
        stored = re.search(r"OpStore %reg_v20 (%\w+)", spv).group(1)
        self.assertNotEqual(defs(spv)[stored], ("OpBitcast", ["%uint", res]))


class Refusals(unittest.TestCase):
    def refused(self, line, text=""):
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            translate("ps_no", line)
        self.assertIn(text, str(ctx.exception))

    def test_coarse_derivative_forms(self):
        for op in ("image_sample_cd_g16", "image_sample_cd_cl_g16", "image_sample_c_cd_g16", "image_sample_c_cd_cl_g16",
                   "image_sample_cd_o_g16", "image_sample_cd_cl_o_g16", "image_sample_c_cd_o_g16",
                   "image_sample_c_cd_cl_o_g16"):
            with self.subTest(op=op):
                self.refused(line_for(op, "2D", False, 7))

    def test_compare_on_3d(self):
        self.refused(f"image_sample_c_d_g16 v20, {regs(8)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D", "3D")

    def test_offset_fallback(self):
        saved = rdna2spirv.TEXEL_OFFSET_FALLBACK
        rdna2spirv.TEXEL_OFFSET_FALLBACK = True
        try:
            self.refused(line_for("image_sample_d_o_g16", "2D", False, 5), "texel-offset-fallback")
        finally:
            rdna2spirv.TEXEL_OFFSET_FALLBACK = saved

    def test_short_address(self):
        self.refused(f"image_sample_c_d_o_g16 v20, v[{V}:{V + 2}], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")


class Validation(unittest.TestCase):
    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        for (op, dim, a16), (n, _) in CASES.items():
            for stage in ("ps_val", "cs_val"):
                with self.subTest(op=op, dim=dim, a16=a16, stage=stage):
                    spv = translate(stage, line_for(op, dim, a16, n))
                    with tempfile.TemporaryDirectory() as tmp:
                        a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                        with open(a, "w", encoding="utf-8", newline="\n") as f:
                            f.write(spv)
                        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
                        # a register texel offset needs VK_KHR_maintenance8, as for the other _o forms
                        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", b],
                                       check=True)


if __name__ == "__main__":
    unittest.main()
