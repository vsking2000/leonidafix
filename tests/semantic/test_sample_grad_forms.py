"""Explicit-gradient samples with a depth compare and/or a texel offset (SHADER-SAMPLE-GRAD-FORMS).

image_sample_c_d, image_sample_c_d_cl, image_sample_d_o, image_sample_d_cl_o, image_sample_c_d_o and
image_sample_c_d_cl_o. Standard library only (spirv-val is used when it is on PATH); no GPU, no LLVM, no game files.

Each expected operand position is written here by hand from the GFX10 MIMG address order that two independent
compilers use for these opcodes. LLVM (IntrinsicsAMDGPU.td AMDGPUSampleVariants: offset, then zcompare as the extra
address arguments; SIISelLowering lowerImage: extra arguments, gradients, coordinates, clamp) and Mesa ACO (its texture
selection: offset, bias, compare, ddx, ddy, coordinates, then the LOD or clamp) give the same order:
offset, z-compare, dP/dx, dP/dy, coordinates, clamp. The tests follow each SPIR-V sample operand back to the VGPR it
was loaded from and compare that with the hand-written position. Not run on a GPU here; the pieces (compare,
gradients, offset, clamp) are the GPU-tested ones of the sibling forms (docs/design.md).
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
FIRST = 30          # the address VGPRs start at v30


def translate(name, line):
    fssm = CS_FSSM if name.startswith("cs_") else None
    body = [line, "exp mrt0 v20, v20, v20, v20 done vm", "s_endpgm"] if name.startswith("ps_") else [line, "s_endpgm"]
    return support.translate_lines(name, body, fssm=fssm)


def the_sample(spv):
    (line,) = [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]
    return line


def defs(spv):
    return {m.group(1): (m.group(2), m.group(3).split()) for m in
            re.finditer(r"^\s*(%\w+) = (\w+) (.*)$", spv, re.M)}


def consts(spv):
    return {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}


def vgprs_behind(spv, value):
    """The VGPR indices a value is built from, through bitcasts, composites, extracts and half unpacking, in operand
    order. A value read from a whole VGPR gives [k]; one of the halves of an UnpackHalf2x16 gives [(k, half)]."""
    d, c = defs(spv), consts(spv)

    def walk(v):
        op, args = d[v]
        if op == "OpLoad":
            src = args[1]
            chain_op, chain = d[src]
            assert chain_op == "OpAccessChain" and chain[1] == "%vgpr", (src, chain_op, chain)
            return [c[chain[2]]]
        if op in ("OpBitcast", "OpCopyObject"):
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


def operands(line):
    """(coordinate, compare or None, {image operand: [values]}) of an OpImageSample*ExplicitLod line."""
    toks = line.split()             # %r = OpImageSample... %type %sampled %coord [%ref] Masks values...
    dref = "Dref" in toks[2]
    coord = toks[5]
    ref = toks[6] if dref else None
    rest = toks[7:] if dref else toks[6:]
    masks = rest[0].split("|")
    vals = rest[1:]
    sizes = {"Grad": 2, "Offset": 1, "MinLod": 1, "Lod": 1}
    out = {}
    for m in masks:
        out[m], vals = vals[:sizes[m]], vals[sizes[m]:]
    assert not vals, line
    return coord, ref, out


def regs(first, n):
    return f"v[{first}:{first + n - 1}]"


def offset_source(spv, value):
    """(VGPR, [bit offsets]) the texel-offset vector is extracted from (OpBitFieldSExtract of 6 bits each)."""
    d, c = defs(spv), consts(spv)
    op, args = d[value]
    parts = args[1:] if op == "OpCompositeConstruct" else [value]
    src, bits = None, []
    for p in parts:
        op, a = d[p]
        if op == "OpBitcast":
            op, a = d[a[1]]
        assert op == "OpBitFieldSExtract", (op, a)
        assert c[a[3]] == 6
        bits.append(c[a[2]])
        src = vgprs_behind(spv, a[1])[0] if d[a[1]][0] != "OpBitFieldSExtract" else None
    return src, bits


# (op, address VGPR count for 2D, expected layout relative to v30) written from the address order above.
LAYOUT_2D = {
    "image_sample_c_d":      (7, dict(ref=0, dx=[1, 2], dy=[3, 4], uv=[5, 6])),
    "image_sample_c_d_cl":   (8, dict(ref=0, dx=[1, 2], dy=[3, 4], uv=[5, 6], clamp=7)),
    "image_sample_d_o":      (7, dict(off=0, dx=[1, 2], dy=[3, 4], uv=[5, 6])),
    "image_sample_d_cl_o":   (8, dict(off=0, dx=[1, 2], dy=[3, 4], uv=[5, 6], clamp=7)),
    "image_sample_c_d_o":    (8, dict(off=0, ref=1, dx=[2, 3], dy=[4, 5], uv=[6, 7])),
    "image_sample_c_d_cl_o": (9, dict(off=0, ref=1, dx=[2, 3], dy=[4, 5], uv=[6, 7], clamp=8)),
}


class AddressOrder(unittest.TestCase):
    def check(self, spv, want, dims_uv=None):
        line = the_sample(spv)
        coord, ref, ops = operands(line)
        self.assertEqual("Dref" in line, "ref" in want, line)
        abs_ = lambda rel: [FIRST + r for r in rel]   # noqa: E731
        self.assertEqual(vgprs_behind(spv, coord), abs_(dims_uv or want["uv"]))
        if "ref" in want:
            self.assertEqual(vgprs_behind(spv, ref), [FIRST + want["ref"]])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][0]), abs_(want["dx"]))
        self.assertEqual(vgprs_behind(spv, ops["Grad"][1]), abs_(want["dy"]))
        if "clamp" in want:
            self.assertEqual(vgprs_behind(spv, ops["MinLod"][0]), [FIRST + want["clamp"]])
        else:
            self.assertNotIn("MinLod", ops)
        if "off" in want:
            src, bits = offset_source(spv, ops["Offset"][0])
            self.assertEqual(src, FIRST + want["off"])
            self.assertEqual(bits, [0, 8][:len(bits)])
        else:
            self.assertNotIn("Offset", ops)
        # SPIR-V image-operand order: Grad, Offset, MinLod.
        masks = line.split()[7 if "Dref" in line else 6].split("|")
        self.assertEqual(masks, [m for m in ("Grad", "Offset", "MinLod") if m in masks])
        return line

    def test_2d_forms_in_a_pixel_shader(self):
        for op, (n, want) in LAYOUT_2D.items():
            with self.subTest(op=op):
                dmask, dst = ("0x1", "v20") if "_c_" in op else ("0xf", "v[20:23]")
                spv = translate("ps_grad", f"{op} {dst}, {regs(FIRST, n)}, {DESC} dmask:{dmask} dim:SQ_RSRC_IMG_2D")
                line = self.check(spv, want)
                self.assertIn("ExplicitLod", line)

    def test_2d_forms_in_a_compute_shader(self):
        # Explicit gradients need no quad: the same sample outside the pixel stage.
        for op, (n, want) in LAYOUT_2D.items():
            with self.subTest(op=op):
                dmask, dst = ("0x1", "v20") if "_c_" in op else ("0xf", "v[20:23]")
                spv = translate("cs_grad", f"{op} {dst}, {regs(FIRST, n)}, {DESC} dmask:{dmask} dim:SQ_RSRC_IMG_2D")
                self.check(spv, want)

    def test_2d_array(self):
        # A layer coordinate after u, v; still two-component gradients (the layer has none).
        spv = translate("ps_grad_arr", f"image_sample_c_d_cl_o v20, {regs(FIRST, 10)}, {DESC} dmask:0x1 "
                                       "dim:SQ_RSRC_IMG_2D_ARRAY")
        self.check(spv, dict(off=0, ref=1, dx=[2, 3], dy=[4, 5], uv=[6, 7, 8], clamp=9))

    def test_1d(self):
        spv = translate("ps_grad_1d", f"image_sample_c_d_o v20, {regs(FIRST, 5)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_1D")
        line = the_sample(spv)
        coord, ref, ops = operands(line)
        self.assertEqual(vgprs_behind(spv, coord), [FIRST + 4])
        self.assertEqual(vgprs_behind(spv, ref), [FIRST + 1])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][0]), [FIRST + 2])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][1]), [FIRST + 3])
        src, bits = offset_source(spv, ops["Offset"][0])
        self.assertEqual((src, bits), (FIRST, [0]))

    def test_3d_offset_form(self):
        # Three-component gradients and offsets (x [5:0], y [13:8], z [21:16]).
        spv = translate("ps_grad_3d", f"image_sample_d_o v[20:23], {regs(FIRST, 10)}, {DESC} dmask:0xf "
                                      "dim:SQ_RSRC_IMG_3D")
        line = the_sample(spv)
        coord, _ref, ops = operands(line)
        self.assertEqual(vgprs_behind(spv, coord), [FIRST + 7, FIRST + 8, FIRST + 9])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][0]), [FIRST + 1, FIRST + 2, FIRST + 3])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][1]), [FIRST + 4, FIRST + 5, FIRST + 6])
        self.assertEqual(offset_source(spv, ops["Offset"][0]), (FIRST, [0, 8, 16]))

    def test_a16_keeps_compare_and_gradients_whole(self):
        # GFX10 a16 without g16: the offset, z-compare and gradients keep a whole VGPR each and only the coordinates
        # and the clamp after them are packed f16 halves (LLVM lowerImage; docs/design.md "Packed sample addresses").
        spv = translate("ps_grad_a16", f"image_sample_c_d_cl_o v20, {regs(FIRST, 8)}, {DESC} dmask:0x1 "
                                       "dim:SQ_RSRC_IMG_2D a16")
        line = the_sample(spv)
        coord, ref, ops = operands(line)
        self.assertEqual(vgprs_behind(spv, ref), [FIRST + 1])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][0]), [FIRST + 2, FIRST + 3])
        self.assertEqual(vgprs_behind(spv, ops["Grad"][1]), [FIRST + 4, FIRST + 5])
        self.assertEqual(vgprs_behind(spv, coord), [(FIRST + 6, 0), (FIRST + 6, 1)])
        self.assertEqual(vgprs_behind(spv, ops["MinLod"][0]), [(FIRST + 7, 0)])
        self.assertEqual(offset_source(spv, ops["Offset"][0])[0], FIRST)

    def test_d16_compare_result_is_an_f16_in_the_low_half(self):
        spv = translate("ps_grad_d16", f"image_sample_c_d v20, {regs(FIRST, 7)}, {DESC} dmask:0x1 "
                                       "dim:SQ_RSRC_IMG_2D d16")
        res = the_sample(spv).split()[0]
        self.assertIn("OpImageSampleDrefExplicitLod", spv)
        # The f32 compare result is narrowed to f16 bits before the store, not stored as its own bits.
        stored = re.search(r"OpStore %reg_v20 (%\w+)", spv).group(1)
        self.assertNotEqual(defs(spv)[stored], ("OpBitcast", ["%uint", res]))
        self.assertRegex(spv, rf"OpBitcast %uint {res}\n(?:.*\n){{0,3}}.*OpShiftRightLogical %uint")

    def test_colour_result_follows_dmask(self):
        spv = translate("ps_grad_dmask", f"image_sample_d_o v[20:21], {regs(FIRST, 7)}, {DESC} dmask:0x5 "
                                         "dim:SQ_RSRC_IMG_2D")
        res = the_sample(spv).split()[0]
        comps = [int(k) for k in re.findall(rf"OpCompositeExtract %float {res} (\d)", spv)]
        self.assertEqual(comps, [0, 2])


class Refusals(unittest.TestCase):
    def test_compare_on_a_3d_image(self):
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            translate("ps_grad_3dc", f"image_sample_c_d v20, {regs(FIRST, 10)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D")
        self.assertIn("3D", str(ctx.exception))

    def test_offset_fallback(self):
        saved = rdna2spirv.TEXEL_OFFSET_FALLBACK
        rdna2spirv.TEXEL_OFFSET_FALLBACK = True
        try:
            with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                translate("ps_grad_fb", f"image_sample_d_o v[20:23], {regs(FIRST, 7)}, {DESC} dmask:0xf "
                                        "dim:SQ_RSRC_IMG_2D")
            self.assertIn("texel-offset-fallback", str(ctx.exception))
            # without an offset the fallback does not matter
            translate("ps_grad_fb2", f"image_sample_c_d v20, {regs(FIRST, 7)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
        finally:
            rdna2spirv.TEXEL_OFFSET_FALLBACK = saved

    def test_still_refused_neighbours(self):
        # Coarse-derivative and 16-bit-gradient forms are not part of this change.
        for line in (f"image_sample_cd v[20:23], {regs(FIRST, 6)}, {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D",
                     f"image_sample_c_d_g16 v20, {regs(FIRST, 5)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                translate("ps_grad_no", line)

    def test_3d_status_with_gradients(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate("ps_grad_tfe", f"image_sample_d_o v[20:24], {regs(FIRST, 10)}, {DESC} dmask:0xf "
                                     "dim:SQ_RSRC_IMG_3D tfe")


class Validation(unittest.TestCase):
    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        cases = [(op, f"{op} {'v20' if '_c_' in op else 'v[20:23]'}, {regs(FIRST, n)}, {DESC} "
                      f"dmask:{'0x1' if '_c_' in op else '0xf'} dim:SQ_RSRC_IMG_2D") for op, (n, _) in LAYOUT_2D.items()]
        cases.append(("3D d_o", f"image_sample_d_o v[20:23], {regs(FIRST, 10)}, {DESC} dmask:0xf dim:SQ_RSRC_IMG_3D"))
        cases.append(("a16", f"image_sample_c_d_cl_o v20, {regs(FIRST, 8)}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16"))
        for name, line in cases:
            for stage in ("ps_val", "cs_val"):
                with self.subTest(case=name, stage=stage):
                    spv = translate(stage, line)
                    with tempfile.TemporaryDirectory() as tmp:
                        a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                        with open(a, "w", encoding="utf-8", newline="\n") as f:
                            f.write(spv)
                        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
                        # a texel offset that comes from a register needs VK_KHR_maintenance8 (design.md)
                        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", b],
                                       check=True)


if __name__ == "__main__":
    unittest.main()
