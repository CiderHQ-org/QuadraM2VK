import unittest

import regen_device_profiles as regen
import required_formats as rf

# A reduced Required Format Support section in the specification's own markup: the two
# "Feature bits" tables, the TRANSFER sentence, one table with every footnote form but the table
# choice, and three compressed tables that choose between each other.
HEADER = """\
4+>| ename:VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT              .5+^.^| {downarrow}
3+>| ename:VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT   .4+^.^| {downarrow}
2+>| ename:VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT              .3+^.^| {downarrow}
1+>| ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT              .2+^.^| {downarrow}
s| Format
"""
COMPRESSED_HEADER = """\
3+>| ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT .4+^.^| {downarrow}
2+>| ename:VK_FORMAT_FEATURE_BLIT_SRC_BIT                    .3+^.^| {downarrow}
1+>| ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT               .2+^.^| {downarrow}
s| Format
"""


def compressed(anchor, name, others):
    return f"""\
[[formats-mandatory-features-{anchor}]]
.Mandatory format support: {anchor}
[width="100%",cols="12,^1,^1,^1",options="unbreakable"]
|====
{COMPRESSED_HEADER}| ename:{name} | {{sym2}} | {{sym2}} | {{sym2}}
4+| The ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
ename:VK_FORMAT_FEATURE_BLIT_SRC_BIT and
ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT features must: be
supported in pname:optimalTilingFeatures for all the formats in at least
one of: this table, <<formats-mandatory-features-{others[0]}>>, or
<<formats-mandatory-features-{others[1]}>>.
|====
"""


SPEC = f"""\
== Earlier Section

ifndef::VK_VERSION_1_1,VK_KHR_maintenance1[]
Text outside the section may use any conditional.
endif::VK_VERSION_1_1,VK_KHR_maintenance1[]

[[features-required-format-support]]
== Required Format Support

The following tables show which feature bits must: be supported for each
format.
ifdef::VK_VERSION_1_1,VK_KHR_maintenance1[]
Formats that are required to support
ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT must: also support
ename:VK_FORMAT_FEATURE_TRANSFER_SRC_BIT and
ename:VK_FORMAT_FEATURE_TRANSFER_DST_BIT.
endif::VK_VERSION_1_1,VK_KHR_maintenance1[]

.Feature bits in pname:optimalTilingFeatures
[width="70%"]
|====
ifdef::VK_VERSION_1_1,VK_KHR_maintenance1[]
|ename:VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
|ename:VK_FORMAT_FEATURE_TRANSFER_DST_BIT
endif::VK_VERSION_1_1,VK_KHR_maintenance1[]
|ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
|ename:VK_FORMAT_FEATURE_BLIT_SRC_BIT
|ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
|ename:VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
|ename:VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
|====

.Feature bits in pname:bufferFeatures
[width="70%"]
|====
|ename:VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT
|====

[[formats-mandatory-features-mixed]]
.Mandatory format support: mixed
[width="100%",cols="12,^1,^1,^1,^1",options="unbreakable"]
|====
{HEADER}| ename:VK_FORMAT_R8_UNORM      | {{sym1}} | {{sym3}} |   | {{sym1}}
| ename:VK_FORMAT_R32_UINT              | {{sym1}}
|
ifdef::VK_EXT_toy[{{sym2}}]
                                        |   |
// A comment line inside a table.
| ename:VK_FORMAT_X8_D24_UNORM_PACK32   |   |   | {{sym2}} |
| ename:VK_FORMAT_D32_SFLOAT            |   |   | {{sym2}} |
| ename:VK_FORMAT_D24_UNORM_S8_UINT     |   |   | {{sym2}} |
| ename:VK_FORMAT_D32_SFLOAT_S8_UINT    |   |   | {{sym2}} |
ifdef::VK_EXT_other[]
| ename:VK_FORMAT_R64_UINT              |   | {{sym1}} |   |
endif::VK_EXT_other[]
5+| Format features marked with {{sym3}} must: be supported for
pname:optimalTilingFeatures if the sname:VkPhysicalDevice supports the
<<features-shaderStorageImageExtendedFormats,
pname:shaderStorageImageExtendedFormats>> feature.
5+| ename:VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT feature must: be
supported for at least one of ename:VK_FORMAT_X8_D24_UNORM_PACK32 and
ename:VK_FORMAT_D32_SFLOAT, and must: be supported for at least one of
ename:VK_FORMAT_D24_UNORM_S8_UINT and ename:VK_FORMAT_D32_SFLOAT_S8_UINT.
5+| pname:bufferFeatures must: not support any features for these formats
|====

{compressed("bcn", "VK_FORMAT_BC1_RGB_UNORM_BLOCK", ["etc", "astc"])}
{compressed("etc", "VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK", ["bcn", "astc"])}
{compressed("astc", "VK_FORMAT_ASTC_4x4_UNORM_BLOCK", ["bcn", "etc"])}
=== Next Section
"""

SAMPLED = ["VK_FORMAT_FEATURE_TRANSFER_SRC_BIT", "VK_FORMAT_FEATURE_TRANSFER_DST_BIT",
           "VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT"]
COMPRESSED = SAMPLED + ["VK_FORMAT_FEATURE_BLIT_SRC_BIT",
                        "VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT"]
DEPTH = ["VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT"]


def depth(name):
    return {name: {"optimalTilingFeatures": DEPTH}}


class RequiredFormatsTest(unittest.TestCase):
    def test_tables_become_typed_requirements(self):
        result = rf.required_formats(SPEC, rf.VERSION_1_2)
        self.assertEqual(result, {
            "tables": ["formats-mandatory-features-mixed", "formats-mandatory-features-bcn",
                       "formats-mandatory-features-etc", "formats-mandatory-features-astc"],
            "unconditional": {
                "VK_FORMAT_R32_UINT": {"optimalTilingFeatures": SAMPLED},
                "VK_FORMAT_R8_UNORM": {"bufferFeatures": ["VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT"],
                                       "optimalTilingFeatures": SAMPLED},
            },
            "conditional": [
                {"tables": ["formats-mandatory-features-mixed"], "any_of": [
                    depth("VK_FORMAT_X8_D24_UNORM_PACK32"), depth("VK_FORMAT_D32_SFLOAT")]},
                {"tables": ["formats-mandatory-features-mixed"], "any_of": [
                    depth("VK_FORMAT_D24_UNORM_S8_UINT"), depth("VK_FORMAT_D32_SFLOAT_S8_UINT")]},
                {"tables": ["formats-mandatory-features-mixed"],
                 "if_feature": "shaderStorageImageExtendedFormats",
                 "any_of": [{"VK_FORMAT_R8_UNORM": {
                     "optimalTilingFeatures": ["VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"]}}]},
                {"tables": ["formats-mandatory-features-astc", "formats-mandatory-features-bcn",
                            "formats-mandatory-features-etc"],
                 "any_of": [{name: {"optimalTilingFeatures": COMPRESSED}} for name in [
                     "VK_FORMAT_ASTC_4x4_UNORM_BLOCK", "VK_FORMAT_BC1_RGB_UNORM_BLOCK",
                     "VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK"]]},
            ],
            "not_represented": [{
                "table": "formats-mandatory-features-mixed",
                "text": "pname:bufferFeatures must: not support any features for these formats",
            }],
        })

    def test_extension_attributes_add_rows_and_marks(self):
        result = rf.required_formats(SPEC, rf.VERSION_1_2 + ["VK_EXT_other"])
        self.assertEqual(result["unconditional"]["VK_FORMAT_R64_UINT"],
                         {"optimalTilingFeatures": ["VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"]})
        # The single-line ifdef puts {sym2} in R32_UINT's storage column, which the table's
        # {sym2} footnote does not name: an error, not a guess.
        with self.assertRaisesRegex(rf.SpecError, "sym2 marks"):
            rf.required_formats(SPEC, rf.VERSION_1_2 + ["VK_EXT_toy"])

    def test_the_transfer_sentence_is_required(self):
        with self.assertRaisesRegex(rf.SpecError, "TRANSFER_DST sentence"):
            rf.required_formats(SPEC, ["VK_VERSION_1_0"])

    def test_unknown_footnotes_and_marks_are_errors(self):
        with self.assertRaisesRegex(rf.SpecError, "unrecognized footnote"):
            rf.required_formats(SPEC.replace("must: not support any", "must: not advertise"),
                                rf.VERSION_1_2)
        with self.assertRaisesRegex(rf.SpecError, "unknown mark"):
            rf.required_formats(SPEC.replace("| {sym1} | {sym3} |", "| {sym1} | yes |", 1),
                                rf.VERSION_1_2)
        with self.assertRaisesRegex(rf.SpecError, "sym4, which no cell uses"):
            rf.required_formats(SPEC.replace("marked with {sym3}", "marked with {sym4}"),
                                rf.VERSION_1_2)

    def test_a_table_choice_must_be_stated_by_each_table(self):
        broken = SPEC.replace("<<formats-mandatory-features-etc>>, or\n<<formats-mandatory-"
                              "features-astc>>", "<<formats-mandatory-features-etc>>, or\n"
                              "<<formats-mandatory-features-mixed>>", 1)
        with self.assertRaisesRegex(rf.SpecError, "not stated by each"):
            rf.required_formats(broken, rf.VERSION_1_2)

    def test_requirements_profiles_get_the_blocks_their_features_imply(self):
        formats = rf.required_formats(SPEC, rf.VERSION_1_2)
        for enabled, blocks in [(False, ["base", regen.REQUIRED_FORMATS_BLOCK]),
                                (True, ["base", regen.REQUIRED_FORMATS_BLOCK,
                                        regen.REQUIRED_FORMATS_BLOCK
                                        + "_if_shaderStorageImageExtendedFormats"])]:
            capabilities = {"base": {"features": {"VkPhysicalDeviceFeatures": {
                "shaderStorageImageExtendedFormats": enabled}}}}
            required = ["base"]
            provenance = {}
            regen.required_format_blocks(capabilities, required, provenance, formats)
            self.assertEqual(required, blocks)
            self.assertEqual(capabilities[regen.REQUIRED_FORMATS_BLOCK]["formats"][
                "VK_FORMAT_R32_UINT"], {"VkFormatProperties": {"optimalTilingFeatures": SAMPLED}})
            self.assertEqual(provenance["required_formats"]["conditional"],
                             formats["conditional"])
            self.assertEqual(provenance["required_formats"]["sha256"], rf.SPEC["sha256"])
        self.assertEqual(capabilities[blocks[-1]], {"formats": {"VK_FORMAT_R8_UNORM": {
            "VkFormatProperties": {
                "optimalTilingFeatures": ["VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT"]}}}})

    def test_one_alternative_of_a_choice_requires_no_feature(self):
        def block(enabled):
            return {"features": {"VkPhysicalDeviceFeatures": {
                "shaderStorageImageExtendedFormats": enabled}}}
        capabilities = {"a": block(True), "b": block(False), "c": block(True)}
        self.assertEqual(regen.required_features(capabilities, [["a", "b"]]), set())
        self.assertEqual(regen.required_features(capabilities, [["a", "c"]]),
                         {"shaderStorageImageExtendedFormats"})
        self.assertEqual(regen.required_features(capabilities, ["b", ["a", "b"], "c"]),
                         {"shaderStorageImageExtendedFormats"})


if __name__ == "__main__":
    unittest.main()
