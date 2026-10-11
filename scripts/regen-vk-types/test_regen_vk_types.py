import unittest

from regen_vk_types import (
    Registry, aspect_numeric_formats, const_name, format_blocks, render_format_blocks,
    render_single_bits, value_prefix,
)


def registry(extra_enums="", extensions="", features=""):
    return f"""<registry>
      <types>
        <type requires="VkThingFlagBits" category="bitmask">typedef <type>VkFlags</type> <name>VkThingFlags</name>;</type>
      </types>
      <enums name="VkColor" type="enum">
        <enum value="0" name="VK_COLOR_RED"/>
        <enum value="1" name="VK_COLOR_GREEN"/>
        <enum value="2" name="VK_COLOR_SC_ONLY" api="vulkansc"/>
        <unused start="3"/>
        {extra_enums}
      </enums>
      <enums name="VkThingFlagBits" type="bitmask">
        <enum bitpos="0" name="VK_THING_ONE_BIT"/>
        <enum value="0x3" name="VK_THING_BOTH"/>
      </enums>
      {features}
      <extensions>{extensions}</extensions>
    </registry>""".encode()


class RegistryTests(unittest.TestCase):
    def test_extension_values_use_extension_number_offset_and_direction(self):
        reg = Registry(registry(extensions="""
          <extension name="VK_EXT_a" number="3" supported="vulkan">
            <require>
              <enum offset="0" extends="VkColor" name="VK_COLOR_BLUE_EXT"/>
              <enum offset="1" dir="-" extends="VkColor" name="VK_COLOR_ERROR_EXT"/>
              <enum offset="2" extnumber="9" extends="VkColor" name="VK_COLOR_OTHER_EXT"/>
              <enum bitpos="4" extends="VkThingFlagBits" name="VK_THING_FOUR_BIT_EXT"/>
            </require>
          </extension>"""))
        values = reg.values["VkColor"]
        self.assertEqual(values["VK_COLOR_BLUE_EXT"], 1000002000)
        self.assertEqual(values["VK_COLOR_ERROR_EXT"], -1000002001)
        self.assertEqual(values["VK_COLOR_OTHER_EXT"], 1000008002)
        self.assertEqual(reg.values["VkThingFlagBits"]["VK_THING_FOUR_BIT_EXT"], 16)

    def test_sc_only_beta_disabled_and_provisional_values_are_excluded(self):
        reg = Registry(registry(extensions="""
          <extension name="VK_X_sc" number="4" supported="vulkansc">
            <require><enum offset="0" extends="VkColor" name="VK_COLOR_SC_EXT"/></require>
          </extension>
          <extension name="VK_X_off" number="5" supported="disabled">
            <require><enum offset="0" extends="VkColor" name="VK_COLOR_OFF_EXT"/></require>
          </extension>
          <extension name="VK_X_beta" number="6" supported="vulkan" platform="provisional">
            <require><enum offset="0" extends="VkColor" name="VK_COLOR_BETA_EXT"/></require>
          </extension>
          <extension name="VK_X_mixed" number="7" supported="vulkan">
            <require>
              <enum offset="0" extends="VkColor" name="VK_COLOR_PROTECTED" protect="BETA"/>
              <enum offset="1" extends="VkColor" api="vulkansc" name="VK_COLOR_API_SC"/>
            </require>
            <require api="vulkansc">
              <enum offset="2" extends="VkColor" name="VK_COLOR_BLOCK_SC"/>
            </require>
          </extension>"""))
        self.assertEqual(sorted(reg.values["VkColor"]), ["VK_COLOR_GREEN", "VK_COLOR_RED"])

    def test_aliases_resolve_to_canonical_constants(self):
        reg = Registry(registry(extra_enums='<enum name="VK_COLOR_RED_KHR" alias="VK_COLOR_RED"/>'))
        rows, aliases = reg.entries("VkColor", is_bits=False)
        self.assertEqual([r[0] for r in rows], ["RED", "GREEN"])
        self.assertEqual(aliases, [("VK_COLOR_RED_KHR", "RED")])

    def test_conflicting_redefinition_and_unknown_forms_fail(self):
        with self.assertRaises(ValueError):
            Registry(registry(features="""
              <feature api="vulkan" name="VK_VERSION_1_1" number="1.1">
                <require><enum value="7" extends="VkColor" name="VK_COLOR_RED"/></require>
              </feature>"""))
        with self.assertRaises(ValueError):
            Registry(registry(extra_enums='<enum name="VK_COLOR_ODD" future="1"/>'))

    def test_single_bits_exclude_combinations(self):
        reg = Registry(registry(extensions="""
          <extension name="VK_EXT_a" number="3" supported="vulkan">
            <require>
              <enum bitpos="2" extends="VkThingFlagBits" name="VK_THING_TWO_BIT_EXT"/>
              <enum extends="VkThingFlagBits" name="VK_THING_ONE_BIT_KHR" alias="VK_THING_ONE_BIT"/>
              <enum extends="VkThingFlagBits" name="VK_THING_BOTH_KHR" alias="VK_THING_BOTH"/>
            </require>
          </extension>"""))
        out = "\n".join(render_single_bits(reg, "VkThingFlagBits", "Thing", "VkThingFlags"))
        self.assertIn('ONE = 0x00000001, "VK_THING_ONE_BIT";', out)
        self.assertIn('TWO_EXT = 0x00000004, "VK_THING_TWO_BIT_EXT";', out)
        self.assertIn('"VK_THING_ONE_BIT_KHR" => ONE;', out)
        self.assertNotIn("BOTH", out)
        with self.assertRaises(ValueError):
            render_single_bits(reg, "VkThingFlagBits", "Thing", "VkOtherFlags")

    def test_constant_names(self):
        self.assertEqual(value_prefix("VkPipelineStageFlagBits2"), "VK_PIPELINE_STAGE_2_")
        self.assertEqual(value_prefix("VkDriverId"), "VK_DRIVER_ID_")
        self.assertEqual(const_name("VK_SHADER_STAGE_TASK_BIT_EXT", "VK_SHADER_STAGE_", True),
                         "TASK_EXT")
        self.assertEqual(const_name("VK_IMAGE_VIEW_TYPE_2D", "VK_IMAGE_VIEW_TYPE_", False),
                         "TYPE_2D")
        self.assertEqual(const_name("VK_FORMAT_ASTC_4x4_UNORM_BLOCK", "VK_FORMAT_", False),
                         "ASTC_4X4_UNORM_BLOCK")
        with self.assertRaises(ValueError):
            const_name("VK_OTHER_X", "VK_FORMAT_", False)


def format_registry(formats):
    return f"""<registry>
      <types/>
      <enums name="VkFormat" type="enum">
        <enum value="0" name="VK_FORMAT_UNDEFINED"/>
        <enum value="1" name="VK_FORMAT_R8_UNORM"/>
        <enum value="2" name="VK_FORMAT_A2B10G10R10_UNORM_PACK32"/>
        <enum value="3" name="VK_FORMAT_BC1_RGB_UNORM_BLOCK"/>
        <enum value="4" name="VK_FORMAT_D16_UNORM_S8_UINT"/>
        <enum name="VK_FORMAT_R8_UNORM_KHR" alias="VK_FORMAT_R8_UNORM"/>
      </enums>
      <feature api="vulkan" name="VK_VERSION_1_4" number="1.4">
        <require>
          <enum extends="VkFormat" extnumber="9" offset="0" name="VK_FORMAT_R8G8B8A8_SRGB"/>
        </require>
      </feature>
      <formats>{formats}</formats>
      <extensions>
        <extension name="VK_KHR_srgb" number="9" supported="vulkan">
          <require>
            <enum extends="VkFormat" name="VK_FORMAT_R8G8B8A8_SRGB_KHR" alias="VK_FORMAT_R8G8B8A8_SRGB"/>
          </require>
        </extension>
        <extension name="VK_IMG_bc" number="10" supported="vulkan">
          <require>
            <enum extends="VkFormat" offset="0" name="VK_FORMAT_BC_IMG"/>
          </require>
        </extension>
      </extensions>
    </registry>""".encode()


BLOCKS = """
  <format name="VK_FORMAT_BC1_RGB_UNORM_BLOCK" class="BC1_RGB" blockSize="8" texelsPerBlock="16"
          blockExtent="4,4,1" compressed="BC">
    <component name="R" bits="compressed" numericFormat="UNORM"/>
  </format>
  <format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="1" texelsPerBlock="1">
    <component name="R" bits="8" numericFormat="UNORM"/>
    <spirvimageformat name="R8"/>
  </format>
  <format name="VK_FORMAT_A2B10G10R10_UNORM_PACK32" class="32-bit" blockSize="4"
          texelsPerBlock="1" packed="32">
    <component name="A" bits="2" numericFormat="UNORM"/>
    <component name="B" bits="10" numericFormat="UNORM"/>
  </format>
  <format name="VK_FORMAT_D16_UNORM_S8_UINT" class="D16S8" blockSize="3" texelsPerBlock="1">
    <component name="D" bits="16" numericFormat="UNORM"/>
    <component name="S" bits="8" numericFormat="UINT"/>
  </format>
  <format name="VK_FORMAT_R8G8B8A8_SRGB" class="32-bit" blockSize="4" texelsPerBlock="1">
    <component name="R" bits="8" numericFormat="SRGB"/>
    <component name="G" bits="8" numericFormat="SRGB"/>
    <component name="B" bits="8" numericFormat="SRGB"/>
    <component name="A" bits="8" numericFormat="UNORM"/>
  </format>
  <format name="VK_FORMAT_BC_IMG" class="BC" blockSize="8" texelsPerBlock="16"
          compressed="BC">
    <component name="R" bits="compressed" numericFormat="UNORM"/>
  </format>
  <format name="VK_FORMAT_EXCLUDED_BETA" class="8-bit" blockSize="1" texelsPerBlock="1">
    <component name="R" bits="8" numericFormat="UNORM"/>
  </format>"""


class FormatBlockTests(unittest.TestCase):
    def test_blocks_follow_enum_order_with_packed_and_shared_widths(self):
        rows = format_blocks(Registry(format_registry(BLOCKS)))
        core = (1, 0)
        self.assertEqual(rows, [
            ("R8_UNORM", 1, 1, "8-bit", None, 8, "Unorm", None, None, core, []),
            ("A2B10G10R10_UNORM_PACK32", 4, 1, "32-bit", 32, None, "Unorm", None, None, core,
             []),
            ("BC1_RGB_UNORM_BLOCK", 8, 16, "BC1_RGB", None, None, "Unorm", None, None, core, []),
            ("D16_UNORM_S8_UINT", 3, 1, "D16S8", None, None, None, "Unorm", "Uint", core, []),
            ("R8G8B8A8_SRGB", 4, 1, "32-bit", None, 8, "Srgb", None, None, (1, 4),
             ["VK_KHR_srgb"]),
            ("BC_IMG", 8, 16, "BC", None, None, "Unorm", None, None, None, ["VK_IMG_bc"]),
        ])
        self.assertEqual(render_format_blocks(Registry(format_registry(BLOCKS))), [
            "vk_format_blocks! {",
            '    R8_UNORM = 1, 1, class "8-bit", bits 8, color Unorm, core 1 0;',
            '    A2B10G10R10_UNORM_PACK32 = 4, 1, class "32-bit", packed 32, color Unorm, core 1 0;',
            '    BC1_RGB_UNORM_BLOCK = 8, 16, class "BC1_RGB", color Unorm, core 1 0;',
            '    D16_UNORM_S8_UINT = 3, 1, class "D16S8", depth Unorm, stencil Uint, core 1 0;',
            '    R8G8B8A8_SRGB = 4, 1, class "32-bit", bits 8, color Srgb, core 1 4,'
            ' ext "VK_KHR_srgb";',
            '    BC_IMG = 8, 16, class "BC", color Unorm, ext "VK_IMG_bc";',
            "}",
            "",
        ])

    def test_aspect_numeric_formats(self):
        self.assertEqual(aspect_numeric_formats("x", [("R", "UINT"), ("G", "UINT")]),
                         ("Uint", None, None))
        self.assertEqual(aspect_numeric_formats("x", [("D", "SFLOAT"), ("S", "UINT")]),
                         (None, "Sfloat", "Uint"))
        self.assertEqual(aspect_numeric_formats("x", [("S", "UINT")]), (None, None, "Uint"))
        # sRGB color with a linear alpha is the SRGB numeric format.
        self.assertEqual(
            aspect_numeric_formats("x", [("B", "SRGB"), ("G", "SRGB"), ("R", "SRGB"), ("A", "UNORM")]),
            ("Srgb", None, None))
        for bad in [
            [("R", "UNORM"), ("G", "SNORM")],
            [("R", "UNORM"), ("A", "SRGB")],
            [("R", "SRGB"), ("G", "UNORM")],
            [("R", "UNORM"), ("D", "UNORM")],
            [("X", "UNORM")],
            [("R", "FUTURE")],
        ]:
            with self.assertRaises(ValueError, msg=bad):
                aspect_numeric_formats("x", bad)

    def test_unknown_forms_aliases_and_duplicates_fail(self):
        for bad in [
            '<format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="1" texelsPerBlock="1" future="1">'
            '<component name="R" bits="8" numericFormat="UNORM"/></format>',
            '<format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="1" texelsPerBlock="1">'
            '<swizzle/><component name="R" bits="8" numericFormat="UNORM"/></format>',
            '<format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="0" texelsPerBlock="1">'
            '<component name="R" bits="8" numericFormat="UNORM"/></format>',
            '<format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="1" texelsPerBlock="1"/>',
            '<format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="1" texelsPerBlock="1">'
            '<component name="R" bits="8"/></format>',
            '<format name="VK_FORMAT_R8_UNORM_KHR" class="8-bit" blockSize="1" texelsPerBlock="1">'
            '<component name="R" bits="8" numericFormat="UNORM"/></format>',
            BLOCKS + '<format name="VK_FORMAT_R8_UNORM" class="8-bit" blockSize="1" texelsPerBlock="1">'
            '<component name="R" bits="8" numericFormat="UNORM"/></format>',
            '<format name="VK_FORMAT_R8_UNORM" blockSize="1" texelsPerBlock="1">'
            '<component name="R" bits="8" numericFormat="UNORM"/></format>',
        ]:
            with self.assertRaises(ValueError, msg=bad):
                format_blocks(Registry(format_registry(bad)))


if __name__ == "__main__":
    unittest.main()
