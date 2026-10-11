import unittest
from registry import extract, header, render_by_name


class RegistryTests(unittest.TestCase):
    def registry(self, rule):
        return ("<registry><spirvcapabilities>" + rule +
                "</spirvcapabilities><spirvextensions/></registry>").encode()

    def test_alternative_feature_routes_keep_their_prerequisites(self):
        result = extract(self.registry('''<spirvcapability name="Atomic">
          <enable struct="Core" feature="buffer" requires="VK_VERSION_1_2,EXT_buffer"/>
          <enable struct="Shared" feature="shared" requires="EXT_shared"/>
        </spirvcapability>'''))
        routes = result["spirvcapabilities"]["Atomic"]
        self.assertEqual(len(routes), 2)
        self.assertEqual(routes[0]["requires"], "VK_VERSION_1_2,EXT_buffer")
        self.assertEqual(routes[1]["feature"], "shared")

    def test_alias_bits_remain_distinct_and_compound_routes_fail(self):
        rules = extract(self.registry('''<spirvcapability name="Address">
          <enable struct="Core" feature="address" requires="VK_VERSION_1_2"/>
          <enable struct="Ext" feature="address" alias="addressEXT" requires="VK_EXT_address"/>
        </spirvcapability>'''))
        rust = render_by_name(rules)
        self.assertIn('"Address" => &[Exposure::Core(1, 2)],', rust)
        self.assertIn('"AddressEXT" => &[Exposure::Extension("ExtAddress")],', rust)
        rules["spirvcapabilities"]["Address"][0]["requires"] = "VK_EXT_one+VK_EXT_two"
        with self.assertRaises(ValueError):
            render_by_name(rules)

    def test_by_name_table_uses_the_rust_spelling_and_feature_unions(self):
        rules = extract(self.registry('''<spirvcapability name="Address">
          <enable struct="Core" feature="address" requires="VK_VERSION_1_2"/>
          <enable struct="Ext" feature="address" alias="addressEXT" requires="VK_EXT_address"/>
          <enable property="P" member="subgroupSupportedOperations" value="VK_SUBGROUP_FEATURE_VOTE_BIT" requires="VK_VERSION_1_1"/>
          <enable extension="VK_KHR_other"/>
        </spirvcapability>'''))
        rust = render_by_name(rules)
        self.assertIn('Enablement::Feature("Address"),', rust)
        self.assertIn('Enablement::Feature("AddressEXT"),', rust)
        self.assertIn('"AddressEXT" => &[Exposure::Extension("ExtAddress")],', rust)
        self.assertIn('Enablement::Property("SubgroupSupportedOperationsSubgroupFeatureVoteBit"),', rust)
        self.assertIn('Enablement::Exposed(Exposure::Extension("KhrOther")),', rust)

    def test_unknown_forms_and_duplicate_identities_fail(self):
        for rule in [
            '<spirvcapability name="A"><enable future="yes"/></spirvcapability>',
            '<spirvcapability name="A"/>',
            '<spirvcapability name="A"><enable version=""/></spirvcapability>',
            '<spirvcapability name="A"><enable version="V"/></spirvcapability>' * 2,
        ]:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                extract(self.registry(rule))


    def test_the_header_keeps_the_registry_s_copyright_and_mit_notice(self):
        def registry(licence):
            return (f"<registry><comment>\nCopyright 2015-2026 The Khronos Group Inc.\n\n"
                    f"SPDX-License-Identifier: {licence}\n</comment></registry>").encode()

        text = header(registry("Apache-2.0 OR MIT"))
        self.assertIn("// Copyright 2015-2026 The Khronos Group Inc.\n", text)
        self.assertIn("// Permission is hereby granted, free of charge,", text)
        self.assertIn("// The above copyright notice and this permission notice shall be included", text)
        self.assertTrue(all(len(line) <= 100 for line in text.splitlines()))
        for licence in ["Apache-2.0", "Apache-2.0 OR MIT OR BSD-3-Clause"]:
            with self.subTest(licence=licence), self.assertRaises(ValueError):
                header(registry(licence))

if __name__ == "__main__":
    unittest.main()
