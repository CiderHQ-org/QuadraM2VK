#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Host-only tests; no client shaders, GPU, or Darling prefix required."""
import copy
import struct
import unittest
import shader_cache as cache


def op(code, *words):
    return [(len(words) + 1) << 16 | code, *words]


def spirv(*instructions):
    words = [0x07230203, 0x00010500, 0, 100, 0]
    words.extend(word for instruction in instructions for word in instruction)
    return struct.pack(f'<{len(words)}I', *words)


def descriptor(binding=0, kind='Buffer', index=0):
    return {'kind': kind, 'metal_index': index,
            'descriptor': {'set': 0, 'binding': binding, 'count': 1}}


class ShaderCacheTests(unittest.TestCase):
    def test_descriptor_buffers_use_ssbo_contract(self):
        reflection = {'bindings': [descriptor()]}
        self.assertEqual(cache.binding_packet(reflection, 0), [(4, 0, 0, 0)])
        module = spirv(op(30, 1), op(32, 2, 12, 1), op(59, 2, 3, 12),
                       op(71, 3, 34, 0), op(71, 3, 33, 0))
        cache.audit_descriptors(module, reflection)
        wrong = copy.deepcopy(reflection)
        wrong['bindings'][0]['kind'] = 'Sampler'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            cache.audit_descriptors(module, wrong)

    def test_sampler_and_texture_audit(self):
        module = spirv(op(26, 1), op(32, 2, 0, 1), op(59, 2, 3, 0),
                       op(71, 3, 34, 0), op(71, 3, 33, 160),
                       op(25, 4, 10, 1, 0, 0, 0, 1, 0), op(32, 5, 0, 4),
                       op(59, 5, 6, 0), op(71, 6, 34, 0), op(71, 6, 33, 32))
        cache.audit_descriptors(module, {'bindings': [descriptor(160, 'Sampler'),
                                                     descriptor(32, 'Texture')]})

    def test_unknown_abi_is_rejected(self):
        for stage, update in [(2, {}), (0, {'function_constants': [1]}),
                              (0, {'runtime_sampler_specializations': [1]}),
                              (0, {'argument_buffer_fields': [1]})]:
            with self.subTest(stage=stage, update=update), self.assertRaises(ValueError):
                cache.binding_packet({'bindings': [], **update}, stage)
        for count in (0, 2):
            binding = descriptor()
            binding['descriptor']['count'] = count
            with self.assertRaises(ValueError):
                cache.binding_packet({'bindings': [binding]}, 0)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            cache.binding_packet({'bindings': [descriptor(), descriptor()]}, 0)

    def test_entry_point_set_and_interface_remap(self):
        module = spirv(op(15, 4, 1, 0x6e69616d, 0, 9), op(59, 2, 9, 1),
                       op(71, 9, 30, 0), op(71, 8, 34, 0))
        translated = cache.adapt_spirv(module, 1, 'main0',
                                     {'varyings': [{'location': 0, 'user_semantic': 'user(locn7)'}]})
        words = struct.unpack(f'<{len(translated) // 4}I', translated)
        self.assertIn(tuple(op(71, 9, 30, 7)), [words[i:i + 4] for i in range(len(words) - 3)])
        self.assertIn(tuple(op(71, 8, 34, 1)), [words[i:i + 4] for i in range(len(words) - 3)])
        self.assertIn(b'main0\0', translated)

    def test_push_constants_and_named_interfaces_rejected(self):
        with self.assertRaisesRegex(ValueError, 'push-constant'):
            cache.adapt_spirv(spirv(op(59, 1, 2, 9)), 0, 'main0', {'varyings': []})
        with self.assertRaisesRegex(ValueError, 'shared'):
            cache.adapt_spirv(spirv(), 0, 'main0',
                             {'varyings': [{'location': 0, 'user_semantic': 'CUSTOM'}]})

    def test_malformed_modules_rejected(self):
        for module in (b'', b'12345', spirv([0]), spirv(op(15, 0, 1, 0x41414141))):
            with self.subTest(module=module), self.assertRaises(ValueError):
                cache.adapt_spirv(module, 0, 'main0', {'varyings': []})
        for library in (b'', b'MTLB' + bytes(84)):
            with self.assertRaises(ValueError):
                cache.functions(library)

    def test_metallib_table_length_excludes_function_count(self):
        tag = lambda name, value: name + struct.pack('<H', len(value)) + value
        tags = (tag(b'NAME', b'main0\0') + tag(b'TYPE', b'\0') +
                tag(b'MDSZ', struct.pack('<Q', 4)) + tag(b'OFFT', struct.pack('<3Q', 0, 0, 0)) +
                b'ENDT')
        group = struct.pack('<I', len(tags) + 4) + tags
        header = bytearray(88)
        header[:4] = b'MTLB'
        bitcode_offset = 88 + 4 + len(group)
        struct.pack_into('<9Q', header, 16, bitcode_offset + 4, 88, len(group),
                         0, 0, 0, 0, bitcode_offset, 4)
        library = bytes(header) + struct.pack('<I', 1) + group + b'AIR!'
        self.assertEqual(cache.functions(library), [('main0', 0, b'AIR!')])


if __name__ == '__main__':
    unittest.main()
