# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Build validated shader-cache packets for the experimental Indium adapter.

Reads locally installed shader libraries; never changes the client or invokes
an unlicensed extraction script. A cache miss or unsupported ABI is an error.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import struct
import subprocess

DEFAULT_WORK = Path(__file__).resolve().parents[2] / 'work' / 'metal-backend-repro'


def functions(library):
    if len(library) < 88 or library[:4] != b'MTLB':
        raise ValueError('invalid metallib header')
    size, offset, length, _, _, _, _, bc_offset, bc_length = struct.unpack_from('<9Q', library, 16)
    # The function table's length excludes its initial four-byte count.
    table_end = offset + 4 + length
    if size != len(library) or table_end > size or bc_offset + bc_length > size:
        raise ValueError('metallib range outside file')
    count = struct.unpack_from('<I', library, offset)[0]
    if count > 1024:
        raise ValueError('too many metallib functions')
    pos = offset + 4
    result = []
    for _ in range(count):
        if pos + 4 > table_end:
            raise ValueError('metallib function outside table')
        group_size = struct.unpack_from('<I', library, pos)[0]
        end = pos + group_size
        if group_size < 8 or end > table_end:
            raise ValueError('metallib group outside table')
        pos += 4
        tags = {}
        while pos + 4 <= end:
            tag = library[pos:pos + 4]
            pos += 4
            if tag == b'ENDT':
                break
            if pos + 2 > end:
                raise ValueError('truncated metallib tag')
            tag_length = struct.unpack_from('<H', library, pos)[0]
            pos += 2
            if pos + tag_length > end:
                raise ValueError('metallib tag outside group')
            tags[tag] = library[pos:pos + tag_length]
            pos += tag_length
        pos = end
        if any(tag not in tags for tag in (b'NAME', b'TYPE', b'MDSZ', b'OFFT')):
            raise ValueError('missing metallib function metadata')
        name = tags[b'NAME'].rstrip(b'\0').decode('utf-8')
        if not name or len(tags[b'TYPE']) != 1 or tags[b'TYPE'][0] > 2:
            raise ValueError('invalid metallib function stage/name')
        if len(tags[b'MDSZ']) != 8 or len(tags[b'OFFT']) != 24:
            raise ValueError('invalid metallib bitcode metadata')
        stage = tags[b'TYPE'][0]
        bitcode_size = struct.unpack('<Q', tags[b'MDSZ'])[0]
        bitcode_offset = struct.unpack('<3Q', tags[b'OFFT'])[2] + bc_offset
        if bitcode_offset + bitcode_size > size:
            raise ValueError('function bitcode outside metallib')
        result.append((name, stage, library[bitcode_offset:bitcode_offset + bitcode_size]))
    return result


def libraries(pack):
    pos = 0
    while True:
        start = pack.find(b'MTLB', pos)
        if start < 0:
            return
        if start + 88 > len(pack):
            return
        size = struct.unpack_from('<Q', pack, start + 16)[0]
        if size < 88 or size > len(pack) - start:
            pos = start + 4
            continue
        library = pack[start:start + size]
        funcs = functions(library)
        yield start, library, funcs
        pos = start + size


def adapt_spirv(data, stage, name, reflection):
    if len(data) < 20 or len(data) % 4 or stage not in (0, 1):
        raise ValueError('invalid SPIR-V size or unsupported stage')
    words = list(struct.unpack(f'<{len(data) // 4}I', data))
    if words[0] != 0x07230203:
        raise ValueError('invalid SPIR-V header')
    result = words[:5]
    pos = 5
    entry_count = 0
    set_index = 1 if stage == 1 else 0
    varying_locations = {}
    for varying in reflection.get('varyings', []):
        semantic = re.fullmatch(r'user\(locn(\d+)\)', varying.get('user_semantic') or '')
        if not semantic:
            raise ValueError('named varyings need a shared vertex/fragment interface allocator')
        varying_locations[varying['location']] = int(semantic[1])
    variable_storage = {}
    scan = 5
    while scan < len(words):
        count, opcode = words[scan] >> 16, words[scan] & 0xFFFF
        if not count or scan + count > len(words):
            raise ValueError('malformed SPIR-V instruction')
        if opcode == 59 and count >= 4:  # OpVariable
            variable_storage[words[scan + 2]] = words[scan + 3]
            if words[scan + 3] == 9:  # PushConstant
                raise ValueError('shader needs an unimplemented push-constant ABI')
        scan += count
    name_bytes = name.encode('utf-8') + b'\0'
    name_bytes += b'\0' * (-len(name_bytes) % 4)
    name_words = list(struct.unpack(f'<{len(name_bytes) // 4}I', name_bytes))
    while pos < len(words):
        count, opcode = words[pos] >> 16, words[pos] & 0xFFFF
        if not count or pos + count > len(words):
            raise ValueError('malformed SPIR-V instruction')
        instruction = words[pos:pos + count]
        if opcode == 15:  # OpEntryPoint; translator entry is always "main".
            if count < 4 or instruction[1] != (0 if stage == 0 else 4):
                raise ValueError('SPIR-V entry point disagrees with shader stage')
            entry_count += 1
            old_name_end = 3
            while old_name_end < count and b'\0' not in struct.pack('<I', instruction[old_name_end]):
                old_name_end += 1
            if old_name_end == count:
                raise ValueError('unterminated SPIR-V entry-point name')
            instruction = instruction[:3] + name_words + instruction[old_name_end + 1:]
            instruction[0] = (len(instruction) << 16) | opcode
        elif opcode == 71 and count == 4 and instruction[2] == 34:  # DescriptorSet
            if instruction[3] != 0:
                raise ValueError('translator uses an unexpected descriptor set')
            instruction[3] = set_index
        elif opcode == 71 and count == 4 and instruction[2] == 30:  # Location
            expected_storage = 3 if stage == 0 else 1  # Vertex Output, fragment Input
            if variable_storage.get(instruction[1]) == expected_storage:
                if instruction[3] not in varying_locations:
                    raise ValueError('SPIR-V varying is missing from reflection')
                instruction[3] = varying_locations[instruction[3]]
        result.extend(instruction)
        pos += count
    if entry_count != 1:
        raise ValueError('adapter supports exactly one entry per library')
    return struct.pack(f'<{len(result)}I', *result)


def binding_packet(reflection, stage):
    result = []
    unsupported = []
    occupied = set()
    for binding in reflection['bindings']:
        kind = binding['kind']
        descriptor = binding.get('descriptor')
        if not descriptor:
            unsupported.append(f'{kind}: resource has no descriptor contract')
            continue
        if descriptor['set'] != 0 or descriptor['count'] != 1:
            unsupported.append(f'{kind}: descriptor arrays/other sets need backend support')
            continue
        index = binding['metal_index']
        internal = descriptor['binding']
        if (not isinstance(index, int) or index < 0 or index > 127 or
                not isinstance(internal, int) or internal < 0 or internal > 4095 or
                internal in occupied):
            raise ValueError('invalid or duplicate descriptor binding')
        occupied.add(internal)
        if kind == 'Buffer':
            binding_type, access = 4, 0  # New DescriptorBuffer; sampled is zero.
        elif kind in ('Texture', 'StorageImage'):
            binding_type, access = 1, 0 if kind == 'Texture' else 3
        elif kind == 'Sampler':
            binding_type, access = 2, 0
        else:
            unsupported.append(f'{kind}: explicit adapter implementation required')
            continue
        result.append((binding_type, index, internal, access))
    if stage == 2:
        unsupported.append('compute dispatch/push-constant ABI is not implemented')
    for field in ('runtime_sampler_specializations', 'runtime_storage_image_specializations',
                  'function_constants', 'argument_buffer_fields', 'tessellation',
                  'imageblock_layouts', 'implicit_imageblock_attachments', 'fragment_imageblock'):
        if reflection.get(field):
            unsupported.append(f'{field}: adapter implementation required')
    if unsupported:
        raise ValueError('; '.join(unsupported))
    return result


def audit_descriptors(data, reflection):
    """Reject reflection that disagrees with the actual SPIR-V resource ABI."""
    if len(data) < 20 or len(data) % 4:
        raise ValueError('invalid SPIR-V size')
    words = struct.unpack(f'<{len(data) // 4}I', data)
    types, variables, decorations = {}, {}, {}
    pos = 5
    while pos < len(words):
        count, opcode = words[pos] >> 16, words[pos] & 0xFFFF
        if not count or pos + count > len(words):
            raise ValueError('malformed SPIR-V instruction')
        instruction = words[pos:pos + count]
        if 19 <= opcode <= 33 and count >= 2:
            types[instruction[1]] = instruction
        elif opcode == 59 and count >= 4:
            variables[instruction[2]] = (instruction[1], instruction[3])
        elif opcode == 71 and count == 4 and instruction[2] in (33, 34):
            key = (instruction[1], instruction[2])
            if key in decorations:
                raise ValueError('duplicate descriptor decoration')
            decorations[key] = instruction[3]
        pos += count
    actual = {}
    for variable, (pointer_id, storage) in variables.items():
        if storage not in (0, 2, 12):  # UniformConstant, Uniform, StorageBuffer
            continue
        pointer = types.get(pointer_id, ())
        if len(pointer) != 4 or pointer[0] & 0xFFFF != 32 or pointer[2] != storage:
            raise ValueError('invalid descriptor pointer type')
        value = types.get(pointer[3], ())
        opcode = value[0] & 0xFFFF if value else 0
        if opcode == 30 and storage == 12:
            kind = 'Buffer'
        elif opcode == 26 and storage == 0:
            kind = 'Sampler'
        elif opcode == 25 and storage == 0 and len(value) >= 9 and value[7] in (1, 2):
            kind = 'Texture' if value[7] == 1 else 'StorageImage'
        else:
            raise ValueError('unsupported descriptor type/array ABI')
        if (variable, 33) not in decorations or (variable, 34) not in decorations:
            raise ValueError('descriptor lacks set/binding decorations')
        key = (decorations[variable, 34], decorations[variable, 33])
        if key in actual:
            raise ValueError('duplicate SPIR-V descriptor binding')
        actual[key] = kind
    expected = {(binding['descriptor']['set'], binding['descriptor']['binding']): binding['kind']
                for binding in reflection['bindings']}
    if actual != expected:
        raise ValueError('SPIR-V descriptors disagree with shader reflection')


def convert(library, funcs, output, translator):
    if len(funcs) != 1:
        raise ValueError('adapter supports exactly one function per metallib')
    name, stage, bitcode = funcs[0]
    digest = hashlib.sha256(library).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{digest}.metallib').write_bytes(library)
    air = output / f'{digest}.air'
    air.write_bytes(bitcode)
    spv_path = output / f'{digest}.spv'
    meta_path = output / f'{digest}.json'
    subprocess.run([translator, str(air), str(spv_path), '--stage',
                    ('vertex', 'fragment', 'kernel')[stage], '--raster-samples', '1',
                    '--emit-meta', str(meta_path)], check=True)
    reflection = json.loads(meta_path.read_text())
    if reflection['entry_point'] != name:
        raise ValueError('AIR and metallib disagree about entry-point identity')
    bindings = binding_packet(reflection, stage)
    data = spv_path.read_bytes()
    audit_descriptors(data, reflection)
    data = adapt_spirv(data, stage, name, reflection)
    spv_path.write_bytes(data)
    subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', str(spv_path)], check=True)
    name_bytes = name.encode('utf-8')
    packet = (b'MVK1' + struct.pack('<5I', 1, stage + 1, len(name_bytes), len(bindings), len(data))
              + name_bytes + b''.join(struct.pack('<4I', *binding) for binding in bindings) + data)
    (output / f'{digest}.mvk').write_bytes(packet)
    print(json.dumps({'sha256': digest, 'name': name, 'stage': stage,
                      'bindings': bindings, 'validated_bytes': len(data)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('--out', type=Path, default=DEFAULT_WORK / 'shader-cache')
    parser.add_argument('--translator', default='metal2vulkan')
    parser.add_argument('--index', type=int, action='append', default=[])
    options = parser.parse_args()
    entries = list(libraries(options.pack.read_bytes()))
    if not options.index:
        counts = collections.Counter(func[1] for _, _, funcs in entries for func in funcs)
        print('Libraries:', len(entries), 'function stages:', dict(counts))
        for index, (offset, library, funcs) in enumerate(entries):
            if index < 12 or (funcs and funcs[0][1] == 0 and index < 100):
                print(index, offset, len(library), [(name, stage) for name, stage, _ in funcs])
    for index in options.index:
        _, library, funcs = entries[index]
        convert(library, funcs, options.out, options.translator)


if __name__ == '__main__':
    main()
