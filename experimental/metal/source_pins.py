#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the source contents used by the native Metal/Vulkan prototype."""
from pathlib import Path
import argparse
import hashlib
import json


def tree_digest(root, selections=None):
    """SHA-256 of relative path + NUL + per-file SHA-256 + newline, sorted."""
    root = Path(root)
    selected = [root / name for name in selections] if selections else [root]
    files = []
    for entry in selected:
        if entry.is_file() or entry.is_symlink():
            files.append(entry)
        elif entry.is_dir():
            files.extend(path for path in entry.rglob('*') if path.is_file() or path.is_symlink())
        else:
            raise ValueError(f'Missing source: {entry}')
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda value: value.relative_to(root).as_posix()):
        content = path.readlink().as_posix().encode() if path.is_symlink() else path.read_bytes()
        kind = b'L' if path.is_symlink() else b'F'
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + kind +
                      hashlib.sha256(content).hexdigest().encode() + b'\n')
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--darling', type=Path, required=True)
    parser.add_argument('--vulkan', type=Path, required=True, help='Vulkan-Headers include directory')
    parser.add_argument('--translator', type=Path, help='optional metal2vulkan source root')
    options = parser.parse_args()
    pins = json.loads(Path(__file__).with_name('source_pins.json').read_text())
    roots = {'metal': options.darling / 'src/external/metal',
             'libcxx': options.darling / 'src/external/libcxx/include',
             'vulkan': options.vulkan, 'translator': options.translator}
    for component, root in roots.items():
        if root is None:
            continue
        actual = tree_digest(root, pins[component].get('selections'))
        if actual != pins[component]['sha256']:
            raise SystemExit(f'{component} source differs from tested content: {actual}')
        print(f'{component}: source pin verified')


if __name__ == '__main__':
    main()
