//! The CPU model of a sequence: the bytes of every memory once it ran, computed by running its
//! commands one after another in encoded order. A sequence never races ([`super::order`]), so
//! every order Metal may run it in that respects its synchronization ends with these bytes, and
//! this order is one of them. Metal's records check the model; the model gives a runtime
//! difference its expected bytes, and the minimizer its question.
//!
//! A heap's buffers are its bytes at their offsets, a view's texels its texture's slice; a texel
//! is stored and read through a format ([`Format`]), and a view's reads take its swizzle. Blits
//! copy a texture's bytes as they are, whatever format they name it in.

use super::super::record::Value;
use super::program::{
    At, Command, Format, Kernel, Kind, Load, Place, Program, Resource, Step, Surface, CHUNK_BYTES,
    RECORD_BYTES, RECT_VERTICES, ROW_BYTES, TARGET, TEXEL_BYTES, TEXTURE_BYTES,
};

/// Every memory's bytes: each buffer's own (none for a heap's buffer), each heap's window, and
/// each texture's slices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct State {
    pub(crate) buffers: Vec<Vec<u8>>,
    pub(crate) heaps: Vec<Vec<u8>>,
    pub(crate) textures: Vec<Vec<u8>>,
}

impl State {
    /// The bytes of `memory`.
    fn memory(&mut self, memory: Resource) -> &mut Vec<u8> {
        match memory {
            Resource::Buffer(index) => &mut self.buffers[index],
            Resource::Heap(index) => &mut self.heaps[index],
            Resource::Texture(index) => &mut self.textures[index],
        }
    }

    /// `len` bytes of the buffer at `at` of `program`.
    fn bytes(&mut self, program: &Program, at: At, len: u64) -> &mut [u8] {
        let (memory, start, _) = program.memory(at);
        &mut self.memory(memory)[start as usize..(start + len) as usize]
    }

    fn word(&mut self, program: &Program, at: At, index: u64) -> u32 {
        let at = At {
            buffer: at.buffer,
            offset: at.offset + 4 * index,
        };
        let bytes = self.bytes(program, at, 4);
        u32::from_le_bytes((&*bytes).try_into().expect("four bytes"))
    }

    fn set_word(&mut self, program: &Program, at: At, value: u32) {
        self.bytes(program, at, 4)
            .copy_from_slice(&value.to_le_bytes());
    }

    /// The bytes of texel `(x, y)` of slice `slice` of `surface`.
    fn texel(
        &mut self,
        program: &Program,
        surface: Surface,
        slice: u32,
        x: u32,
        y: u32,
    ) -> &mut [u8] {
        let seen = program.seen(surface, slice);
        let start = u64::from(seen.slice) * TEXTURE_BYTES
            + u64::from(y) * ROW_BYTES
            + u64::from(x) * TEXEL_BYTES;
        &mut self.textures[seen.texture][start as usize..(start + TEXEL_BYTES) as usize]
    }

    /// Rows `rows` of slice `slice` of `surface`, as they are.
    fn rows(
        &mut self,
        program: &Program,
        surface: Surface,
        slice: u32,
        rows: (u32, u32),
    ) -> &mut [u8] {
        let seen = program.seen(surface, slice);
        let base = u64::from(seen.slice) * TEXTURE_BYTES;
        let start = (base + u64::from(rows.0) * ROW_BYTES) as usize;
        let end = (base + u64::from(rows.1) * ROW_BYTES) as usize;
        &mut self.textures[seen.texture][start..end]
    }
}

/// The format `surface` stores texels in.
fn format_of(program: &Program, surface: Surface) -> Format {
    program.seen(surface, 0).format
}

/// `seq_mix`'s word of `word` and `salt`.
pub(crate) fn mix(word: u32, salt: u32) -> u32 {
    (word ^ salt)
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(0x7F4A_7C15)
}

/// The bytes the sequence ends with.
pub(crate) fn run(program: &Program) -> State {
    let mut state = State {
        buffers: program
            .buffers
            .iter()
            .map(|buffer| buffer.initial.clone())
            .collect(),
        heaps: program
            .heaps
            .iter()
            .map(|heap| heap.initial.clone())
            .collect(),
        textures: program
            .textures
            .iter()
            .map(|texture| texture.initial.clone())
            .collect(),
    };
    for spec in &program.command_buffers {
        for step in &spec.steps {
            let Step::Encoder(encoder) = step else {
                continue;
            };
            let target = match encoder.kind {
                Kind::Render {
                    target,
                    slice,
                    load,
                } => {
                    if let Load::Clear(color) = load {
                        let stored = format_of(program, target).store(color);
                        for texel in state
                            .rows(program, target, slice, (0, TARGET))
                            .chunks_mut(TEXEL_BYTES as usize)
                        {
                            texel.copy_from_slice(&stored);
                        }
                    }
                    Some((target, slice))
                }
                _ => None,
            };
            for command in &encoder.commands {
                execute(program, &mut state, command, target);
            }
        }
    }
    state
}

fn execute(
    program: &Program,
    state: &mut State,
    command: &Command,
    target: Option<(Surface, u32)>,
) {
    match *command {
        Command::Fill {
            buffer,
            start,
            end,
            value,
        } => {
            let at = At {
                buffer,
                offset: start,
            };
            state.bytes(program, at, end - start).fill(value);
        }
        Command::Copy { from, to, size } => {
            let bytes = state.bytes(program, from, size).to_vec();
            state.bytes(program, to, size).copy_from_slice(&bytes);
        }
        Command::ToTexture {
            from,
            to,
            slice,
            rows,
        } => {
            let len = u64::from(rows.1 - rows.0) * ROW_BYTES;
            let bytes = state.bytes(program, from, len).to_vec();
            state.rows(program, to, slice, rows).copy_from_slice(&bytes);
        }
        Command::FromTexture {
            from,
            slice,
            rows,
            to,
        } => {
            let bytes = state.rows(program, from, slice, rows).to_vec();
            state
                .bytes(program, to, bytes.len() as u64)
                .copy_from_slice(&bytes);
        }
        Command::TexelRead { from, rows, to } => {
            let seen = program.seen(from, 0);
            for y in 0..rows {
                for x in 0..TARGET {
                    let stored: [u8; 4] = (&*state.texel(program, from, 0, x, y))
                        .try_into()
                        .expect("a texel");
                    let rgba = seen.format.load(stored);
                    let read = match seen.swizzle {
                        Some(channels) => channels.map(|channel| channel.of(rgba)),
                        None => rgba,
                    };
                    let at = At {
                        buffer: to.buffer,
                        offset: to.offset + u64::from(y) * ROW_BYTES + u64::from(x) * TEXEL_BYTES,
                    };
                    state.bytes(program, at, 4).copy_from_slice(&read);
                }
            }
        }
        Command::TexelWrite { from, to, rows } => {
            let format = format_of(program, to);
            for y in 0..rows {
                for x in 0..TARGET {
                    let at = At {
                        buffer: from.buffer,
                        offset: from.offset + u64::from(y) * ROW_BYTES + u64::from(x) * TEXEL_BYTES,
                    };
                    let rgba: [u8; 4] = (&*state.bytes(program, at, 4)).try_into().expect("a word");
                    state
                        .texel(program, to, 0, x, y)
                        .copy_from_slice(&format.store(rgba));
                }
            }
        }
        Command::Dispatch {
            kernel,
            a,
            b,
            indirect,
        } => {
            if let Some(at) = indirect {
                let groups = [
                    state.word(program, at, 0),
                    state.word(program, at, 1),
                    state.word(program, at, 2),
                ];
                assert!(
                    groups.iter().all(|&count| count <= 1),
                    "dispatch arguments {groups:?} at {at}"
                );
                if groups != [1, 1, 1] {
                    return;
                }
            }
            match kernel {
                Kernel::Copy => {
                    let word = state.word(program, a, 0);
                    state.set_word(program, b, word);
                }
                Kernel::Mix => {
                    let (word, salt) = (state.word(program, a, 0), state.word(program, b, 0));
                    state.set_word(program, a, mix(word, salt));
                }
            }
        }
        Command::Barrier { .. } => {}
        Command::Draw {
            rects,
            colors,
            instances,
            indirect,
        } => {
            let (target, slice) = target.expect("a draw in a render pass");
            let format = format_of(program, target);
            let instances = match indirect {
                Some(at) => {
                    let arguments = [
                        state.word(program, at, 0),
                        state.word(program, at, 1),
                        state.word(program, at, 2),
                        state.word(program, at, 3),
                    ];
                    assert!(
                        (arguments[0] == 0 || arguments[0] == RECT_VERTICES)
                            && arguments[1] <= super::program::MAX_INSTANCES
                            && arguments[2] == 0
                            && arguments[3] == 0,
                        "draw arguments {arguments:?} at {at}"
                    );
                    if arguments[0] == 0 {
                        0
                    } else {
                        arguments[1]
                    }
                }
                None => instances,
            };
            for instance in 0..u64::from(instances) {
                let rect = At {
                    buffer: rects.buffer,
                    offset: rects.offset + RECORD_BYTES * instance,
                };
                let corners = [
                    state.word(program, rect, 0),
                    state.word(program, rect, 1),
                    state.word(program, rect, 2),
                    state.word(program, rect, 3),
                ];
                assert!(
                    corners.iter().all(|&c| c <= TARGET),
                    "a rectangle {corners:?} at {rect}"
                );
                let color = format.store(state.word(program, colors, instance).to_le_bytes());
                let (x0, x1) = (corners[0].min(corners[2]), corners[0].max(corners[2]));
                let (y0, y1) = (corners[1].min(corners[3]), corners[1].max(corners[3]));
                for y in y0..y1 {
                    for x in x0..x1 {
                        state
                            .texel(program, target, slice, x, y)
                            .copy_from_slice(&color);
                    }
                }
            }
        }
    }
}

/// The names and bytes the runner records of the bytes a sequence ends with, its readback: each
/// buffer with bytes of its own, each heap's window, in chunks of [`CHUNK_BYTES`], then each
/// texture's slices.
pub(crate) fn readback_records(program: &Program, state: &State) -> Vec<(String, Vec<u8>)> {
    let mut records = Vec::new();
    let mut chunks = |name: String, bytes: &[u8]| {
        for (chunk, part) in bytes.chunks(CHUNK_BYTES as usize).enumerate() {
            records.push((format!("{name} bytes {chunk}"), part.to_vec()));
        }
    };
    for (index, spec) in program.buffers.iter().enumerate() {
        if !matches!(spec.place, Place::Heap { .. }) {
            chunks(format!("buffer{index}"), &state.buffers[index]);
        }
    }
    for (index, bytes) in state.heaps.iter().enumerate() {
        chunks(format!("heap{index}"), bytes);
    }
    for (index, bytes) in state.textures.iter().enumerate() {
        for (slice, part) in bytes.chunks(TEXTURE_BYTES as usize).enumerate() {
            records.push((format!("texture{index} slice{slice} bytes"), part.to_vec()));
        }
    }
    records
}

/// The records of the bytes `program` ends with, named as the runner names its readback for
/// `case` ([`readback_records`]).
pub(crate) fn records(case: &str, program: &Program) -> Vec<(String, Value)> {
    readback_records(program, &run(program))
        .into_iter()
        .map(|(name, bytes)| (format!("{case}/{name}"), Value::bytes(&bytes)))
        .collect()
}

/// Each resource whose bytes in `records` differ from `expected`'s, or were not read back: its
/// name and the byte ranges that differ, with both sides' bytes there.
pub(crate) fn byte_differences(
    records: &[(String, Value)],
    expected: &[(String, Value)],
) -> Vec<String> {
    let mut differences = Vec::new();
    for (name, want) in expected {
        let found = records
            .iter()
            .find(|(ran, _)| ran == name)
            .map(|(_, value)| value);
        let (
            Value::Bytes {
                sha256: a,
                hex: want_hex,
                ..
            },
            Some(Value::Bytes {
                sha256: b,
                hex: got_hex,
                ..
            }),
        ) = (want, found)
        else {
            differences.push(format!("{name}: expected {want:?}, here {found:?}"));
            continue;
        };
        if a == b {
            continue;
        }
        let (Some(want), Some(got)) = (
            want_hex.as_deref().map(unhex),
            got_hex.as_deref().map(unhex),
        ) else {
            differences.push(format!("{name}: SHA-256 {a} expected, here {b}"));
            continue;
        };
        let mut ranges = Vec::new();
        let mut index = 0;
        let len = want.len().max(got.len());
        while index < len {
            if want.get(index) == got.get(index) {
                index += 1;
                continue;
            }
            let start = index;
            while index < len && want.get(index) != got.get(index) {
                index += 1;
            }
            let side = |bytes: &[u8]| hex(&bytes[start.min(bytes.len())..index.min(bytes.len())]);
            ranges.push(format!(
                "[{start}..{index}] expected {} here {}",
                side(&want),
                side(&got)
            ));
        }
        differences.push(format!("{name}: {}", ranges.join(", ")));
    }
    differences
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|at| u8::from_str_radix(&text[2 * at..2 * at + 2], 16).expect("hex"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
