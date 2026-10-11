//! The seeded generator: a seed's sequence, the same on every host. Each command is drawn at
//! random from the commands the consumer reaches (fills and copies of buffers, copies between
//! buffers and textures, one-thread compute dispatches, direct and indirect, kernels that read or
//! write texels, render passes that clear or load their target and draw rectangles, direct and
//! indirect), over the resources it reaches (buffers of their own, over host pages without a
//! copy, or placed in placement heaps where they overlap; textures of one or two slices, alone or
//! in a heap, and views of them in another format, of one slice, or swizzled), and every access
//! it makes is then ordered after each earlier conflicting one ([`super::order`]): where Metal's
//! order does not already hold, the generator adds what makes it hold, in the encoder (a memory
//! barrier of a concurrent compute encoder), between encoders of one command buffer (a fence),
//! or between command buffers (an event), or moves the command to an encoder of its own. So a
//! sequence never races, and its every byte is defined.

use super::order::{command_accesses, pass_accesses, Access, Order, Position};
use super::program::{
    At, BufferSpec, Channel, Command, CommandBufferSpec, Encoder, Format, HeapSpec, Kernel, Kind,
    Load, Place, Program, Role, Step, Surface, TextureSpec, ViewForm, ViewSpec, BUFFER_BYTES,
    HEAP_PLACES, HEAP_STEP, HEAP_WINDOW, MAX_INSTANCES, MAX_SLICES, RECORD_BYTES, RECT_VERTICES,
    ROW_BYTES, TARGET, TEXTURE_BYTES,
};

/// SplitMix64: a small generator whose stream depends on the seed alone.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x5EED_5EED_5EED_5EED)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n` (`n > 0`).
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// True with probability `percent` in 100.
    pub(crate) fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }

    fn bytes(&mut self, len: u64) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

/// The sequence of `seed`.
pub(crate) fn generate(seed: u64) -> Program {
    let mut rng = Rng::new(seed);
    let mut program = Program {
        heaps: Vec::new(),
        buffers: Vec::new(),
        textures: Vec::new(),
        views: Vec::new(),
        queues: 1 + rng.below(3) as usize,
        events: 0,
        fences: 0,
        command_buffers: Vec::new(),
    };
    for _ in 0..rng.below(3) {
        program.heaps.push(HeapSpec {
            private: rng.chance(40),
            tracked: rng.chance(50),
            initial: rng.bytes(HEAP_WINDOW),
        });
    }
    let data = 2 + rng.below(3);
    for _ in 0..data {
        let spec = buffer(&mut rng, Role::Data, &program.heaps);
        program.buffers.push(spec);
    }
    for role in [Role::Rects, Role::DrawArguments, Role::DispatchArguments] {
        let spec = buffer(&mut rng, role, &program.heaps);
        program.buffers.push(spec);
    }
    for _ in 0..1 + rng.below(2) {
        let slices = if rng.chance(30) { MAX_SLICES } else { 1 };
        program.textures.push(TextureSpec {
            tracked: rng.chance(60),
            slices,
            heap: rng.chance(30),
            initial: rng.bytes(u64::from(slices) * TEXTURE_BYTES),
        });
    }
    for _ in 0..rng.below(4) {
        let view = view(&mut rng, &program.textures);
        program.views.push(view);
    }
    let command_buffers = 2 + rng.below(5) as usize;
    let mut generator = Generator {
        program,
        rng,
        cb: 0,
        step: 0,
    };
    for cb in 0..command_buffers {
        generator.command_buffer(cb);
    }
    generator.program
}

/// A buffer of `role`: its storage, tracking, place and first bytes. Argument buffers are
/// tracked, since the order takes no fence to cover an argument read. Only data buffers are
/// placed in heaps, where they overlap one another, so every byte a role's buffer holds stays one
/// of its role's; a buffer over a host page is `Shared`.
fn buffer(rng: &mut Rng, role: Role, heaps: &[HeapSpec]) -> BufferSpec {
    if role == Role::Data && !heaps.is_empty() && rng.chance(45) {
        let index = rng.below(heaps.len() as u64) as usize;
        let heap = &heaps[index];
        return BufferSpec {
            role,
            private: heap.private,
            tracked: heap.tracked,
            place: Place::Heap {
                heap: index,
                offset: HEAP_STEP * rng.below(HEAP_PLACES),
            },
            initial: Vec::new(),
        };
    }
    let words = BUFFER_BYTES / 4;
    let initial: Vec<u8> = match role {
        Role::Data => rng.bytes(BUFFER_BYTES),
        Role::Rects => (0..words)
            .flat_map(|_| (rng.below(u64::from(TARGET) + 1) as u32).to_le_bytes())
            .collect(),
        Role::DrawArguments => (0..BUFFER_BYTES / RECORD_BYTES)
            .flat_map(|_| {
                let vertices = if rng.chance(80) { RECT_VERTICES } else { 0 };
                let instances = rng.below(u64::from(MAX_INSTANCES) + 1) as u32;
                [vertices, instances, 0, 0]
            })
            .flat_map(u32::to_le_bytes)
            .collect(),
        Role::DispatchArguments => (0..BUFFER_BYTES / RECORD_BYTES)
            .flat_map(|_| [u32::from(rng.chance(80)), 1, 1, 0])
            .flat_map(u32::to_le_bytes)
            .collect(),
    };
    let arguments = matches!(role, Role::DrawArguments | Role::DispatchArguments);
    let no_copy = rng.chance(20);
    BufferSpec {
        role,
        private: !no_copy && rng.chance(if role == Role::Data { 40 } else { 30 }),
        tracked: arguments || rng.chance(60),
        place: if no_copy { Place::NoCopy } else { Place::Own },
        initial,
    }
}

/// A view of one of `textures`: of one slice, in either format, and made each way the texture
/// allows; a swizzled one maps each channel from any source.
fn view(rng: &mut Rng, textures: &[TextureSpec]) -> ViewSpec {
    let texture = rng.below(textures.len() as u64) as usize;
    let slices = textures[texture].slices;
    let format = if rng.chance(50) {
        Format::Bgra8Unorm
    } else {
        Format::Rgba8Unorm
    };
    let channels = [
        Channel::Zero,
        Channel::One,
        Channel::Red,
        Channel::Green,
        Channel::Blue,
        Channel::Alpha,
    ];
    let form = match rng.below(3) {
        0 if slices == 1 => ViewForm::Whole,
        2 => ViewForm::Swizzled([
            rng.pick(&channels),
            rng.pick(&channels),
            rng.pick(&channels),
            rng.pick(&channels),
        ]),
        _ => ViewForm::Slices,
    };
    ViewSpec {
        texture,
        slice: if form == ViewForm::Whole {
            0
        } else {
            rng.below(u64::from(slices)) as u32
        },
        format,
        form,
    }
}

struct Generator {
    program: Program,
    rng: Rng,
    /// The current command buffer and step, where the current encoder is.
    cb: usize,
    step: usize,
}

impl Generator {
    fn buffers_of(&self, role: Role) -> Vec<usize> {
        (0..self.program.buffers.len())
            .filter(|&index| self.program.buffers[index].role == role)
            .collect()
    }

    fn one_of(&self, role: Role) -> usize {
        self.buffers_of(role)[0]
    }

    fn encoder(&mut self) -> &mut Encoder {
        match &mut self.program.command_buffers[self.cb].steps[self.step] {
            Step::Encoder(encoder) => encoder,
            other => panic!("the current step is no encoder: {other:?}"),
        }
    }

    fn command_buffer(&mut self, cb: usize) {
        let queue = self.rng.below(self.program.queues as u64) as usize;
        self.program.command_buffers.push(CommandBufferSpec {
            queue,
            steps: Vec::new(),
        });
        self.cb = cb;
        // Now and then a wait no access needs, for an earlier command buffer of another queue.
        if cb > 0 && self.rng.chance(25) {
            let earlier = self.rng.below(cb as u64) as usize;
            let event = self.new_event();
            self.program.command_buffers[earlier]
                .steps
                .push(Step::Signal(event));
            self.program.command_buffers[cb]
                .steps
                .push(Step::Wait(event));
        }
        for _ in 0..1 + self.rng.below(3) {
            let kind = match self.rng.below(100) {
                0..=29 => Kind::Blit,
                30..=49 => Kind::Compute { concurrent: false },
                50..=64 => Kind::Compute { concurrent: true },
                _ => {
                    let (target, slice) = self.surface(false);
                    Kind::Render {
                        target,
                        slice,
                        load: if self.rng.chance(50) {
                            Load::Clear([
                                self.rng.next() as u8,
                                self.rng.next() as u8,
                                self.rng.next() as u8,
                                self.rng.next() as u8,
                            ])
                        } else {
                            Load::Load
                        },
                    }
                }
            };
            self.open(kind);
            let commands = match kind {
                Kind::Render { .. } => 1 + self.rng.below(u64::from(MAX_INSTANCES)),
                _ => 1 + self.rng.below(4),
            };
            for _ in 0..commands {
                let command = self.command(kind);
                self.add(command);
            }
        }
    }

    /// Open an encoder of `kind` at the end of the current command buffer, with its pass's
    /// accesses ordered.
    fn open(&mut self, kind: Kind) {
        let steps = &mut self.program.command_buffers[self.cb].steps;
        steps.push(Step::Encoder(Encoder {
            kind,
            waits: Vec::new(),
            updates: Vec::new(),
            commands: Vec::new(),
        }));
        self.step = steps.len() - 1;
        for access in pass_accesses(&self.program, kind) {
            self.order(None, &access);
        }
    }

    /// Add `command` to the current encoder, ordering each of its accesses, again until none
    /// needed anything more: a move to an encoder of its own leaves the fences the old one
    /// waited for behind.
    fn add(&mut self, command: Command) {
        let accesses = command_accesses(&self.program, &command);
        let encoder = self.encoder();
        encoder.commands.push(command);
        let mut index = encoder.commands.len() - 1;
        loop {
            let mut changed = false;
            for access in &accesses {
                let (at, more) = self.order(Some(index), access);
                index = at.expect("a command's index");
                changed |= more;
            }
            if !changed {
                return;
            }
        }
    }

    /// Order `access` of the current encoder's command `command` (`None`: the pass itself) after
    /// every earlier conflicting access: the command's index once done (a barrier before it, or
    /// a move to an encoder of its own, changes it), and whether anything was added.
    fn order(&mut self, mut command: Option<usize>, access: &Access) -> (Option<usize>, bool) {
        let mut changed = false;
        loop {
            let at = Position {
                cb: self.cb,
                step: self.step,
                command,
            };
            let earlier = {
                let order = Order::new(&self.program);
                order.unordered_before(at, access)
            };
            let Some((earlier, first)) = earlier else {
                return (command, changed);
            };
            changed = true;
            if (earlier.cb, earlier.step) == (self.cb, self.step) {
                let Some(index) = command else {
                    panic!("a pass's access conflicts inside its own encoder");
                };
                let kind = self.encoder().kind;
                match kind {
                    Kind::Compute { concurrent: true } => {
                        let textures = first.texture() || access.texture();
                        let barrier = Command::Barrier { textures };
                        self.encoder().commands.insert(index, barrier);
                        command = Some(index + 1);
                    }
                    Kind::Render { .. } => panic!("a draw's access conflicts inside its pass"),
                    _ => {
                        // An encoder of its own for this command and the rest of the current one.
                        let moved = self.encoder().commands.split_off(index);
                        let steps = &mut self.program.command_buffers[self.cb].steps;
                        steps.insert(
                            self.step + 1,
                            Step::Encoder(Encoder {
                                kind,
                                waits: Vec::new(),
                                updates: Vec::new(),
                                commands: moved,
                            }),
                        );
                        self.step += 1;
                        command = Some(0);
                    }
                }
            } else if earlier.cb == self.cb {
                let fence = self.program.fences;
                self.program.fences += 1;
                match &mut self.program.command_buffers[earlier.cb].steps[earlier.step] {
                    Step::Encoder(encoder) => encoder.updates.push(fence),
                    other => panic!("an access outside an encoder: {other:?}"),
                }
                self.encoder().waits.push(fence);
            } else {
                let event = self.new_event();
                self.program.command_buffers[earlier.cb]
                    .steps
                    .insert(earlier.step + 1, Step::Signal(event));
                self.program.command_buffers[self.cb]
                    .steps
                    .insert(self.step, Step::Wait(event));
                self.step += 1;
            }
        }
    }

    fn new_event(&mut self) -> usize {
        self.program.events += 1;
        self.program.events - 1
    }

    /// A word offset whose `words` words fit a buffer.
    fn word(&mut self, words: u64) -> u64 {
        4 * self.rng.below(BUFFER_BYTES / 4 - words + 1)
    }

    /// A record offset whose `records` records fit a buffer.
    fn record(&mut self, records: u64) -> u64 {
        RECORD_BYTES * self.rng.below(BUFFER_BYTES / RECORD_BYTES - records + 1)
    }

    fn command(&mut self, kind: Kind) -> Command {
        match kind {
            Kind::Blit => self.blit(),
            Kind::Compute { concurrent } => {
                if concurrent && self.rng.chance(15) && !self.encoder_is_empty() {
                    Command::Barrier {
                        textures: self.rng.chance(50),
                    }
                } else if self.rng.chance(25) && self.has_flat() {
                    self.texel()
                } else {
                    self.dispatch()
                }
            }
            Kind::Render { .. } => self.draw(),
        }
    }

    fn encoder_is_empty(&mut self) -> bool {
        self.encoder().commands.is_empty()
    }

    fn blit(&mut self) -> Command {
        let data = self.buffers_of(Role::Data);
        match self.rng.below(4) {
            0 => {
                let buffer = self.rng.below(self.program.buffers.len() as u64) as usize;
                let start = self.word(1);
                let end = start + 4 * (1 + self.rng.below((BUFFER_BYTES - start) / 4));
                let value = if self.program.buffers[buffer].role == Role::Data {
                    self.rng.next() as u8
                } else {
                    0
                };
                Command::Fill {
                    buffer,
                    start,
                    end,
                    value,
                }
            }
            1 => {
                // A copy keeps a role's words in their places: within one role, a record's
                // fields at the same offsets in their records.
                let role = self.rng.pick(&[
                    Role::Data,
                    Role::Data,
                    Role::Rects,
                    Role::DrawArguments,
                    Role::DispatchArguments,
                ]);
                let of = self.buffers_of(role);
                let from = self.rng.pick(&of);
                // Metal's API validation raises on a copy between two buffers over bytes in
                // common, even of bytes apart (it sends `width` to a buffer), where Metal alone
                // copies them. The records are made under validation, so a copy's buffers are one,
                // or apart in memory.
                let apart: Vec<usize> = of
                    .iter()
                    .copied()
                    .filter(|&to| !self.program.share_memory(from, to))
                    .collect();
                let to = self.rng.pick(&apart);
                let size = 4 * (1 + self.rng.below(16));
                loop {
                    let a = self.word(size / 4);
                    let b = if role == Role::Data || role == Role::Rects {
                        self.word(size / 4)
                    } else {
                        let shift = a % RECORD_BYTES;
                        let room = (BUFFER_BYTES - size - shift) / RECORD_BYTES;
                        RECORD_BYTES * self.rng.below(room + 1) + shift
                    };
                    let (one, a_at, _) = self.program.memory(At {
                        buffer: from,
                        offset: a,
                    });
                    let (other, b_at, _) = self.program.memory(At {
                        buffer: to,
                        offset: b,
                    });
                    if one != other || a_at + size <= b_at || b_at + size <= a_at {
                        return Command::Copy {
                            from: At {
                                buffer: from,
                                offset: a,
                            },
                            to: At {
                                buffer: to,
                                offset: b,
                            },
                            size,
                        };
                    }
                }
            }
            2 | 3 => {
                let (surface, slice) = self.surface(false);
                let first = self.rng.below(u64::from(TARGET)) as u32;
                let last = first + 1 + self.rng.below(u64::from(TARGET - first)) as u32;
                let rows = (first, last.min(first + (BUFFER_BYTES / ROW_BYTES) as u32));
                let bytes = u64::from(rows.1 - rows.0) * ROW_BYTES;
                let at = At {
                    buffer: self.rng.pick(&data),
                    offset: 16 * self.rng.below((BUFFER_BYTES - bytes) / 16 + 1),
                };
                if self.rng.chance(50) {
                    Command::ToTexture {
                        from: at,
                        to: surface,
                        slice,
                        rows,
                    }
                } else {
                    Command::FromTexture {
                        from: surface,
                        slice,
                        rows,
                        to: at,
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    fn dispatch(&mut self) -> Command {
        let data = self.buffers_of(Role::Data);
        let any = self.rng.below(self.program.buffers.len() as u64) as usize;
        let (kernel, a, b) = if self.rng.chance(40) {
            let a = At {
                buffer: self.rng.pick(&data),
                offset: self.word(1),
            };
            // The kernel's buffers do not alias (its arguments are `air-buffer-no-alias`), so
            // the salt is another word than the one mixed.
            let b = loop {
                let b = At {
                    buffer: any,
                    offset: self.word(1),
                };
                if self.program.memory(a).0 != self.program.memory(b).0
                    || self.program.memory(a).1 != self.program.memory(b).1
                {
                    break b;
                }
            };
            (Kernel::Mix, a, b)
        } else {
            // A copy into a role's buffer takes a word of that role from the same place of
            // its record; into a data buffer, any word.
            let to_role = self.rng.pick(&[
                Role::Data,
                Role::Data,
                Role::Data,
                Role::Rects,
                Role::DrawArguments,
                Role::DispatchArguments,
            ]);
            let to = At {
                buffer: self.rng.pick(&self.buffers_of(to_role)),
                offset: self.word(1),
            };
            let from = match to_role {
                Role::Data => At {
                    buffer: any,
                    offset: self.word(1),
                },
                Role::Rects => At {
                    buffer: to.buffer,
                    offset: self.word(1),
                },
                _ => At {
                    buffer: to.buffer,
                    offset: self.record(1) + to.offset % RECORD_BYTES,
                },
            };
            (Kernel::Copy, from, to)
        };
        let indirect = if self.rng.chance(30) {
            let arguments = self.one_of(Role::DispatchArguments);
            let written = match kernel {
                Kernel::Copy => b,
                Kernel::Mix => a,
            };
            let mut offset = self.record(1);
            // The dispatch never writes the arguments it reads (no heap holds them).
            while written.buffer == arguments
                && written.offset >= offset
                && written.offset < offset + RECORD_BYTES
            {
                offset = self.record(1);
            }
            Some(At {
                buffer: arguments,
                offset,
            })
        } else {
            None
        };
        Command::Dispatch {
            kernel,
            a,
            b,
            indirect,
        }
    }

    /// Whether a 2D surface exists: a texture of one slice, or a view.
    fn has_flat(&self) -> bool {
        !self.program.views.is_empty()
            || self
                .program
                .textures
                .iter()
                .any(|texture| texture.slices == 1)
    }

    /// A surface and slice of it: any slice of a texture, or a view; with `kernel`, a 2D one
    /// (a kernel's texture is a `texture2d`), else one that is not swizzled (a render pass's
    /// target or a blit's texture).
    fn surface(&mut self, kernel: bool) -> (Surface, u32) {
        loop {
            let views = self.program.views.len() as u64;
            let textures = self.program.textures.len() as u64;
            let pick = self.rng.below(textures + views);
            let surface = if pick < textures {
                Surface::Texture(pick as usize)
            } else {
                Surface::View((pick - textures) as usize)
            };
            let fits = match surface {
                Surface::Texture(texture) => !kernel || self.program.textures[texture].slices == 1,
                Surface::View(view) => kernel || self.program.views[view].swizzle().is_none(),
            };
            if !fits {
                continue;
            }
            let slice = match surface {
                Surface::Texture(texture) => self
                    .rng
                    .below(u64::from(self.program.textures[texture].slices))
                    as u32,
                Surface::View(_) => 0,
            };
            return (surface, slice);
        }
    }

    /// A texel kernel's dispatch over rows of a 2D surface: a read through any (a swizzled view
    /// too), or a write through one that is not swizzled.
    fn texel(&mut self) -> Command {
        let rows = 1 + self.rng.below(u64::from(TARGET)) as u32;
        let bytes = u64::from(rows) * ROW_BYTES;
        let data = self.buffers_of(Role::Data);
        let at = At {
            buffer: self.rng.pick(&data),
            offset: 4 * self.rng.below((BUFFER_BYTES - bytes) / 4 + 1),
        };
        loop {
            let (surface, _) = self.surface(true);
            let swizzled = matches!(surface, Surface::View(view) if self.program.views[view].swizzle().is_some());
            if self.rng.chance(50) {
                return Command::TexelRead {
                    from: surface,
                    rows,
                    to: at,
                };
            }
            if !swizzled {
                return Command::TexelWrite {
                    from: at,
                    to: surface,
                    rows,
                };
            }
        }
    }

    fn draw(&mut self) -> Command {
        let instances = 1 + self.rng.below(u64::from(MAX_INSTANCES)) as u32;
        let rects = At {
            buffer: self.one_of(Role::Rects),
            offset: self.record(u64::from(MAX_INSTANCES)),
        };
        let data = self.buffers_of(Role::Data);
        let colors = At {
            buffer: self.rng.pick(&data),
            offset: self.word(u64::from(MAX_INSTANCES)),
        };
        let indirect = self.rng.chance(30).then(|| At {
            buffer: self.one_of(Role::DrawArguments),
            offset: self.record(1),
        });
        Command::Draw {
            rects,
            colors,
            instances,
            indirect,
        }
    }
}
