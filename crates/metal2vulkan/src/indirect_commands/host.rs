//! The host's side of the record format: initializing a command buffer, and decoding and
//! checking what a producer wrote, before a replay reads it. The runtime's indirect command
//! buffers, a consumer's own runtime, validation's oracle and case model, and the crate's tests
//! use it; the binder's device replay decodes records on the device instead.

use super::descriptor::CommandDescriptor;
use super::{
    layout, require_length, Action, Bindings, BufferBinding, BufferStage, CommandRecord,
    Inheritance, RecordError, StateSource, BINDING_COUNT, BUFFER_STAGES, BUFFER_WORDS,
    COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES, HEADER_WORDS, THREADGROUP_COUNT, THREADGROUP_WORD,
};
use std::fmt;
use std::num::NonZeroU64;

/// Why a command buffer's storage was refused: its length, its descriptor, a header other than the
/// descriptor's, a range past the capacity, the sticky fault a producer's refused write set, or a
/// malformed record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandBufferError {
    Length,
    Descriptor(String),
    HeaderPolicyMismatch,
    Range { start: u32, end: u32, capacity: u32 },
    EncodingFault(u32),
    Record { index: u32, error: RecordError },
}

impl fmt::Display for CommandBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length => {
                f.write_str("command buffer length does not match its command capacity")
            }
            Self::Range {
                start,
                end,
                capacity,
            } => write!(
                f,
                "command range {start}..{end} is outside capacity {capacity}"
            ),
            Self::Descriptor(error) => write!(f, "command descriptor: {error}"),
            Self::HeaderPolicyMismatch => {
                f.write_str("command buffer header does not match its descriptor")
            }
            Self::EncodingFault(flags) => {
                write!(f, "command buffer encoding failed (flags {flags:#x})")
            }
            Self::Record { index, error } => write!(f, "command {index}: {error}"),
        }
    }
}

impl std::error::Error for CommandBufferError {}

/// Borrowed records after the producer has finished and its writes are visible to the host.
/// The host must keep the initialized header intact while the producer uses this storage.
pub struct CommandBuffer<'a> {
    records: &'a [u8],
}

impl<'a> CommandBuffer<'a> {
    /// Reset all commands and the fault flag. A size mismatch leaves storage unchanged.
    pub fn initialize(
        bytes: &mut [u8],
        capacity: u32,
        descriptor: &CommandDescriptor,
        inheritance: Inheritance,
    ) -> Result<(), CommandBufferError> {
        let header = descriptor.encode_header(capacity, inheritance)?;
        require_buffer_length(bytes, capacity)?;
        let mut reset = [0; COMMAND_BYTES];
        CommandRecord::default()
            .encode_into(&mut reset)
            .expect("default record is valid");
        bytes[..COMMAND_BUFFER_HEADER_BYTES].copy_from_slice(&header);
        for record in bytes[COMMAND_BUFFER_HEADER_BYTES..]
            .as_chunks_mut::<COMMAND_BYTES>()
            .0
            .iter_mut()
        {
            record.copy_from_slice(&reset);
        }
        Ok(())
    }

    /// Reject producer faults and malformed record structure before exposing any command.
    /// The initialized descriptor and inheritance must match; resource identities and device
    /// limits still require host validation.
    pub fn decode(
        bytes: &'a [u8],
        descriptor: &CommandDescriptor,
        inheritance: Inheritance,
    ) -> Result<Self, CommandBufferError> {
        if bytes.len() < COMMAND_BUFFER_HEADER_BYTES {
            return Err(CommandBufferError::Length);
        }
        let capacity = get_word(bytes, 0);
        require_buffer_length(bytes, capacity)?;
        descriptor.validate_header(&bytes[..COMMAND_BUFFER_HEADER_BYTES], capacity, inheritance)?;
        let records = &bytes[COMMAND_BUFFER_HEADER_BYTES..];
        for (index, record) in records.as_chunks::<COMMAND_BYTES>().0.iter().enumerate() {
            // The checked header capacity bounds the record count to u32.
            let record =
                CommandRecord::decode(record).map_err(|error| CommandBufferError::Record {
                    index: index as u32,
                    error,
                })?;
            descriptor
                .validate_record(&record)
                .map_err(CommandBufferError::Descriptor)?;
            if inheritance.pipeline == StateSource::Parent
                && (record.bindings.pipeline.is_some() || record.retained_pipeline.is_some())
            {
                return Err(CommandBufferError::Descriptor(
                    "indirect command sets pipeline state while inheriting it".into(),
                ));
            }
            if inheritance.buffers == StateSource::Parent
                && BufferStage::of(false)
                    .into_iter()
                    .flat_map(|stage| record.bindings.buffers(stage))
                    .any(Option::is_some)
            {
                return Err(CommandBufferError::Descriptor(
                    "indirect command sets buffer state while inheriting it".into(),
                ));
            }
        }
        Ok(Self { records })
    }

    /// Select a half-open execution range after validating the complete producer buffer.
    /// Empty ranges at any position through capacity are valid.
    pub fn records(
        &self,
        range: std::ops::Range<u32>,
    ) -> Result<impl ExactSizeIterator<Item = CommandRecord> + 'a, CommandBufferError> {
        let capacity = (self.records.len() / COMMAND_BYTES) as u32;
        if range.start > range.end || range.end > capacity {
            return Err(CommandBufferError::Range {
                start: range.start,
                end: range.end,
                capacity,
            });
        }
        // Decode established that capacity * COMMAND_BYTES fits this allocation.
        Ok(
            self.records[range.start as usize * COMMAND_BYTES..range.end as usize * COMMAND_BYTES]
                .as_chunks::<COMMAND_BYTES>()
                .0
                .iter()
                .map(|bytes| {
                    CommandRecord::decode(bytes).expect("immutable records were validated")
                }),
        )
    }
}

fn require_buffer_length(bytes: &[u8], capacity: u32) -> Result<(), CommandBufferError> {
    let expected = usize::try_from(capacity)
        .ok()
        .and_then(|count| count.checked_mul(COMMAND_BYTES))
        .and_then(|size| size.checked_add(COMMAND_BUFFER_HEADER_BYTES));
    if expected == Some(bytes.len()) {
        Ok(())
    } else {
        Err(CommandBufferError::Length)
    }
}

/// Borrowed state selection. The host must still resolve resource identities and pipeline limits.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedBindings<'a> {
    pub pipeline: Option<NonZeroU64>,
    /// Encoded command-local lengths; buffer inheritance does not replace explicit lengths.
    /// Replay must validate the allocation for every used dynamic binding.
    pub threadgroup_memory_lengths: &'a [u32; THREADGROUP_COUNT],
    buffers: &'a [[Option<BufferBinding>; BINDING_COUNT]; BUFFER_STAGES],
}

impl ResolvedBindings<'_> {
    pub fn buffers(&self, stage: BufferStage) -> &[Option<BufferBinding>; BINDING_COUNT] {
        &self.buffers[stage.index()]
    }
}

/// Why a command's state could not be resolved: a command that runs has no pipeline from the
/// state source its descriptor names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingError {
    MissingPipeline { source: StateSource },
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPipeline { source } => write!(
                f,
                "indirect command requires a pipeline from {source:?} state"
            ),
        }
    }
}

impl std::error::Error for BindingError {}

fn validate_record(bytes: &[u8]) -> Result<(Action, bool), RecordError> {
    require_length(bytes)?;
    let grid = [get_word(bytes, 4), get_word(bytes, 5), get_word(bytes, 6)];
    let local_size = [get_word(bytes, 7), get_word(bytes, 8), get_word(bytes, 9)];
    // An indexed or patch draw's arguments past the six (`layout::DRAW_ARGUMENT_WORDS`).
    let wide = |index: usize| get_word(bytes, 4 + index);
    let address = |index: usize| get_u64(bytes, 4 + index);
    let action = match get_word(bytes, 0) {
        0 => Action::Empty,
        1 => Action::DispatchThreadgroups { grid, local_size },
        2 => Action::DispatchThreads { grid, local_size },
        3 => Action::DrawPrimitives {
            primitive_type: grid[0],
            vertex_start: grid[1],
            vertex_count: grid[2],
            instance_count: local_size[0],
            base_instance: local_size[1],
        },
        4 => Action::DrawIndexedPrimitives {
            primitive_type: wide(0),
            index_count: wide(1),
            index_type: wide(2),
            instance_count: wide(3),
            index_buffer: address(4),
            base_vertex: wide(6) as i32,
            base_instance: wide(7),
        },
        5 => Action::DrawPatches {
            control_points: wide(0),
            patch_start: wide(1),
            patch_count: wide(2),
            instance_count: wide(3),
            patch_index_buffer: address(4),
            factor_buffer: address(6),
            base_instance: wide(8),
            factor_instance_stride: wide(9),
            factor_scale: wide(10),
        },
        6 => Action::DrawIndexedPatches {
            control_points: wide(0),
            patch_start: wide(1),
            patch_count: wide(2),
            instance_count: wide(3),
            patch_index_buffer: address(4),
            control_point_index_buffer: get_u64(bytes, layout::CONTROL_POINT_INDICES as usize / 4),
            factor_buffer: address(6),
            base_instance: wide(8),
            factor_instance_stride: wide(9),
            factor_scale: wide(10),
        },
        7 => Action::DrawMeshThreadgroups {
            threadgroups: [wide(0), wide(1), wide(2)],
            object_threads: [wide(3), wide(4), wide(5)],
            mesh_threads: [wide(6), wide(7), wide(8)],
        },
        8 => Action::DrawMeshThreads {
            threads: [wide(0), wide(1), wide(2)],
            object_threads: [wide(3), wide(4), wide(5)],
            mesh_threads: [wide(6), wide(7), wide(8)],
        },
        value => return Err(RecordError::Action { value }),
    };
    let barrier = match get_word(bytes, 1) {
        0 => false,
        1 => true,
        value => return Err(RecordError::Barrier { value }),
    };
    Ok((action, barrier))
}

pub(super) fn get_word(bytes: &[u8], word: usize) -> u32 {
    u32::from_le_bytes(bytes[word * 4..word * 4 + 4].try_into().unwrap())
}

fn get_u64(bytes: &[u8], word: usize) -> u64 {
    u64::from_le_bytes(bytes[word * 4..word * 4 + 8].try_into().unwrap())
}

impl CommandRecord {
    /// Resolve the descriptor's state sources without copying bindings or changing the record.
    /// A nonempty command requires a pipeline from the selected source: the other source is never
    /// a fallback. Threadgroup lengths come from the command; allocation and synchronization
    /// remain separate replay responsibilities.
    pub fn resolve_bindings<'a>(
        &'a self,
        inheritance: Inheritance,
        parent: &'a Bindings,
    ) -> Result<ResolvedBindings<'a>, BindingError> {
        let pipeline = match inheritance.pipeline {
            // A pipeline a reset retained runs, as Metal's does.
            StateSource::Command => self.bindings.pipeline.or(self.retained_pipeline),
            StateSource::Parent => parent.pipeline,
        };
        if self.action != Action::Empty && pipeline.is_none() {
            return Err(BindingError::MissingPipeline {
                source: inheritance.pipeline,
            });
        }
        let buffers = match inheritance.buffers {
            StateSource::Command => &self.bindings.buffers,
            StateSource::Parent => &parent.buffers,
        };
        Ok(ResolvedBindings {
            pipeline,
            buffers,
            threadgroup_memory_lengths: &self.threadgroup_memory_lengths,
        })
    }

    /// Decode persistent state without executing it or resolving its host resource identities.
    pub fn decode(bytes: &[u8]) -> Result<Self, RecordError> {
        let (action, barrier) = validate_record(bytes)?;
        let pipeline = NonZeroU64::new(get_u64(bytes, 2));
        let retained = match get_word(bytes, layout::STATE as usize / 4) {
            0 => false,
            layout::RETAINED if pipeline.is_some() => true,
            _ => return Err(RecordError::PipelineState),
        };
        let mut record = Self {
            action,
            barrier,
            bindings: Bindings {
                pipeline: pipeline.filter(|_| !retained),
                ..Default::default()
            },
            retained_pipeline: pipeline.filter(|_| retained),
            ..Default::default()
        };
        // An address and a stride of 0 are a slot no setter set, as Metal's getters answer a
        // fresh slot's (and a nil buffer's set at offset 0 with stride 0, which is the same).
        for (index, binding) in record.bindings.buffers.iter_mut().flatten().enumerate() {
            let word = HEADER_WORDS + index * BUFFER_WORDS;
            let address = get_u64(bytes, word);
            let stride = get_u64(bytes, word + 2);
            *binding = (address != 0 || stride != 0).then_some(BufferBinding {
                address,
                attribute_stride: (stride != u64::MAX).then_some(stride),
            });
        }
        // An indexed patch draw's control-point index buffer takes the first two threadgroup
        // lengths' words.
        let lengths_from = match record.action {
            Action::DrawIndexedPatches { .. } => 2,
            _ => 0,
        };
        for (index, length) in record
            .threadgroup_memory_lengths
            .iter_mut()
            .enumerate()
            .skip(lengths_from)
        {
            *length = get_word(bytes, THREADGROUP_WORD + index);
        }
        // An indexed or patch draw's arguments take the stage-in region's words.
        if !record.action.takes_stage_in_words() {
            for (index, value) in record.stage_in_region.iter_mut().enumerate() {
                *value = get_word(bytes, layout::STAGE_IN as usize / 4 + index);
            }
        }
        let imageblock = layout::IMAGEBLOCK as usize;
        for (index, value) in record.imageblock_size.iter_mut().enumerate() {
            let at = imageblock + 2 * index;
            *value = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        }
        Ok(record)
    }
}
