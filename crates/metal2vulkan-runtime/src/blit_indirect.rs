//! The blit encoder's indirect command buffer commands (`copyIndirectCommandBuffer:...`,
//! `resetCommandsInBuffer:withRange:`, `optimizeIndirectCommandBuffer:withRange:`).
//!
//! - **What Metal answers** (an M3 Max, with and without `MTL_DEBUG_LAYER`). A copy between buffers
//!   of one descriptor copies the commands: the destination's getters and execution read them.
//!   Between buffers of different descriptors the copy is garbled, and a copy that runs past the
//!   destination's end corrupts its command 0; validation stops neither. A reset resets the
//!   commands as `resetWithRange:` does. An optimize changes nothing a getter or an execution
//!   observes.
//! - **The runtime.** The commands are records of one format whatever the descriptor, so a copy
//!   copies the records' bytes ([`BlitCommandEncoder::copy_from_buffer_to_buffer`], which moves
//!   overlapping ranges of one buffer as `memmove` does); a record the destination's descriptor
//!   does not allow runs nothing where an execution meets it (`ReplayFaults::RECORD`), the defined
//!   answer for Metal's garbled one. A reset copies a reset record (`CommandRecord::default`'s
//!   bytes, in the command buffer's upload arena) over each command. Both write the destination's
//!   records on the GPU as the command buffer runs, so the call marks it written by the GPU
//!   (`IndirectCommandBuffer::written_by_gpu`): its executions run the records as they stand once
//!   the work before them completed. An optimize records nothing.
//! - **Refused** (`Invalid`, changing nothing): a buffer of another device, and a range past
//!   either buffer's size, which Metal runs as a corruption no program relies on.
//! - **What a command holds** (the M4 probe, plain Metal). A copy and a reset hold the
//!   records they read and write until the command buffer completes, but not the buffers: a buffer
//!   the program releases meanwhile gives its ID back at once, as Metal handed a blit's buffer's ID
//!   out again while the blit waited. An optimize holds nothing.

use super::blit::{reads, writes, BlitCommandEncoder};
use super::error::{internal, Error};
use super::indirect_command_buffer::{IndirectCommandBuffer, Overflow};
use super::replay::BlitCall;
use ash::vk;
use metal2vulkan::bind::indirect_commands::{CommandRecord, COMMAND_BYTES};
use std::ops::Range;

impl BlitCommandEncoder {
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`: commands
    /// `source_range` of `source` become commands `destination_index..` of `destination` as the
    /// command buffer runs, as on Metal (the module docs). An empty range copies nothing.
    ///
    /// Refused (`Invalid`): a buffer of another device, a range past either buffer's size
    /// (`IndirectCommandBufferProblem::Range`), and a buffer of no command types beside one of
    /// command types (`NoCommandTypes`, unmeasured on Metal); two buffers of no command types
    /// copy as any others, as plain Metal's do.
    pub fn copy_indirect_command_buffer(
        &mut self,
        source: &IndirectCommandBuffer,
        source_range: Range<u64>,
        destination: &IndirectCommandBuffer,
        destination_index: u64,
    ) -> Result<(), Error> {
        self.log(|_| {
            BlitCall::CopyIndirectCommandBuffer(Box::new((
                source.clone(),
                source_range.clone(),
                destination.clone(),
                destination_index,
            )))
        });
        const CALL: &str = "copy_indirect_command_buffer";
        self.core().check_device(CALL, source.device())?;
        self.core().check_device(CALL, destination.device())?;
        // Two buffers of no command types copy; one beside a buffer of command types was not
        // measured.
        if source.has_no_command_types() != destination.has_no_command_types() {
            source.check_command_types(CALL)?;
            destination.check_command_types(CALL)?;
        }
        let from = source.command_range(CALL, source_range)?;
        let count = u64::from(from.end - from.start);
        let to = destination.command_range(
            CALL,
            destination_index..destination_index.saturating_add(count),
        )?;
        if from.is_empty() {
            return Ok(());
        }
        destination.written_by_gpu();
        // The slots past the records', carried as Metal's copy carries every slot below the count.
        self.copy_overflow(source, from.clone(), destination, to.start)?;
        let bytes = IndirectCommandBuffer::record_bytes(from);
        let at = IndirectCommandBuffer::record_bytes(to).start;
        self.unlogged(|this| {
            this.copy_from_buffer_to_buffer(
                source.storage(),
                bytes.start,
                destination.storage(),
                at,
                bytes.end - bytes.start,
            )
        })?;
        self.core().retain(source.storage().clone());
        self.core().retain(destination.storage().clone());
        Ok(())
    }

    /// `resetCommandsInBuffer:withRange:`: commands `range` of `buffer` are reset, as a fresh
    /// buffer's, as the command buffer runs. An empty range resets nothing.
    ///
    /// Refused (`Invalid`): a buffer of another device, and a range past its size
    /// (`IndirectCommandBufferProblem::Range`). A buffer of no command types is taken as any
    /// other, as plain Metal takes it.
    pub fn reset_commands_in_buffer(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        self.log(|_| BlitCall::ResetCommandsInBuffer(buffer.clone(), range.clone()));
        const CALL: &str = "reset_commands_in_buffer";
        self.core().check_device(CALL, buffer.device())?;
        let range = buffer.command_range(CALL, range)?;
        if range.is_empty() {
            return Ok(());
        }
        let mut reset = [0u8; COMMAND_BYTES];
        CommandRecord::default()
            .encode_into(&mut reset)
            .map_err(|_| internal("a reset record does not encode"))?;
        let device = self.core().device().clone();
        let upload = self.core().command_buffer().state().uploads.upload(
            &device,
            &reset,
            COMMAND_BYTES as u64,
        )?;
        buffer.written_by_gpu();
        let bytes = IndirectCommandBuffer::record_bytes(range.clone());
        let overflow = buffer
            .overflow()
            .map(|overflow| (overflow, overflow.bytes(range.clone())));
        let storage = buffer.storage();
        let (raw, _) = storage.at(0);
        let regions: Vec<vk::BufferCopy> = range
            .map(|index| {
                let at = IndirectCommandBuffer::record_bytes(index..index + 1).start;
                vk::BufferCopy {
                    src_offset: upload.offset(),
                    dst_offset: storage.at(at).1,
                    size: COMMAND_BYTES as u64,
                }
            })
            .collect();
        let from = upload.raw();
        self.command(&[writes(storage, bytes)], |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording; the storage is live and retained
            // until it completes, and the arena's region until the command buffer completes; each
            // region is one record, inside both.
            unsafe { device.cmd_copy_buffer(command_buffer, from, raw, &regions) };
        })?;
        self.core().retain(storage.clone());
        // The slots past the records', cleared as Metal's reset clears every slot.
        if let Some((overflow, bytes)) = overflow {
            self.clear_overflow(overflow, bytes)?;
        }
        Ok(())
    }

    /// Carry the slots past the records' of `source`'s commands `from` into `destination`'s from
    /// command `to`, as the copy carries the records: every slot of a stage both hold, and the
    /// destination's slots past the source's cleared, which the source holds unset.
    fn copy_overflow(
        &mut self,
        source: &IndirectCommandBuffer,
        from: Range<u32>,
        destination: &IndirectCommandBuffer,
        to: u32,
    ) -> Result<(), Error> {
        let Some(into) = destination.overflow() else {
            return Ok(());
        };
        let count = from.end - from.start;
        let held = match source.overflow() {
            None => return self.clear_overflow(into, into.bytes(to..to + count)),
            Some(held) if held.same_layout(into) => {
                let bytes = held.bytes(from);
                let at = into.bytes(to..to).start;
                self.unlogged(|this| {
                    this.copy_from_buffer_to_buffer(
                        held.storage(),
                        bytes.start,
                        into.storage(),
                        at,
                        bytes.end - bytes.start,
                    )
                })?;
                self.core().retain(held.storage().clone());
                self.core().retain(into.storage().clone());
                return Ok(());
            }
            Some(held) => held,
        };
        // Two layouts, so two buffers: no copy overlaps another.
        let read = held.bytes(from.clone());
        let (mut copies, mut clears) = (Vec::new(), Vec::new());
        for (offset, index) in from.enumerate() {
            let runs = held.runs(index, into, to + offset as u32);
            copies.extend(runs.copies);
            clears.extend(runs.clears);
        }
        let (source_raw, _) = held.storage().at(0);
        let (raw, _) = into.storage().at(0);
        let regions: Vec<vk::BufferCopy> = copies
            .iter()
            .map(|&(read, write, size)| vk::BufferCopy {
                src_offset: held.storage().at(read).1,
                dst_offset: into.storage().at(write).1,
                size,
            })
            .collect();
        let fills: Vec<(u64, u64)> = clears
            .iter()
            .map(|&(at, size)| (into.storage().at(at).1, size))
            .collect();
        let written = into.bytes(to..to + count);
        let uses = [reads(held.storage(), read), writes(into.storage(), written)];
        self.command(&uses, |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording; both buffers are live and
            // retained until it completes; each region lies inside both, and none overlaps
            // another (two buffers).
            unsafe {
                if !regions.is_empty() {
                    device.cmd_copy_buffer(command_buffer, source_raw, raw, &regions);
                }
                for &(at, size) in &fills {
                    device.cmd_fill_buffer(command_buffer, raw, at, size, 0);
                }
            }
        })?;
        self.core().retain(held.storage().clone());
        self.core().retain(into.storage().clone());
        Ok(())
    }

    /// Clear `bytes` of `overflow`, the slots of commands a reset or a copy clears.
    fn clear_overflow(&mut self, overflow: &Overflow, bytes: Range<u64>) -> Result<(), Error> {
        if bytes.is_empty() {
            return Ok(());
        }
        let storage = overflow.storage();
        let (raw, at) = storage.at(bytes.start);
        let size = bytes.end - bytes.start;
        self.command(&[writes(storage, bytes)], |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording; the overflow is live and
            // retained until it completes; the range lies inside it, a multiple of 16 bytes.
            unsafe { device.cmd_fill_buffer(command_buffer, raw, at, size, 0) };
        })?;
        self.core().retain(storage.clone());
        Ok(())
    }

    /// `optimizeIndirectCommandBuffer:withRange:`: nothing Metal's optimize changes is
    /// observable, so it records nothing.
    ///
    /// Refused (`Invalid`): a buffer of another device, and a range past its size
    /// (`IndirectCommandBufferProblem::Range`). A buffer of no command types is taken as any
    /// other, as plain Metal takes it.
    pub fn optimize_indirect_command_buffer(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        self.log(|_| BlitCall::OptimizeIndirectCommandBuffer(buffer.clone(), range.clone()));
        const CALL: &str = "optimize_indirect_command_buffer";
        self.core().check_device(CALL, buffer.device())?;
        buffer.command_range(CALL, range)?;
        Ok(())
    }
}
