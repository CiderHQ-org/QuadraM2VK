//! A render encoder's vertex buffers at a stride of their own:
//! `setVertexBuffer:offset:attributeStride:atIndex:` and its `setVertexBuffers` and
//! `setVertexBufferOffset` forms.
//!
//! - **The stride** is the one a vertex descriptor's buffer layout of
//!   `MTLBufferLayoutStrideDynamic` reads its buffer at: the binder's per-bind stride
//!   (`VertexStride::PerBind`), which a draw records as the binding's stride (a dynamic
//!   `VERTEX_INPUT_BINDING_STRIDE`, or the pipeline's static one), or pushes where the pipeline
//!   pulls the buffer. It is held in the encoder's vertex table beside the buffer
//!   (`argument_table.rs`), as the compute encoder holds a stage-in buffer's.
//! - **`MTLAttributeStrideStatic`** (`u64::MAX`) sets no stride, as the plain setters do; a layout
//!   of a fixed stride reads its own, whatever the bind's.
//! - **The plain setters** bind without a stride (`set_vertex_buffer`, `set_vertex_buffers`), and
//!   so does `set_vertex_buffer_offset`: it clears the stride bound at the index. A draw whose
//!   pipeline's layout is dynamic at a buffer bound without a stride reads it at stride 0, as
//!   plain Metal does: measured on an M4 after the stride form and on an M3 Max and an M4 with no
//!   stride set. Metal's debug layer stops such a draw ("providing `MTLAttributeStrideStatic` to
//!   `attributeStride` when stride at buffer-layout index 0 is dynamic"). The binder reads inline
//!   bytes there alike: bytes set by `set_vertex_bytes` at stride 0, and by
//!   `set_vertex_bytes_with_stride` at its stride. An executed indirect render command's buffer set
//!   without a stride reads otherwise (`render/executions.rs`).

use super::RenderCall;
use super::RenderCommandEncoder;
use crate::buffer::Buffer;
use crate::error::Error;

impl RenderCommandEncoder {
    /// `setVertexBuffer:offset:attributeStride:atIndex:`: as
    /// [`set_vertex_buffer`](Self::set_vertex_buffer), the buffer read at `stride` by a layout of
    /// `MTLBufferLayoutStrideDynamic` (the module docs); `MTLAttributeStrideStatic` (`u64::MAX`)
    /// sets no stride; any other stride past 32 bits is taken modulo `2^32`, as plain Metal
    /// reads it and its validation takes it (an M4). Refused as `set_vertex_buffer` is.
    pub fn set_vertex_buffer_with_stride(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
            RenderCall::VertexBufferWithStride(buffer, offset, stride, index)
        });
        self.vertex.set_buffer_with_stride(
            &mut self.core,
            "set_vertex_buffer_with_stride",
            buffer,
            offset,
            stride,
            index,
        )
    }

    /// `setVertexBytes:length:attributeStride:atIndex:`: as
    /// [`set_vertex_bytes`](Self::set_vertex_bytes), the bytes read at `stride`, its low 32 bits,
    /// by a layout of `MTLBufferLayoutStrideDynamic`. At `MTLAttributeStrideStatic` (`u64::MAX`)
    /// a draw whose layout reads them so is refused (`Bind(MissingVertexStride)`): an M4 read
    /// element 0 of the bytes and zeros past it there, which the fetch does not emulate. Refused as
    /// `set_vertex_bytes` is.
    pub fn set_vertex_bytes_with_stride(
        &mut self,
        bytes: &[u8],
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::VertexBytesWithStride(log.byte_span(bytes), stride, index));
        self.vertex.set_bytes(
            &mut self.core,
            "set_vertex_bytes_with_stride",
            bytes,
            Some(stride),
            index,
        )
    }

    /// `setVertexBuffers:offsets:attributeStrides:withRange:`: `buffers[i]` at offset
    /// `offsets[i]` and stride `strides[i]` at index `first + i`, each as
    /// [`set_vertex_buffer_with_stride`](Self::set_vertex_buffer_with_stride). Refused as that
    /// is, and when the slices' lengths differ; a refusal binds none of them.
    pub fn set_vertex_buffers_with_strides(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        strides: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let spans = (
                log.buffer_span(buffers),
                log.word_span(offsets),
                log.word_span(strides),
                first,
            );
            RenderCall::VertexBuffersWithStrides(Box::new(spans))
        });
        self.vertex.set_buffers_with_strides(
            &mut self.core,
            "set_vertex_buffers_with_strides",
            buffers,
            offsets,
            Some(strides),
            first,
        )
    }

    /// `setVertexBufferOffset:attributeStride:atIndex:`: as
    /// [`set_vertex_buffer_offset`](Self::set_vertex_buffer_offset), with the index's stride set
    /// as [`set_vertex_buffer_with_stride`](Self::set_vertex_buffer_with_stride) sets it.
    /// Refused as both are.
    pub fn set_vertex_buffer_offset_with_stride(
        &mut self,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::VertexBufferOffsetWithStride(offset, stride, index));
        self.vertex.set_buffer_offset_with_stride(
            &mut self.core,
            "set_vertex_buffer_offset_with_stride",
            offset,
            stride,
            index,
        )
    }
}
