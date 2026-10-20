//! The runtime's side: Metal's calls on `metal2vulkan_runtime`, on the Vulkan device its loader
//! finds (MoltenVK on a Mac). The device is opened as the consumer opens it (reims-vgpu's
//! `device_options`): GPU times from the host's clock, the exact completion and uncompressed heap
//! textures, the rest of the policy at its defaults, the shader cache in a directory of the
//! bench's own.

use crate::gpu::{Format, Gpu};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::{CompletionPolicy, HeapCompressionPolicy};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, CullMode, IndexType, PixelFormat, PrimitiveType,
    SamplerMinMagFilter, TextureUsage, Winding,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime as rt;
use std::path::Path;
use std::ptr::NonNull;

pub struct Runtime {
    device: rt::Device,
    queue: rt::CommandQueue,
}

impl Runtime {
    /// Open the device with its shader cache in `cache`.
    pub fn open(cache: &Path) -> Runtime {
        let mut options = rt::DeviceOptions::default();
        options.gpu_times_from_host = true;
        options.policy.completion = CompletionPolicy::Exact;
        options.policy.heap_compression = HeapCompressionPolicy::Uncompressed;
        options.shader_cache = rt::ShaderCacheOptions::in_directory(cache);
        let device = rt::Device::system_default_with(&options).expect("open the runtime's device");
        let queue = device.new_command_queue().expect("new_command_queue");
        Runtime { device, queue }
    }

    pub fn device(&self) -> &rt::Device {
        &self.device
    }
}

fn pixel_format(format: Format) -> PixelFormat {
    match format {
        Format::Bgra8 => PixelFormat::BGRA8Unorm,
        Format::Rgba8 => PixelFormat::RGBA8Unorm,
        Format::Rgba16Float => PixelFormat::RGBA16Float,
        Format::Depth32Float => PixelFormat::Depth32Float,
    }
}

impl Gpu for Runtime {
    type Library = rt::Library;
    type RenderPipeline = rt::RenderPipelineState;
    type ComputePipeline = rt::ComputePipelineState;
    type Buffer = rt::Buffer;
    type Texture = rt::Texture;
    type Sampler = rt::SamplerState;
    type CommandBuffer = rt::CommandBuffer;
    type RenderEncoder = rt::RenderCommandEncoder;
    type ComputeEncoder = rt::ComputeCommandEncoder;
    type Heap = rt::Heap;
    type DepthStencil = rt::DepthStencilState;

    fn label(&self) -> String {
        format!("runtime on {}", self.device.name())
    }

    fn library(&self, metallib: &Path) -> rt::Library {
        let bytes = std::fs::read(metallib).expect("read the metallib");
        self.device
            .new_library_with_data(&bytes)
            .expect("new_library_with_data")
    }

    fn render_pipeline(
        &self,
        library: &rt::Library,
        vertex: &str,
        fragment: &str,
        format: Format,
        blend: bool,
    ) -> rt::RenderPipelineState {
        let function = |name: &str| {
            let function = library
                .new_function_with_name(name)
                .expect("new_function_with_name");
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment));
        let mut color = ColorAttachmentDescriptor::new(pixel_format(format));
        color.blending_enabled = blend;
        descriptor.color_attachments[0] = Some(color);
        self.device
            .new_render_pipeline_state(&descriptor, &rt::PipelineOptions::default())
            .expect("new_render_pipeline_state")
    }

    fn depth_pipeline(
        &self,
        library: &rt::Library,
        vertex: &str,
        fragment: &str,
        blend: bool,
    ) -> rt::RenderPipelineState {
        let function = |name: &str| {
            let function = library
                .new_function_with_name(name)
                .expect("new_function_with_name");
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
        if blend {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            color.source_alpha_blend_factor = BlendFactor::One;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        descriptor.color_attachments[0] = Some(color);
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        self.device
            .new_render_pipeline_state(&descriptor, &rt::PipelineOptions::default())
            .expect("new_render_pipeline_state")
    }

    fn depth_stencil(&self, writes: bool) -> rt::DepthStencilState {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = if writes {
            CompareFunction::Less
        } else {
            CompareFunction::LessEqual
        };
        descriptor.depth_write_enabled = writes;
        self.device
            .new_depth_stencil_state(&descriptor)
            .expect("new_depth_stencil_state")
    }

    fn compute_pipeline(&self, library: &rt::Library, kernel: &str) -> rt::ComputePipelineState {
        let function = library
            .new_function_with_name(kernel)
            .expect("new_function_with_name");
        self.device
            .new_compute_pipeline_state_with_function(&function, &rt::PipelineOptions::default())
            .expect("new_compute_pipeline_state_with_function")
    }

    fn buffer(&self, length: usize) -> rt::Buffer {
        self.device
            .new_buffer(length as u64, rt::ResourceOptions::shared())
            .expect("new_buffer")
    }

    fn buffer_with(&self, bytes: &[u8]) -> rt::Buffer {
        self.device
            .new_buffer_with_bytes(bytes, rt::ResourceOptions::shared())
            .expect("new_buffer_with_bytes")
    }

    unsafe fn buffer_no_copy(&self, pointer: *mut u8, length: usize) -> rt::Buffer {
        let pointer = NonNull::new(pointer).expect("a pointer");
        // SAFETY: the caller's contract: page-aligned read-write memory that outlives the buffer.
        unsafe {
            self.device.new_buffer_with_bytes_no_copy(
                rt::HostMemory::new(pointer),
                length as u64,
                rt::ResourceOptions::shared(),
                None,
            )
        }
        .expect("new_buffer_with_bytes_no_copy")
    }

    fn heap(&self, size: usize) -> rt::Heap {
        let descriptor = rt::HeapDescriptor::new(
            rt::HeapType::Placement,
            size as u64,
            rt::ResourceOptions::shared(),
        );
        self.device.new_heap(&descriptor).expect("new_heap")
    }

    unsafe fn heap_no_copy(&self, pointer: *mut u8, size: usize) -> Option<rt::Heap> {
        let pointer = NonNull::new(pointer).expect("a pointer");
        let descriptor = rt::HeapDescriptor::new(
            rt::HeapType::Placement,
            size as u64,
            rt::ResourceOptions::shared(),
        );
        // SAFETY: the caller's contract: page-aligned read-write memory that outlives the heap.
        let heap = unsafe {
            self.device.new_heap_with_bytes_no_copy(
                rt::HostMemory::new(pointer),
                &descriptor,
                None,
                None,
            )
        };
        Some(heap.expect("new_heap_with_bytes_no_copy"))
    }

    fn heap_texture(
        &self,
        heap: &rt::Heap,
        width: usize,
        height: usize,
        offset: usize,
    ) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            PixelFormat::BGRA8Unorm,
            width as u64,
            height as u64,
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = rt::ResourceOptions::shared();
        descriptor.allow_gpu_optimized_contents = false;
        heap.new_texture_at_offset(&descriptor, offset as u64)
            .expect("new_texture_at_offset")
    }

    fn texture(&self, format: Format, width: usize, height: usize, target: bool) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            pixel_format(format),
            width as u64,
            height as u64,
            false,
        );
        if target {
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = rt::ResourceOptions::private();
        } else {
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = rt::ResourceOptions::shared();
        }
        self.device.new_texture(&descriptor).expect("new_texture")
    }

    fn tracked_heap(&self, size: usize) -> rt::Heap {
        let options = rt::ResourceOptions::new(
            rt::StorageMode::Shared,
            rt::CpuCacheMode::DefaultCache,
            rt::HazardTrackingMode::Tracked,
        );
        let descriptor = rt::HeapDescriptor::new(rt::HeapType::Placement, size as u64, options);
        self.device.new_heap(&descriptor).expect("new_heap")
    }

    fn heap_render_texture(
        &self,
        heap: &rt::Heap,
        format: Format,
        side: usize,
        offset: usize,
    ) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            pixel_format(format),
            side as u64,
            side as u64,
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = rt::ResourceOptions::shared();
        heap.new_texture_at_offset(&descriptor, offset as u64)
            .expect("new_texture_at_offset")
    }

    fn replace_region(
        &self,
        texture: &rt::Texture,
        size: (usize, usize),
        bytes: &[u8],
        row: usize,
    ) {
        let region = rt::Region::new_2d(0, 0, size.0 as u64, size.1 as u64);
        texture
            .replace_region(region, 0, 0, bytes, row as u64, 0)
            .expect("replace_region");
    }

    fn get_bytes(&self, texture: &rt::Texture, size: (usize, usize), bytes: &mut [u8], row: usize) {
        let region = rt::Region::new_2d(0, 0, size.0 as u64, size.1 as u64);
        texture
            .get_bytes(bytes, row as u64, 0, region, 0, 0)
            .expect("get_bytes");
    }

    fn read_buffer(&self, buffer: &rt::Buffer, bytes: &mut [u8]) {
        assert!(
            bytes.len() as u64 <= buffer.length(),
            "a read within the buffer"
        );
        let contents = buffer.contents().expect("a Shared buffer's contents");
        // SAFETY: `contents` holds the buffer's `length` bytes, which no GPU work writes now (the
        // caller's contract), and `bytes` is as long as the copy.
        unsafe {
            std::ptr::copy_nonoverlapping(contents.as_ptr(), bytes.as_mut_ptr(), bytes.len())
        };
    }

    fn sampler(&self, linear: bool) -> rt::SamplerState {
        let mut descriptor = SamplerDescriptor::new();
        if linear {
            descriptor.min_filter = SamplerMinMagFilter::Linear;
            descriptor.mag_filter = SamplerMinMagFilter::Linear;
        }
        self.device
            .new_sampler_state(&descriptor)
            .expect("new_sampler_state")
    }

    fn command_buffer(&self) -> rt::CommandBuffer {
        self.queue.command_buffer().expect("command_buffer")
    }

    fn render_encoder(
        &self,
        command_buffer: &rt::CommandBuffer,
        target: &rt::Texture,
        clear: bool,
    ) -> rt::RenderCommandEncoder {
        let mut pass = rt::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = if clear {
            rt::LoadAction::Clear
        } else {
            rt::LoadAction::Load
        };
        color.store_action = rt::StoreAction::Store;
        command_buffer
            .render_command_encoder(&pass)
            .expect("render_command_encoder")
    }

    fn depth_render_encoder(
        &self,
        command_buffer: &rt::CommandBuffer,
        target: &rt::Texture,
        depth: &rt::Texture,
    ) -> rt::RenderCommandEncoder {
        let mut pass = rt::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = rt::LoadAction::Clear;
        color.store_action = rt::StoreAction::Store;
        color.clear_color = rt::ClearColor {
            red: 0.1,
            green: 0.1,
            blue: 0.12,
            alpha: 1.0,
        };
        let attachment = &mut pass.depth_attachment;
        attachment.texture = Some(depth.clone());
        attachment.load_action = rt::LoadAction::Clear;
        attachment.store_action = rt::StoreAction::DontCare;
        attachment.clear_depth = 1.0;
        command_buffer
            .render_command_encoder(&pass)
            .expect("render_command_encoder")
    }

    fn set_render_pipeline(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        state: &rt::RenderPipelineState,
    ) {
        encoder
            .set_render_pipeline_state(state)
            .expect("set_render_pipeline_state");
    }

    fn set_depth_stencil(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        state: &rt::DepthStencilState,
    ) {
        encoder
            .set_depth_stencil_state(state)
            .expect("set_depth_stencil_state");
    }

    fn set_cull_back(&self, encoder: &mut rt::RenderCommandEncoder, back: bool) {
        encoder.set_cull_mode(if back { CullMode::Back } else { CullMode::None });
    }

    fn set_counter_clockwise(&self, encoder: &mut rt::RenderCommandEncoder) {
        encoder.set_front_facing_winding(Winding::CounterClockwise);
    }

    fn set_vertex_bytes(&self, encoder: &mut rt::RenderCommandEncoder, bytes: &[u8], index: usize) {
        encoder
            .set_vertex_bytes(bytes, index as u32)
            .expect("set_vertex_bytes");
    }

    fn set_vertex_buffer(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        index: usize,
    ) {
        encoder
            .set_vertex_buffer(Some(buffer), offset as u64, index as u32)
            .expect("set_vertex_buffer");
    }

    fn set_fragment_buffer(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        index: usize,
    ) {
        encoder
            .set_fragment_buffer(Some(buffer), offset as u64, index as u32)
            .expect("set_fragment_buffer");
    }

    fn set_fragment_texture(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        texture: &rt::Texture,
        index: usize,
    ) {
        encoder
            .set_fragment_texture(Some(texture), index as u32)
            .expect("set_fragment_texture");
    }

    fn set_fragment_sampler(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        sampler: &rt::SamplerState,
        index: usize,
    ) {
        encoder
            .set_fragment_sampler_state(Some(sampler), index as u32)
            .expect("set_fragment_sampler_state");
    }

    fn draw_strip(&self, encoder: &mut rt::RenderCommandEncoder, vertices: usize) {
        encoder
            .draw_primitives(PrimitiveType::TriangleStrip, 0, vertices as u32, 1, 0)
            .expect("draw_primitives");
    }

    fn draw_indexed(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        count: usize,
        indices: &rt::Buffer,
    ) {
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                count as u32,
                IndexType::UInt16,
                indices,
                0,
                1,
                0,
                0,
            )
            .expect("draw_indexed_primitives");
    }

    fn end_render(&self, encoder: rt::RenderCommandEncoder) {
        encoder.end_encoding().expect("end_encoding");
    }

    fn compute_encoder(&self, command_buffer: &rt::CommandBuffer) -> rt::ComputeCommandEncoder {
        command_buffer
            .compute_command_encoder()
            .expect("compute_command_encoder")
    }

    fn set_compute_pipeline(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        state: &rt::ComputePipelineState,
    ) {
        encoder
            .set_compute_pipeline_state(state)
            .expect("set_compute_pipeline_state");
    }

    fn set_buffer(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        index: usize,
    ) {
        encoder
            .set_buffer(Some(buffer), offset as u64, index as u32)
            .expect("set_buffer");
    }

    fn set_texture(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        texture: &rt::Texture,
        index: usize,
    ) {
        encoder
            .set_texture(Some(texture), index as u32)
            .expect("set_texture");
    }

    fn dispatch(&self, encoder: &mut rt::ComputeCommandEncoder, groups: usize, threads: usize) {
        encoder
            .dispatch_threadgroups(
                rt::Size::new(groups as u64, 1, 1),
                rt::Size::new(threads as u64, 1, 1),
            )
            .expect("dispatch_threadgroups");
    }

    fn end_compute(&self, encoder: rt::ComputeCommandEncoder) {
        encoder.end_encoding().expect("end_encoding");
    }

    fn fill(&self, command_buffer: &rt::CommandBuffer, buffer: &rt::Buffer, length: usize) {
        let mut blit = command_buffer
            .blit_command_encoder()
            .expect("blit_command_encoder");
        blit.fill_buffer(buffer, 0..length as u64, 0x5a)
            .expect("fill_buffer");
        blit.end_encoding().expect("end_encoding");
    }

    fn commit(&self, command_buffer: &rt::CommandBuffer) {
        command_buffer.commit().expect("commit");
    }

    fn wait(&self, command_buffer: &rt::CommandBuffer) {
        command_buffer
            .wait_until_completed()
            .expect("wait_until_completed");
        let status = command_buffer.status();
        assert_eq!(
            status,
            rt::CommandBufferStatus::Completed,
            "the command buffer ended {status:?}: {:?}",
            command_buffer.error()
        );
    }

    fn on_completed(&self, command_buffer: &rt::CommandBuffer, handler: Box<dyn FnOnce() + Send>) {
        command_buffer
            .add_completed_handler(Box::new(move |_| handler()))
            .expect("add_completed_handler");
    }

    fn gpu_seconds(&self, _command_buffer: &rt::CommandBuffer) -> Option<f64> {
        // On MoltenVK the runtime answers GPU times from the host's clock (their docs), which
        // is no measure of the GPU.
        None
    }

    fn times(&self, command_buffer: &rt::CommandBuffer) -> [f64; 4] {
        [
            command_buffer.kernel_start_time(),
            command_buffer.kernel_end_time(),
            command_buffer.gpu_start_time().unwrap_or(0.0),
            command_buffer.gpu_end_time().unwrap_or(0.0),
        ]
    }

    fn pool<R>(&self, f: impl FnOnce() -> R) -> R {
        f()
    }

    fn work_counts(&self) -> Vec<(&'static str, f64)> {
        let c = self.device.work_counts();
        let replayed = &c.commands_replayed;
        [
            ("queue_submissions", c.queue_submissions),
            ("submit_infos", c.submit_infos),
            ("vk_command_buffers_submitted", c.command_buffers_submitted),
            ("first_work_buffers", c.command_buffers_begun_batch_work),
            ("barrier_buffers", c.command_buffers_begun_batch_barriers),
            ("finales_apart", c.finales_apart),
            ("render_passes_begun", replayed.cmd_begin_render_pass),
            ("render_encoders_ended", c.render_encoders_ended),
            ("passes_begun_instances", c.render_passes_begun_instances),
            ("passes_begun_inline", c.render_passes_begun_inline),
            (
                "passes_begun_inline_splits",
                c.render_passes_begun_inline_splits,
            ),
            ("passes_begun_resolves", c.render_passes_begun_resolves),
            ("splits_barrier", c.instance_splits_barrier),
            ("splits_tile_dispatch", c.instance_splits_tile_dispatch),
            ("splits_tile_memory", c.instance_splits_tile_memory),
            ("splits_attached_read", c.instance_splits_attached_read),
            ("splits_feedback", c.instance_splits_feedback),
            (
                "splits_memoryless_write",
                c.instance_splits_memoryless_write,
            ),
            (
                "splits_pending_pipeline",
                c.instance_splits_pending_pipeline,
            ),
            ("splits_execution", c.instance_splits_execution),
            ("splits_visibility", c.instance_splits_visibility),
            ("splits_shape", c.instance_splits_shape),
            ("pipeline_barriers", replayed.cmd_pipeline_barrier),
        ]
        .into_iter()
        .map(|(name, count)| (name, count as f64))
        .collect()
    }
}
