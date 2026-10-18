//! The pass's imageblock storage: what a render encoder keeps for a pass whose draws' fragment
//! functions declare `[[imageblock_data]]` (`GraphicsState::fragment_imageblock`), as
//! [`ImageblockMechanism::StorageImageInterlock`]'s contract asks of a consumer, and whose tile
//! functions' explicit imageblocks are the same memory (`RuntimeFact::TileImageblock`).
//!
//! - **The storage** stands in for Metal's tile memory, which keeps the imageblock per pixel for
//!   the whole pass: an `R32_UINT` 2D-array storage image of every pixel of every tile over the
//!   render area (from the origin; the render area itself in a pass of no tile size) and one
//!   layer per word of the cell (`FragmentImageblock::words`), whose layer `w` holds word `w` of
//!   pixel `(x, y)`'s cell at texel `(x, y)`. Made at the pass's first draw or tile dispatch that
//!   reaches it, it rests in `GENERAL` like every runtime image, and is bound to each such draw
//!   and dispatch as a storage view of every layer where the plan writes
//!   `WriteSource::Imageblock`.
//! - **Its contents.** Metal's are undefined as the pass begins (MSL); an M4 read zero there,
//!   through a fragment function's view and a tile function's, in a pass after one that wrote
//!   every cell. The storage is cleared to zero before the pass's first
//!   render pass instance, between barriers from everything before and to everything after: the
//!   answer measured, the one chosen where Metal states none, and no function reads what the
//!   memory held before. It lives across the instances a pass splits into (a barrier from
//!   everything before to everything after runs between them, `render.rs`); the command buffer
//!   holds it until it completes, and nothing reads it after the pass, as nothing reads Metal's
//!   tile memory then.
//! - **Order.** The fragment functions read and write it inside the ordered pixel interlock,
//!   which orders the overlapping fragments of a draw in primitive order, as Metal's raster order
//!   groups order them. Between two draws of a render pass instance that reach it, a barrier from
//!   the first's fragment shader writes to the second's reads and writes, by region
//!   (`render_target::between_storage_draws`, the draws' subpass's self-dependency): RADV on an
//!   AMD 780M (Mesa 26.2.4) did not order such draws by the interlock alone. A case's draw between
//!   the Metal oracle's initializer and resolver (each another pipeline) left no write the
//!   resolver read, in eight public cases, until a split or the barrier came between the draws.
//!   Between instances the split's barrier orders them.
//! - **Masters.** Metal places a pass's explicit imageblock after its color attachments, at the
//!   bytes they pack to rounded up to 8 and then to the layout's alignment
//!   (`explicit_imageblock_sample_length`), and every layout views it at its master's offsets:
//!   a draw whose view is of another master of the same size and alignment reads and writes the
//!   bytes the first one's views did at its own offsets, as the storage's words hold them
//!   (a probe on the M3 Max and an M4: a `uint` pair read as two `ushort`s and a `uint` gave each
//!   byte where it was written). Metal's API validation stops the draw ("input image-block layout
//!   must match the previous state image-block layout"); the runtime draws it as Metal does without
//!   it. A draw whose cell is of another size, or of another alignment, is refused
//!   (`NotImplemented(ImageblockMasters)`): of another size, probe 5's bytes past a smaller layout
//!   read zero after its draw over every pixel, but the partial draw's answer is not read and the
//!   storage keeps them.
//! - **Tile functions**. A tile function's explicit imageblock that stands alone (no implicit one
//!   beside it, and not aliased onto one) is the same storage: its dispatch stages each cell of
//!   each tile it runs from the storage into its threadgroup cells before the function runs, and
//!   back after it where the function may write them (`lower::access::tile`), a tile's cells past
//!   the render area included, which no draw reaches. Measured on an M4 (a
//!   pass of 48x40 pixels in 16x16 tiles): a draw's fragment imageblock and a tile dispatch's are
//!   each pixel's same bytes, so a tile function reads the words a draw wrote and a draw the words
//!   a dispatch wrote, in the order the encoder recorded them; a dispatch reads what the dispatch
//!   before it wrote; and a tile function's layout of other members but the same size reads the
//!   same bytes at its own offsets, which Metal's validation stops ("tile shader input image-block
//!   layout must match the previous state's image-block layout") and the runtime runs as Metal does
//!   without it. A dispatch whose imageblock is not the storage's bytes, and a draw whose cell is
//!   not, are refused (`NotImplemented(ImageblockMasters)`): which bytes Metal's layouts of two
//!   sizes reach is not read. The first draw's alignment is the pass's, after dispatches that made
//!   the storage too. A tile function whose explicit imageblock is aliased onto its implicit one
//!   keeps it in threadgroup memory of its dispatch, staged from and to the attachments, where
//!   Metal's is the tile memory the storage stands in for. Metal's aliased layout views the color
//!   attachments at its first bytes and the explicit imageblock after them (a probe: a `uint` over
//!   an `R32Uint` attachment read the texel a draw wrote, and wrote one the next draw read, the
//!   draw's explicit imageblock kept; a layout of three read the texel, 0, and the explicit
//!   imageblock's first word), so a layout no larger than the attachments pack to reaches only
//!   them, as the dispatch stages them: it runs beside the storage (Metal's validation stops it,
//!   "tile shader input image-block layout must match the previous state's image-block layout"). A
//!   larger one in a pass that holds the storage, and a draw or dispatch that reaches the storage
//!   in a pass that ran one, are refused (`NotImplemented(TileAndFragmentImageblocks)`).
//! - **The cost** is announced once per device (`fragment-imageblock-storage`): the image, and
//!   the bandwidth of each draw's reads and writes of it, which Metal keeps in tile memory.
//!
//! [`ImageblockMechanism::StorageImageInterlock`]: metal2vulkan::pipeline::ImageblockMechanism

use super::binding_view::VkView;
use super::blit::TRANSFER;
use super::command_buffer::CommandBuffer;
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::image_plan::{sample_count, ImagePlan};
use super::render_pass::PassLayout;
use super::resource::ResourceKey;
use super::texture::Texture;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::Device;
use ash::vk;
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{FragmentImageblock, ImageblockMechanism};
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
    ImageViewType,
};

/// A pass's fragment imageblock storage (the module docs).
#[derive(Clone)]
pub(super) struct PassImageblock {
    texture: Texture,
    /// One layer per word of the cell.
    words: u32,
    /// The bytes of a cell: the fragment functions' `air.imageblock_data_size`, or the tile
    /// function's cell.
    bytes: u32,
    /// The alignment of the layouts every draw that reaches the storage views: the first
    /// draw's, or none while only tile dispatches reached it.
    alignment: Option<u32>,
    /// The pass's render area, from the origin.
    extent: [u32; 2],
}

impl PassImageblock {
    /// The storage `used` asks for over the render area `extent` (the module docs).
    pub(super) fn new(
        maker: &CommandBuffer,
        used: &FragmentImageblock,
        layout: &PassLayout,
    ) -> Result<PassImageblock, Error> {
        if used.mechanism != ImageblockMechanism::StorageImageInterlock {
            return Err(internal(
                "a fragment imageblock of no storage the runtime keeps",
            ));
        }
        PassImageblock::of_cell(
            maker,
            used.data_size,
            used.words,
            Some(used.alignment),
            layout,
        )
    }

    /// The storage a tile function's explicit imageblock of `bytes` bytes, whose cells it stages
    /// as `words` words, asks for in a pass of `layout`, before any draw names its alignment.
    pub(super) fn for_tile(
        maker: &CommandBuffer,
        bytes: u32,
        words: u32,
        layout: &PassLayout,
    ) -> Result<PassImageblock, Error> {
        PassImageblock::of_cell(maker, bytes, words, None, layout)
    }

    fn of_cell(
        maker: &CommandBuffer,
        bytes: u32,
        words: u32,
        alignment: Option<u32>,
        layout: &PassLayout,
    ) -> Result<PassImageblock, Error> {
        if words == 0 || words < bytes.div_ceil(4) {
            return Err(internal(
                "an imageblock storage of fewer words than its cell",
            ));
        }
        let [width, height] = covered(layout);
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2DArray,
            pixel_format: PixelFormat::R32Uint,
            width: width.into(),
            height: height.into(),
            array_length: words.into(),
            usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let plan = ImagePlan {
            pixel_format: PixelFormat::R32Uint,
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::TYPE_2D,
            format: Format::R32_UINT,
            extent: [width, height, 1],
            levels: 1,
            layers: words,
            samples: sample_count(1).ok_or_else(|| internal("one sample"))?,
            usage: ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(Format::R32_UINT, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_ATOMIC
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
        Ok(PassImageblock {
            texture,
            words,
            bytes,
            alignment,
            extent: layout.extent,
        })
    }

    /// Whether a draw of `used` may reach the storage: its cell is the storage's bytes, and its
    /// layouts' alignment the pass's (the first draw's, where a draw named one), whatever its
    /// master (the module docs).
    pub(super) fn takes(&self, used: &FragmentImageblock) -> bool {
        used.data_size == self.bytes
            && used.words <= self.words
            && self
                .alignment
                .is_none_or(|alignment| alignment == used.alignment)
    }

    /// Whether a tile dispatch of an explicit imageblock of `bytes` bytes, staged as `words`
    /// words, may reach the storage: its cell is the storage's bytes (the module docs).
    pub(super) fn takes_tile(&self, bytes: u32, words: u32) -> bool {
        bytes == self.bytes && words <= self.words
    }

    /// Hold `alignment` as the pass's, where no draw named one yet.
    pub(super) fn adopt(&mut self, alignment: u32) {
        self.alignment.get_or_insert(alignment);
    }

    /// The storage view the plan's `WriteSource::Imageblock` binds: every layer.
    pub(super) fn view(&self) -> Result<vk::ImageView, Error> {
        self.texture.image_view(&VkView {
            view_type: ImageViewType::TYPE_2D_ARRAY,
            format: Format::R32_UINT,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: self.words,
            usage: ImageUsageFlags::STORAGE,
        })
    }

    /// A draw's uses of the storage: the fragment shader reads and writes every layer.
    pub(super) fn draw_uses(&self) -> Result<Vec<Use<ResourceKey>>, Error> {
        self.uses(vk::PipelineStageFlags2::FRAGMENT_SHADER)
    }

    /// A tile dispatch's uses of the storage: its compute shader reads and writes every layer.
    pub(super) fn dispatch_uses(&self) -> Result<Vec<Use<ResourceKey>>, Error> {
        self.uses(vk::PipelineStageFlags2::COMPUTE_SHADER)
    }

    fn uses(&self, stage: vk::PipelineStageFlags2) -> Result<Vec<Use<ResourceKey>>, Error> {
        let read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
        let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let layers = 0..self.words;
        Ok(vec![
            self.texture.image_use(0..1, layers.clone(), read, false)?,
            self.texture.image_use(0..1, layers, write, true)?,
        ])
    }

    /// The storage's clear to zero before the pass, ready to record.
    pub(super) fn clear(&self) -> Result<ImageblockClear, Error> {
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        Ok(ImageblockClear {
            image: self.texture.raw_image()?,
            layers: self.words,
            uses: [self.texture.image_use(0..1, 0..self.words, write, true)?],
        })
    }

    /// The storage, which the command buffer holds until it completes.
    pub(super) fn held(&self) -> Texture {
        self.texture.clone()
    }

    /// The storage's words a cell, the pass's render area, and the storage's bytes where the
    /// device placed it: what its degrade notice names.
    pub(super) fn described(&self) -> (u32, [u32; 2], Option<u64>) {
        (self.words, self.extent, self.texture.placed_bytes())
    }
}

/// The pixels a pass's storage covers: every pixel of every tile over the pass's render area
/// (`layout.extent`, from the origin), where a tile function's cells past the render area are,
/// or the render area in a pass of no tile size (the module docs).
fn covered(layout: &PassLayout) -> [u32; 2] {
    let [width, height] = layout.extent;
    match layout.tile.tile {
        Some([tw, th]) => [
            width.div_ceil(tw).saturating_mul(tw),
            height.div_ceil(th).saturating_mul(th),
        ],
        None => layout.extent,
    }
}

/// The recorded form of a pass's fragment imageblock storage's clear
/// ([`PassImageblock::clear`]).
pub(super) struct ImageblockClear {
    image: vk::Image,
    layers: u32,
    uses: [Use<ResourceKey>; 1],
}

impl ImageblockClear {
    /// Record the clear of every layer to zero in `core`'s command buffer, outside the pass,
    /// between barriers from everything before and to everything after.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ImageblockClear {
            image,
            layers,
            uses,
        } = self;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        };
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            let zero = vk::ClearColorValue { uint32: [0; 4] };
            // SAFETY: the encoder's command buffer is recording outside a render pass; the image
            // is live until it completes (the encoder retains its texture), rests in `GENERAL`,
            // and takes transfer writes; the range is its one level and every layer.
            unsafe {
                device.cmd_clear_color_image(raw, image, vk::ImageLayout::GENERAL, &zero, &[range]);
            }
        });
        core.barrier(Barrier::everything());
    }
}
