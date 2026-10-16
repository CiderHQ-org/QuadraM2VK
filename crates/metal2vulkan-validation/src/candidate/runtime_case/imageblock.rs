//! A fragment case's explicit imageblock through the runtime: the Metal oracle's initializer and
//! resolver, generated as AIR.
//!
//! The Metal oracle defines every cell of a fragment case's imageblock before the case's draw,
//! and reads the selected member after it, with two helper fragment functions it compiles from
//! MSL at run time (`metal::platform::imageblocks`), drawn in the case's pass with the case's
//! vertex function, the pass's attachment formats and no color writes: its initializer writes each
//! member of the master layout from its authored plane, and its resolver writes the selected
//! member's plane into a buffer. The runtime compiles no MSL, so the runtime path draws the same
//! two helpers generated as AIR ([`helpers_text`]) from the checked reflection's master, over the
//! cells the executor's own context fills its storage with ([`Cells`]):
//!
//! - **The layout.** One struct of the master's members in declaration order, each of its
//!   reflected type (`half`, `half4`, `uchar4`, `ushort`, `float` or `uint`), whose natural offsets must
//!   be the reflected ones; its `air.struct_type_info` repeats the master's (each member's offset,
//!   size, type name, user key and raster order group), and its data size is the master's, so the
//!   product fingerprints it as the case's master (`FragmentImageblock::master`) and the runtime
//!   draws the helpers and the case in one pass. Its stated alignment is its largest member's.
//! - **The initializer** reads pixel `(x, y)`'s cell from a buffer of the cells, at
//!   `(y * width + x) * cell_bytes` with each member at its offset, and returns it as its
//!   imageblock output, after the early fragment tests (so that link takes it beside a depth or
//!   stencil attachment; nothing is tested then, the encoder's state being Metal's default).
//! - **The resolver** reads the cell as its imageblock input and writes each member into a buffer
//!   of cells laid out alike, whose selected member's plane is the output ([`Cells::select`]).
//!   The oracle's resolver writes only that plane: the bytes the output selects are the same.
//!
//! Both read the pixel from `[[position]]` (`air.convert` of its coordinates, as Metal's compiler
//! spells `uint(position.x)`).

use super::{buffer_with, read_buffer, CaseError};
use crate::air::reflect::FragmentImageblock;
use crate::air_library::AirLibrary;
use crate::candidate::engine::imageblock::Cells;
use crate::case::FragmentImageblockFormat;
use metal2vulkan::metal::{ColorWriteMask, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, Device, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState,
};
use std::fmt::Write as _;

/// The initializer's entry point.
pub const INITIALIZE_ENTRY: &str = "metal2vulkan_imageblock_initialize";
/// The resolver's entry point.
pub const RESOLVE_ENTRY: &str = "metal2vulkan_imageblock_resolve";

const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-\
f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-\
v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";
const TRIPLE: &str = "air64_v28-apple-macosx26.0.0";
/// The module's source name, so that two assemblies of the same helpers are the same bitcode.
const SOURCE_FILENAME: &str = "metal2vulkan-imageblock-helpers";
/// The struct of the master's members.
const STRUCT: &str = "%struct.Metal2VulkanImageblock";

/// One member of the generated struct: its LLVM type, its natural alignment, and the alignment
/// of its element, which each access to the cells' buffer states (a cell starts at a multiple of
/// 4 bytes only).
struct Field {
    ty: &'static str,
    align: u32,
    element: u32,
}

/// The field of a member of `format`.
fn field(format: FragmentImageblockFormat) -> Field {
    let (ty, align, element) = match format {
        FragmentImageblockFormat::Half => ("half", 2, 2),
        FragmentImageblockFormat::Half4 => ("<4 x half>", 8, 2),
        FragmentImageblockFormat::Uchar4 => ("<4 x i8>", 4, 1),
        FragmentImageblockFormat::Ushort => ("i16", 2, 2),
        FragmentImageblockFormat::Float => ("float", 4, 4),
        FragmentImageblockFormat::Uint => ("i32", 4, 4),
    };
    Field { ty, align, element }
}

/// The AIR text (opaque pointers, for `llvm-as`) of the initializer and resolver of `master`, the
/// checked reflection's master layout, over `cells` (their width and cell size), as the module
/// docs state them. Refused: a member of a type the oracle's helpers do not take, and a master
/// whose members do not lie at their natural offsets in declaration order.
pub fn helpers_text(master: &FragmentImageblock, cells: &Cells) -> Result<String, String> {
    let mut fields = Vec::with_capacity(master.members.len());
    let mut end = 0u32;
    let mut alignment = 1u32;
    for member in &master.members {
        let format = FragmentImageblockFormat::from_air_type(&member.type_name, member.size)
            .ok_or_else(|| {
                format!(
                    "imageblock member {} of type {} and size {}, which the helpers do not take",
                    member.semantic, member.type_name, member.size
                )
            })?;
        let field = field(format);
        let at = end.next_multiple_of(field.align);
        if at != member.offset {
            return Err(format!(
                "imageblock member {} at {}, where a natural struct places it at {at}",
                member.semantic, member.offset
            ));
        }
        end = at + member.size;
        alignment = alignment.max(field.align);
        fields.push((member, field));
    }
    if fields.is_empty() || end > cells.cell_bytes {
        return Err(format!(
            "an imageblock master of {} members ending at {end} over {}-byte cells",
            fields.len(),
            cells.cell_bytes
        ));
    }
    let width = cells.dimensions[0];
    let cell_bytes = cells.cell_bytes;
    let types: Vec<&str> = fields.iter().map(|(_, field)| field.ty).collect();
    let mut text = format!(
        "; metal2vulkan generated imageblock helpers.\n\
         source_filename = \"{SOURCE_FILENAME}\"\n\
         target datalayout = \"{DATA_LAYOUT}\"\n\
         target triple = \"{TRIPLE}\"\n\n\
         {STRUCT} = type {{ {} }}\n\n",
        types.join(", ")
    );
    // The address of pixel `position`'s cell in the buffer `cells`.
    let address = |text: &mut String| {
        for line in [
            "%x = extractelement <4 x float> %position, i64 0".to_owned(),
            "%y = extractelement <4 x float> %position, i64 1".to_owned(),
            "%xi = tail call i32 @air.convert.u.i32.f.f32(float %x)".to_owned(),
            "%yi = tail call i32 @air.convert.u.i32.f.f32(float %y)".to_owned(),
            format!("%row = mul i32 %yi, {width}"),
            "%pixel = add i32 %row, %xi".to_owned(),
            format!("%start = mul i32 %pixel, {cell_bytes}"),
            "%start64 = zext i32 %start to i64".to_owned(),
            "%cell = getelementptr inbounds i8, ptr addrspace(1) %cells, i64 %start64".to_owned(),
        ] {
            let _ = writeln!(text, "  {line}");
        }
        for (index, (member, _)) in fields.iter().enumerate() {
            let _ = writeln!(
                text,
                "  %p{index} = getelementptr inbounds i8, ptr addrspace(1) %cell, i64 {}",
                member.offset
            );
        }
    };
    let _ = writeln!(
        text,
        "define <{{ {STRUCT} }}> @{INITIALIZE_ENTRY}(<4 x float> noundef %position, \
         ptr addrspace(1) noundef readonly %cells) {{"
    );
    address(&mut text);
    let mut value = "poison".to_owned();
    for (index, (_, field)) in fields.iter().enumerate() {
        let _ = writeln!(
            text,
            "  %v{index} = load {ty}, ptr addrspace(1) %p{index}, align {align}",
            ty = field.ty,
            align = field.element
        );
        let _ = writeln!(
            text,
            "  %s{index} = insertvalue {STRUCT} {value}, {ty} %v{index}, {index}",
            ty = field.ty
        );
        value = format!("%s{index}");
    }
    let _ = writeln!(
        text,
        "  %out = insertvalue <{{ {STRUCT} }}> undef, {STRUCT} {value}, 0\n  \
         ret <{{ {STRUCT} }}> %out\n}}\n"
    );
    let _ = writeln!(
        text,
        "define void @{RESOLVE_ENTRY}({STRUCT} %data, <4 x float> noundef %position, \
         ptr addrspace(1) noundef %cells) {{"
    );
    address(&mut text);
    for (index, (_, field)) in fields.iter().enumerate() {
        let _ = writeln!(
            text,
            "  %e{index} = extractvalue {STRUCT} %data, {index}\n  \
             store {ty} %e{index}, ptr addrspace(1) %p{index}, align {align}",
            ty = field.ty,
            align = field.element
        );
    }
    text.push_str("  ret void\n}\n\ndeclare i32 @air.convert.u.i32.f.f32(float)\n\n");
    // The metadata: the master's layout, each entry's outputs and inputs, and the module's.
    let layout = fields
        .iter()
        .map(|(member, _)| {
            format!(
                "i32 {}, i32 {}, i32 0, !\"{}\", !\"{}\", !\"air.raster_order_group\", i32 {}",
                member.offset,
                member.size,
                member.type_name,
                member.semantic,
                member.raster_order_group
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let imageblock = format!(
        "!\"air.imageblock_data\", !\"air.imageblock_data_size\", i32 {}, \
         !\"air.struct_type_info\", !0, !\"air.arg_type_align_size\", i32 {alignment}, \
         !\"air.arg_type_name\", !\"Metal2VulkanImageblock\", !\"air.arg_name\", !\"data\"",
        master.sample_size
    );
    let position = |index: u32| {
        format!(
            "!{{i32 {index}, !\"air.position\", !\"air.center\", !\"air.no_perspective\", \
             !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}}"
        )
    };
    let buffer = |index: u32, access: &str| {
        format!(
            "!{{i32 {index}, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
             !\"{access}\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, \
             !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", \
             !\"air.arg_name\", !\"cells\"}}"
        )
    };
    let nodes = [
        format!("!{{{layout}}}"),
        // The initializer: its output, its inputs, and its entry, after the early tests.
        format!("!{{!{{{imageblock}}}}}"),
        format!("!{{{}, {}}}", position(0), buffer(1, "air.read")),
        format!("!{{ptr @{INITIALIZE_ENTRY}, !1, !2, !\"early_fragment_tests\"}}"),
        // The resolver: no output, its inputs, and its entry.
        "!{}".to_owned(),
        format!(
            "!{{!{{i32 0, {imageblock}}}, {}, {}}}",
            position(1),
            buffer(2, "air.read_write")
        ),
        format!("!{{ptr @{RESOLVE_ENTRY}, !4, !5}}"),
        "!{i32 7, !\"air.max_device_buffers\", i32 31}".to_owned(),
        "!{!\"air.compile.denorms_disable\"}".to_owned(),
        "!{!\"air.compile.fast_math_enable\"}".to_owned(),
        "!{!\"air.compile.framebuffer_fetch_enable\"}".to_owned(),
        "!{i32 2, i32 8, i32 0}".to_owned(),
        "!{!\"Metal\", i32 4, i32 0, i32 0}".to_owned(),
    ];
    text.push_str(
        "!llvm.module.flags = !{!7}\n\
         !air.fragment = !{!3, !6}\n\
         !air.compile_options = !{!8, !9, !10}\n\
         !air.version = !{!11}\n\
         !air.language_version = !{!12}\n\n",
    );
    for (id, node) in nodes.iter().enumerate() {
        let _ = writeln!(text, "!{id} = {node}");
    }
    Ok(text)
}

/// A fragment case's helpers on a device, as the Metal oracle draws them (the module docs): the
/// initializer's state and the buffer of the cells it reads, and, where the case's output is an
/// imageblock member, the resolver's state and the buffer it writes.
pub(super) struct Helpers {
    initialize: RenderPipelineState,
    cells: Buffer,
    resolve: Option<(RenderPipelineState, Buffer)>,
    /// The cells' extent and cell size, which the resolver's buffer holds.
    dimensions: [u32; 2],
    cell_bytes: u32,
}

impl Helpers {
    /// The helpers of `master` over `cells` on `device`, drawn with `case`'s vertex function and
    /// attachment formats (the case's descriptor), no color written; the resolver where
    /// `resolves`.
    pub(super) fn new(
        device: &Device,
        case: &RenderPipelineDescriptor,
        master: &FragmentImageblock,
        cells: &Cells,
        resolves: bool,
    ) -> Result<Helpers, CaseError> {
        let text = helpers_text(master, cells)?;
        let bitcode = crate::check::loop_free::assemble(&text)?;
        let library = device
            .new_library_with_air(&bitcode)
            .map_err(CaseError::helper(
                "the imageblock helpers' new_library_with_air",
            ))?;
        let state = |entry: &str| {
            let function = library
                .new_function_with_name(entry)
                .map_err(CaseError::helper(format!(
                    "{entry}: new_function_with_name"
                )))?;
            let mut descriptor = RenderPipelineDescriptor::new(case.vertex.clone());
            descriptor.fragment = Some(StageSource::Function(function.function_descriptor()));
            for (helper, own) in descriptor
                .color_attachments
                .iter_mut()
                .zip(&case.color_attachments)
            {
                *helper = own.as_ref().map(|own| {
                    let mut attachment = ColorAttachmentDescriptor::new(own.pixel_format);
                    attachment.write_mask = ColorWriteMask::NONE;
                    attachment
                });
            }
            descriptor.depth_format = case.depth_format;
            descriptor.stencil_format = case.stencil_format;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map_err(CaseError::helper(format!(
                    "{entry}: new_render_pipeline_state"
                )))
        };
        let initialize = state(INITIALIZE_ENTRY)?;
        let resolve = if resolves {
            let resolved = buffer_with(device, &vec![0; cells.bytes.len()])?;
            Some((state(RESOLVE_ENTRY)?, resolved))
        } else {
            None
        };
        Ok(Helpers {
            initialize,
            cells: buffer_with(device, &cells.bytes)?,
            resolve,
            dimensions: cells.dimensions,
            cell_bytes: cells.cell_bytes,
        })
    }

    /// Hold `pass` to the case's `pipeline` as the oracle does: its imageblock sample length the
    /// pipeline's, which each helper's must be ("generated and target fragment imageblock layouts
    /// disagree" otherwise), and a 16x16 tile.
    pub(super) fn prepare_pass(
        &self,
        pass: &mut RenderPassDescriptor,
        pipeline: &RenderPipelineState,
    ) -> Result<(), CaseError> {
        let length = |state: &RenderPipelineState| {
            state
                .imageblock_sample_length()
                .map_err(CaseError::runtime("imageblock_sample_length"))
        };
        let sample_length = length(pipeline)?;
        let helpers =
            std::iter::once(&self.initialize).chain(self.resolve.as_ref().map(|(state, _)| state));
        for helper in helpers {
            let helper = length(helper)?;
            if helper != sample_length {
                return Err(format!(
                    "generated and target fragment imageblock layouts disagree: the helper's \
                     sample length is {helper}, the case's {sample_length}"
                )
                .into());
            }
        }
        pass.imageblock_sample_length = sample_length;
        pass.tile_width = 16;
        pass.tile_height = 16;
        Ok(())
    }

    /// Draw the initializer: the fullscreen triangle of the case's vertex function, reading the
    /// cells at fragment buffer 0.
    pub(super) fn initialize(&self, encoder: &mut RenderCommandEncoder) -> Result<(), CaseError> {
        draw(encoder, &self.initialize, &self.cells)
    }

    /// Draw the resolver, where the case has one, writing the cells at fragment buffer 0.
    pub(super) fn resolve(&self, encoder: &mut RenderCommandEncoder) -> Result<(), CaseError> {
        match &self.resolve {
            Some((state, resolved)) => draw(encoder, state, resolved),
            None => Ok(()),
        }
    }

    /// The draws the helpers add to the case's, before it and after it.
    pub(super) fn draws(&self) -> (usize, usize) {
        (1, usize::from(self.resolve.is_some()))
    }

    /// The cells the resolver wrote, once the command buffer completed.
    pub(super) fn resolved(&self) -> Result<Cells, CaseError> {
        let (_, buffer) = self
            .resolve
            .as_ref()
            .ok_or("an imageblock output without its resolver")?;
        Ok(Cells {
            dimensions: self.dimensions,
            cell_bytes: self.cell_bytes,
            bytes: read_buffer(buffer, 0, buffer.length())?,
        })
    }
}

/// Draw `state`'s fullscreen triangle with `buffer` at fragment buffer 0.
fn draw(
    encoder: &mut RenderCommandEncoder,
    state: &RenderPipelineState,
    buffer: &Buffer,
) -> Result<(), CaseError> {
    encoder
        .set_render_pipeline_state(state)
        .map_err(CaseError::helper(
            "an imageblock helper's set_render_pipeline_state",
        ))?;
    encoder
        .set_fragment_buffer(Some(buffer), 0, 0)
        .map_err(CaseError::helper(
            "an imageblock helper's set_fragment_buffer",
        ))?;
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .map_err(CaseError::helper("an imageblock helper's draw_primitives"))
}
