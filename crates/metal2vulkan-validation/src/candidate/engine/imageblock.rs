//! A fragment case's imageblock in the engine render candidate.
//!
//! Metal keeps a fragment function's explicit imageblock in tile memory: one cell per pixel, of
//! the master layout's bytes, undefined when the render pass begins. The Metal oracle defines
//! every cell before the case's draw: its generated initializer, drawn first in the same pass,
//! writes each master member's authored plane (zero for a member the case does not author), and
//! its resolver, drawn after the case's draw, reads the selected member's plane
//! (`metal::platform::imageblocks`). So a case observes only cells the oracle defined, and never
//! a byte the pass left undefined.
//!
//! The render executor holds the same cells in the product's representation
//! (`ImageblockMechanism::StorageImageInterlock`): one `R32Uint` 2D-array storage image per pass,
//! layer `w` holding word `w` of every pixel's cell at the pixel's texel, each word the cell's
//! bytes `4w..4w + 4` in little-endian order (the leaves). The cells follow the master of the
//! checked reflection ([`layout`]), the one the checker validates the planes against and the
//! oracle's initializer and resolver use, so they hold the bytes the oracle's pass holds whichever
//! master the product plans its views by. [`Cells::initial`] lays the authored planes out at their
//! master members' offsets, [`Cells::image_bytes`] is the image's contents layer by layer,
//! [`Cells::from_image`] reads the cells back from it, and [`Cells::select`] is the oracle
//! resolver's plane of one member. Bytes no member covers are zero; no view reaches them (a view
//! field no member covers is refused), so no observation depends on them.

use crate::air::reflect::FragmentImageblock;
use crate::literal::LiteralFragmentImageblock;
use std::collections::BTreeMap;

/// One member of a master layout: its byte offset in the cell and its size in bytes, as the
/// reflection states them (an array member, which is refused, is never drawn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub offset: u32,
    pub size: u32,
}

/// The master layout a fragment function's imageblock views view (MSL §4.7.1): its members by
/// user key, which is what an authored plane's `semantic` names, and the bytes of one cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub members: BTreeMap<String, Member>,
    /// The master's `air.imageblock_data_size`, or the end of its last member if larger, rounded
    /// up to a whole word.
    pub cell_bytes: u32,
}

impl Layout {
    /// The 32-bit words of one cell.
    pub fn words(&self) -> u32 {
        self.cell_bytes / 4
    }
}

/// The master layout of a fragment function's explicit imageblock, from the checked reflection
/// (`ShaderInterface::fragment_imageblock`): the largest master of the function's views (the
/// first output view's on a tie, else the first input view's), each member at its reflected
/// offset, keyed by its semantic, and the cell [`crate::check::caps::fragment_imageblock_cell_bytes`]
/// long, the size the checker caps.
///
/// Views of two masters each reach their own master's bytes of the one cell, which spans the
/// larger (as the product plans them). The cells are the bytes the Metal oracle's pass holds, which
/// its initializer writes and its resolver reads through this master; each view the product plans
/// reads and writes them through its own master, as Metal's does. Two members of one semantic, and
/// a cell past `u32` bytes, are refused.
pub fn layout(reflected: &FragmentImageblock) -> Result<Layout, String> {
    let mut members = BTreeMap::new();
    for member in &reflected.members {
        let at = Member {
            offset: member.offset,
            size: member.size,
        };
        if members.insert(member.semantic.clone(), at).is_some() {
            return Err(format!(
                "two imageblock master members are named {}",
                member.semantic
            ));
        }
    }
    let cell_bytes = u32::try_from(crate::check::caps::fragment_imageblock_cell_bytes(
        reflected,
    ))
    .map_err(|_| "an imageblock cell past u32 bytes")?;
    Ok(Layout {
        members,
        cell_bytes,
    })
}

/// Every pixel's cell of a pass's imageblock: `dimensions` pixels in rows, each `cell_bytes`
/// bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cells {
    pub dimensions: [u32; 2],
    pub cell_bytes: u32,
    pub bytes: Vec<u8>,
}

impl Cells {
    /// The cells the Metal oracle's initializer writes before the case's draw: each authored
    /// plane at its master member's offset in every pixel's cell, every other byte zero.
    pub fn initial(layout: &Layout, authored: &LiteralFragmentImageblock) -> Result<Cells, String> {
        let [width, height] = authored.dimensions;
        let pixels = width as usize * height as usize;
        let cell = layout.cell_bytes as usize;
        // The checker caps the cells with the case's resources
        // (`caps::fragment_imageblock_cap_errors`); this bounds an unchecked caller too.
        let length = pixels
            .checked_mul(cell)
            .filter(|&length| length as u64 <= crate::check::caps::MAX_RESOURCE_BYTES)
            .ok_or_else(|| {
                format!("{width}x{height} imageblock cells of {cell} bytes exceed the resource cap")
            })?;
        let mut bytes = vec![0; length];
        for plane in &authored.members {
            let at = layout.members.get(&plane.semantic).ok_or_else(|| {
                format!(
                    "authored imageblock plane {} names no member of the master",
                    plane.semantic
                )
            })?;
            let size = plane.format.byte_size();
            if size != at.size as usize {
                return Err(format!(
                    "imageblock plane {} is {size}-byte {:?}, its master member {} bytes",
                    plane.semantic, plane.format, at.size
                ));
            }
            if plane.bytes.len() != pixels * size {
                return Err(format!(
                    "imageblock plane {} holds {} bytes, not {width}x{height} of {size}",
                    plane.semantic,
                    plane.bytes.len()
                ));
            }
            let offset = at.offset as usize;
            for (pixel, value) in plane.bytes.chunks_exact(size).enumerate() {
                let start = pixel * cell + offset;
                bytes[start..start + size].copy_from_slice(value);
            }
        }
        Ok(Cells {
            dimensions: authored.dimensions,
            cell_bytes: layout.cell_bytes,
            bytes,
        })
    }

    fn pixels(&self) -> usize {
        self.dimensions[0] as usize * self.dimensions[1] as usize
    }

    /// The 32-bit words of one cell.
    pub fn words(&self) -> u32 {
        self.cell_bytes / 4
    }

    /// The layers of the pass's storage image for a pipeline whose imageblock spans
    /// `pipeline_words` words (at least that many): these cells' words. A pipeline whose words
    /// exceed them plans its views by a larger master than the reflected one the cells follow,
    /// which this executor does not model.
    pub fn storage_layers(&self, pipeline_words: u32) -> Result<u32, String> {
        if pipeline_words > self.words() {
            return Err(format!(
                "a pipeline imageblock of {pipeline_words} words over {}-word reflected cells",
                self.words()
            ));
        }
        Ok(self.words())
    }

    /// The contents of a `layers`-layer `R32Uint` image of these cells, tightly packed layer by
    /// layer (the copy a `VkBufferImageCopy` of every layer reads): layer `w` holds word `w` of
    /// each pixel's cell, and a layer beyond the cell's words zero.
    pub fn image_bytes(&self, layers: u32) -> Vec<u8> {
        let pixels = self.pixels();
        let cell = self.cell_bytes as usize;
        let mut image = vec![0; layers as usize * pixels * 4];
        for word in 0..(layers.min(self.words()) as usize) {
            for pixel in 0..pixels {
                let texel = (word * pixels + pixel) * 4;
                let from = pixel * cell + word * 4;
                image[texel..texel + 4].copy_from_slice(&self.bytes[from..from + 4]);
            }
        }
        image
    }

    /// The cells a `layers`-layer image of [`Cells::image_bytes`]'s layout holds: these cells'
    /// extent and cell size, each word read from its layer.
    pub fn from_image(&self, layers: u32, image: &[u8]) -> Result<Cells, String> {
        let pixels = self.pixels();
        if layers < self.words() || image.len() != layers as usize * pixels * 4 {
            return Err(format!(
                "an imageblock image of {} bytes in {layers} layers does not hold {pixels} cells \
                 of {} words",
                image.len(),
                self.words()
            ));
        }
        let cell = self.cell_bytes as usize;
        let mut bytes = vec![0; pixels * cell];
        for word in 0..self.words() as usize {
            for pixel in 0..pixels {
                let texel = (word * pixels + pixel) * 4;
                let to = pixel * cell + word * 4;
                bytes[to..to + 4].copy_from_slice(&image[texel..texel + 4]);
            }
        }
        Ok(Cells {
            dimensions: self.dimensions,
            cell_bytes: self.cell_bytes,
            bytes,
        })
    }

    /// The `(origin, dimensions)` region of `member`'s plane, rows in order, tightly packed: what
    /// the Metal oracle's resolver writes and its readback selects.
    pub fn select(
        &self,
        member: Member,
        origin: [u32; 2],
        dimensions: [u32; 2],
    ) -> Result<Vec<u8>, String> {
        let cell = self.cell_bytes as usize;
        let (offset, size) = (member.offset as usize, member.size as usize);
        if offset + size > cell {
            return Err(format!(
                "imageblock member at {offset}+{size} exceeds its {cell}-byte cell"
            ));
        }
        let plane: Vec<u8> = self
            .bytes
            .chunks_exact(cell)
            .flat_map(|bytes| &bytes[offset..offset + size])
            .copied()
            .collect();
        crate::literal::select_tightly_packed_2d(&plane, self.dimensions, origin, dimensions, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{FragmentImageblockFormat, ResourceRole};
    use crate::literal::LiteralFragmentImageblockMember;

    fn plane(
        semantic: &str,
        format: FragmentImageblockFormat,
        bytes: Vec<u8>,
    ) -> LiteralFragmentImageblockMember {
        LiteralFragmentImageblockMember {
            semantic: semantic.into(),
            format,
            role: ResourceRole::InOut,
            bytes,
        }
    }

    /// A master of `a` (half at 0), `b` (half at 2, the high half of word 0), `tag` (uchar4 at 4),
    /// and `n` (ushort at 8) in a 12-byte cell, over 2x1 pixels.
    fn master() -> Layout {
        Layout {
            members: BTreeMap::from([
                ("a".into(), Member { offset: 0, size: 2 }),
                ("b".into(), Member { offset: 2, size: 2 }),
                ("tag".into(), Member { offset: 4, size: 4 }),
                ("n".into(), Member { offset: 8, size: 2 }),
            ]),
            cell_bytes: 12,
        }
    }

    /// Each plane lands at its member's offset in every cell, little-endian within the words, an
    /// unauthored member and the padding are zero, layer `w` of the image is word `w` of each
    /// pixel's cell (a layer past the cell zero), the image reads back to the same cells, and a
    /// member's plane selects exactly its bytes.
    #[test]
    fn cells_lay_planes_out_at_master_offsets_one_word_per_layer() {
        use FragmentImageblockFormat as F;
        let authored = LiteralFragmentImageblock {
            dimensions: [2, 1],
            members: vec![
                plane("b", F::Half, vec![0x11, 0x22, 0x33, 0x44]),
                plane("tag", F::Uchar4, vec![1, 2, 3, 4, 5, 6, 7, 8]),
                plane("n", F::Ushort, vec![0xaa, 0xbb, 0xcc, 0xdd]),
            ],
        };
        let cells = Cells::initial(&master(), &authored).unwrap();
        assert_eq!(cells.words(), 3);
        assert_eq!(
            cells.bytes,
            [
                [0, 0, 0x11, 0x22, 1, 2, 3, 4, 0xaa, 0xbb, 0, 0],
                [0, 0, 0x33, 0x44, 5, 6, 7, 8, 0xcc, 0xdd, 0, 0],
            ]
            .concat()
        );
        let image = cells.image_bytes(4);
        assert_eq!(
            image,
            [
                [0, 0, 0x11, 0x22, 0, 0, 0x33, 0x44],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0xaa, 0xbb, 0, 0, 0xcc, 0xdd, 0, 0],
                [0; 8],
            ]
            .concat()
        );
        assert_eq!(cells.from_image(4, &image).unwrap(), cells);
        assert!(cells.from_image(2, &image[..16]).is_err());
        assert_eq!(
            cells
                .select(Member { offset: 2, size: 2 }, [0, 0], [2, 1])
                .unwrap(),
            [0x11, 0x22, 0x33, 0x44]
        );
        assert_eq!(
            cells
                .select(Member { offset: 4, size: 4 }, [1, 0], [1, 1])
                .unwrap(),
            [5, 6, 7, 8]
        );
        assert!(cells
            .select(
                Member {
                    offset: 10,
                    size: 4
                },
                [0, 0],
                [1, 1]
            )
            .is_err());
        assert!(cells
            .select(Member { offset: 0, size: 2 }, [1, 0], [2, 1])
            .is_err());
    }

    /// The storage image has the cells' words, which must cover the pipeline's; a pipeline spanning
    /// more words than the reflected cells plans its views by another master and is refused. Cells
    /// past the resource cap are refused before they are allocated.
    #[test]
    fn the_storage_has_the_cells_words_and_the_cells_are_capped() {
        let cells = Cells::initial(
            &master(),
            &LiteralFragmentImageblock {
                dimensions: [2, 1],
                members: vec![],
            },
        )
        .unwrap();
        assert_eq!(cells.storage_layers(1), Ok(3));
        assert_eq!(cells.storage_layers(3), Ok(3));
        assert!(cells.storage_layers(4).unwrap_err().contains("4 words"));
        let huge = Layout {
            members: BTreeMap::new(),
            cell_bytes: 1 << 12,
        };
        let refused = Cells::initial(
            &huge,
            &LiteralFragmentImageblock {
                dimensions: [1024, 1024],
                members: vec![],
            },
        )
        .unwrap_err();
        assert!(refused.contains("exceed the resource cap"), "{refused}");
    }

    /// A plane that names no master member, has another member's size, or holds the wrong number
    /// of bytes is refused.
    #[test]
    fn a_plane_must_be_a_master_member_of_its_size() {
        use FragmentImageblockFormat as F;
        let refused = |member| {
            Cells::initial(
                &master(),
                &LiteralFragmentImageblock {
                    dimensions: [2, 1],
                    members: vec![member],
                },
            )
            .unwrap_err()
        };
        assert!(refused(plane("c", F::Half, vec![0; 4])).contains("names no member"));
        assert!(refused(plane("tag", F::Half, vec![0; 4])).contains("master member 4 bytes"));
        assert!(refused(plane("a", F::Half, vec![0; 2])).contains("holds 2 bytes"));
    }
}
