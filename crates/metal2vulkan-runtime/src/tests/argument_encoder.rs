//! The argument encoder's pure parts: where `setArgumentBuffer` puts the arguments, which argument
//! a setter writes, and the word `setBuffer` writes. Expected values and refusals are what Metal's
//! encoder and its API validation answered (probes, an M3 Max and an M4).

use super::super::argument_encoder::{argument_place, buffer_word, constant_offset, target_offset};
use super::super::{ArgumentEncoderProblem, Problem};
use metal2vulkan::bind::{ArgumentBufferLayout, ArgumentDescriptor, ArgumentType as T};
use metal2vulkan::metal::DataType as D;

fn layout(descriptors: &[(D, u32, u32)]) -> ArgumentBufferLayout {
    let descriptors: Vec<ArgumentDescriptor> = descriptors
        .iter()
        .map(|&(data_type, index, length)| {
            let mut d = ArgumentDescriptor::new(data_type, index);
            d.array_length = length;
            d
        })
        .collect();
    ArgumentBufferLayout::from_descriptors(&descriptors).expect("a layout")
}

/// A layout of 256 bytes, as probe 1's `A`: a pointer, a texture, and 30 floats.
fn a256() -> ArgumentBufferLayout {
    let l = layout(&[(D::Pointer, 0, 0), (D::Texture, 1, 0), (D::Float4, 2, 15)]);
    assert_eq!(l.encoded_length, 256);
    l
}

#[test]
fn arguments_start_at_the_offset_plus_whole_encoded_lengths() {
    let l = a256();
    // probe 1: offset 256 and (0, element 1) both put texture 1 at 264; (64, element 2) at 584.
    assert_eq!(target_offset(&l, 8192, 256, 0), Ok(256));
    assert_eq!(target_offset(&l, 8192, 0, 1), Ok(256));
    assert_eq!(target_offset(&l, 8192, 64, 2), Ok(576));
    // probe 3: an offset Metal does not align (start 8, element 1) is taken as given.
    assert_eq!(target_offset(&l, 8192, 8, 1), Ok(264));
    // The arguments may end at the buffer's end.
    assert_eq!(target_offset(&l, 8192, 8192 - 256, 0), Ok(8192 - 256));
}

#[test]
fn arguments_past_the_buffer_are_refused() {
    let l = a256();
    let refused = |offset, length| {
        Err(Problem::RangeOutOfBounds {
            offset,
            size: 256,
            length,
        })
    };
    // probe 3: "offset (8064) + encodedLength (256) should be smaller or equal to the buffer
    // length (8192)", and the same for element 40 and for a buffer of 64 bytes.
    assert_eq!(target_offset(&l, 8192, 8192 - 128, 0), refused(8064, 8192));
    assert_eq!(target_offset(&l, 8192, 0, 40), refused(10240, 8192));
    assert_eq!(target_offset(&l, 64, 0, 0), refused(0, 64));
    assert_eq!(
        target_offset(&l, 8192, 0, u64::MAX),
        refused(u64::MAX, 8192)
    );
    assert_eq!(
        target_offset(&l, u64::MAX, u64::MAX, 0),
        refused(u64::MAX, u64::MAX)
    );
}

#[test]
fn a_setter_writes_only_an_argument_of_its_type() {
    let l = layout(&[(D::Pointer, 0, 0), (D::Texture, 1, 3), (D::Float, 4, 2)]);
    let texture = argument_place(&l, 3, T::Texture).expect("texture 3");
    assert_eq!((texture.offset, texture.size), (24, 8));
    let constant = argument_place(&l, 5, T::Constant).expect("constant 5");
    assert_eq!((constant.offset, constant.size), (36, 4));
    // probe 3: a texture at a buffer, a buffer at a texture or a constant, and indices no
    // argument has (in a gap, past the layout) are Metal validation's refusals.
    assert_eq!(
        argument_place(&l, 0, T::Texture),
        Err(ArgumentEncoderProblem::ArgumentType {
            index: 0,
            expected: T::Texture,
            found: T::Buffer
        })
    );
    assert_eq!(
        argument_place(&l, 4, T::Buffer),
        Err(ArgumentEncoderProblem::ArgumentType {
            index: 4,
            expected: T::Buffer,
            found: T::Constant
        })
    );
    for index in [6, 1000] {
        assert_eq!(
            argument_place(&l, index, T::Texture),
            Err(ArgumentEncoderProblem::NoArgument { index })
        );
    }
}

#[test]
fn a_buffer_is_written_as_its_address_plus_the_offset() {
    // probes 1 and 4: scratch+12, nil, and an offset one below the length.
    let address = 0x0000_0100_0001_8000;
    assert_eq!(buffer_word(Some((address, 4096)), 0), Ok(address));
    assert_eq!(buffer_word(Some((address, 4096)), 12), Ok(address + 12));
    assert_eq!(
        buffer_word(Some((address, 4096)), 4095),
        Ok(0x0000_0100_0001_8fff)
    );
    assert_eq!(buffer_word(None, 0), Ok(0));
    // No buffer writes 0 whatever the offset, as plain Metal writes it (where validation stops
    // "offset(16) must be 0 when buffer is nil").
    assert_eq!(buffer_word(None, 100_000), Ok(0));
    // probe 4's refusal: "offset (4096) should be smaller than the buffer length (4096)".
    assert_eq!(
        buffer_word(Some((address, 4096)), 4096),
        Err(Problem::OffsetOutOfRange {
            offset: 4096,
            length: 4096
        })
    );
}

#[test]
fn constant_data_answers_as_plain_metal() {
    // Measured on an M4, plain Metal: `constantDataAtIndex:` answers a constant's offset, the
    // arguments' start (offset 0) at an index no argument has (in a gap or past the layout), and
    // the other argument's own offset at an argument of another type (a texture at index 3 ->
    // its offset 32). `MTL_DEBUG_LAYER=1` asserts at the last two; the runtime follows plain
    // Metal.
    let probe = layout(&[(D::Float, 0, 0), (D::Float4, 1, 0), (D::Texture, 3, 0)]);
    assert_eq!(probe.encoded_length, 48);
    for (index, offset) in [(0, 0), (1, 16), (2, 0), (3, 32), (7, 0), (1000, 0)] {
        assert_eq!(constant_offset(&probe, index), offset, "index {index}");
    }
    let l = layout(&[(D::Pointer, 0, 0), (D::Texture, 1, 3), (D::Float, 4, 2)]);
    assert_eq!(constant_offset(&l, 5), 36);
    assert_eq!(constant_offset(&l, 2), 16);
    assert_eq!(constant_offset(&l, u32::MAX), 0);
}
