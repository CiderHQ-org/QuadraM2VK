//! A placement heap's `new_texture_at_offset` that makes no texture at an offset inside the heap
//! names it in an informational message (`metal2vulkan-heap-texture-refused`): one off the size
//! query's alignment, and one at an aligned offset whose bytes run past the heap's end. One at the
//! heap's end, outside it, is not named, nor is one the heap makes.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::device::lock;
use crate::{
    DebugMessage, Device, DeviceKind, DeviceOptions, Error, HeapDescriptor, HeapType,
    MessageSeverity, MessageSink, ResourceOptions, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::{Arc, Mutex};

/// What the device's sink heard of the message: each one's severity and text.
type Heard = Arc<Mutex<Vec<(MessageSeverity, String)>>>;

/// The device the cfg selects, with a sink that keeps every `metal2vulkan-heap-texture-refused`
/// message.
fn listening() -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        if message.id_name == "metal2vulkan-heap-texture-refused" {
            lock(&kept).push((message.severity, message.text.clone()));
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

#[test]
fn a_placement_heap_names_a_texture_it_does_not_make_inside_it() {
    let (device, heard) = listening();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R16Float, 383, 152, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let answer = device.heap_texture_size_and_align(&d).unwrap();
    let hd = HeapDescriptor::new(
        HeapType::Placement,
        answer.size + answer.align,
        ResourceOptions::shared(),
    );
    let heap = device.new_heap(&hd).unwrap();
    let size = heap.size();
    // The last offset at the alignment inside the heap, whose bytes run past its end.
    let last = (size - 1) / answer.align * answer.align;
    assert!(last + answer.size > size, "{answer:?} in {size:#x}");
    // Made at the heap's start: nothing said.
    assert!(heap.new_texture_at_offset(&d, 0).is_ok());
    for offset in [1, last, size] {
        let Err(e) = heap.new_texture_at_offset(&d, offset) else {
            panic!("a texture at {offset:#x}");
        };
        assert!(matches!(e, Error::Nil(_)), "{offset:#x}: {e}");
    }
    let heard = lock(&heard).clone();
    assert_eq!(heard.len(), 2, "{heard:?}");
    for ((severity, text), (offset, against)) in heard.iter().zip([
        (
            1,
            "the offset not a multiple of it, the bytes inside the heap",
        ),
        (
            last,
            "the offset a multiple of it, the bytes not inside the heap",
        ),
    ]) {
        assert_eq!(*severity, MessageSeverity::Info, "{text}");
        let texture = "R16Float texture of 383x152x1, 1 levels, 1 slices, 1 samples";
        assert!(text.contains(texture), "{text}");
        let at = format!("at offset {offset:#x} of a placement heap of {size:#x} bytes");
        assert!(text.contains(&at), "{text}");
        let answered = format!(
            "the size query answers it {:#x} bytes at an alignment of {:#x} ({against})",
            answer.size, answer.align
        );
        assert!(text.contains(&answered), "{text}");
        let planned = format!(
            "the making planned {:#x} bytes at an alignment of {:#x} ({against}), its image of ",
            answer.size, answer.align
        );
        assert!(text.contains(&planned), "{text}");
    }
}
