//! Leaving a game releases most of its resources at once. Thousands of textures and buffers, some
//! with work still in flight, are dropped together while another thread reads
//! `Device::memory_statistics` as the performance trace does; the test prints how long the drop took
//! and how long that reader was kept waiting.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{LoadAction, RenderPassDescriptor, ResourceOptions, StoreAction, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
#[ignore = "makes thousands of resources and prints timings; run it on purpose"]
fn dropping_a_games_worth_of_resources_does_not_hold_the_device_for_long() {
    let scratch = Scratch::new("release-storm");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let textures_wanted = 3000;
    let mut textures = Vec::new();
    for _ in 0..textures_wanted {
        let mut desc = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 256, 256, true);
        desc.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        desc.resource_options = ResourceOptions::private();
        textures.push(device.new_texture(&desc).unwrap());
    }
    let mut buffers = Vec::new();
    for index in 0..20_000u64 {
        let length = 256 << (index % 8);
        let options = if index % 5 == 0 {
            ResourceOptions::private()
        } else {
            ResourceOptions::shared()
        };
        buffers.push(device.new_buffer(length, options).unwrap());
    }
    eprintln!(
        "STORM: {} textures, {} buffers; {} MiB live, {} MiB device-local",
        textures.len(),
        buffers.len(),
        device.memory_statistics().live_bytes >> 20,
        device.memory_statistics().device_local_live_bytes >> 20
    );
    // Work in flight that uses a third of the textures when they are dropped.
    let mut commands = Vec::new();
    for chunk in textures.chunks(250).take(4) {
        let command = queue.command_buffer().unwrap();
        for texture in chunk {
            let mut pass = RenderPassDescriptor::new();
            pass.color_attachments[0].texture = Some(texture.clone());
            pass.color_attachments[0].load_action = LoadAction::Clear;
            pass.color_attachments[0].store_action = StoreAction::Store;
            command
                .render_command_encoder(&pass)
                .unwrap()
                .end_encoding()
                .unwrap();
        }
        command.commit().unwrap();
        commands.push(command);
    }
    // The reader: what the performance trace does once a second, here every 5 ms.
    let stop = Arc::new(AtomicBool::new(false));
    let worst_wait = Arc::new(AtomicU64::new(0));
    let reader = {
        let (device, stop, worst) = (device.clone(), stop.clone(), worst_wait.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let start = Instant::now();
                let _ = device.memory_statistics();
                worst.fetch_max(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(5));
            }
        })
    };
    let start = Instant::now();
    let mut slowest = Duration::ZERO;
    let mut drop_each = |count: usize, what: &str, all: &mut dyn FnMut() -> Option<()>| {
        let begin = Instant::now();
        for _ in 0..count {
            let one = Instant::now();
            if all().is_none() {
                break;
            }
            slowest = slowest.max(one.elapsed());
        }
        eprintln!(
            "STORM: dropped {what} in {:.0} ms",
            begin.elapsed().as_secs_f64() * 1000.0
        );
    };
    drop_each(textures.len(), "the textures", &mut || {
        textures.pop().map(drop)
    });
    drop_each(buffers.len(), "the buffers", &mut || {
        buffers.pop().map(drop)
    });
    let dropped = start.elapsed();
    for command in &commands {
        command.wait_until_completed_for(WAIT).unwrap();
    }
    drop(commands);
    drop(queue);
    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();
    eprintln!(
        "STORM: everything dropped in {:.0} ms, slowest single drop {:.1} ms, the statistics reader waited at most {:.1} ms; {} MiB live after",
        dropped.as_secs_f64() * 1000.0,
        slowest.as_secs_f64() * 1000.0,
        worst_wait.load(Ordering::Relaxed) as f64 / 1000.0,
        device.memory_statistics().live_bytes >> 20
    );
}
