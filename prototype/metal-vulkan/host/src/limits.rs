//! Operational compile limits and device-local memory budgets.
use metal2vulkan_runtime::Device;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Once;

const COMPILE_BYTES: u64 = 512 << 20;

pub fn turns(threads: usize, available: Option<u64>) -> usize {
    let cpu = if threads >= 16 {
        12
    } else {
        (threads / 2).clamp(1, 12)
    };
    available.map_or(cpu, |bytes| {
        let reserve = (512 << 20).max(bytes / 4);
        let memory = bytes.saturating_sub(reserve) / COMPILE_BYTES;
        cpu.min(memory.max(1) as usize)
    })
}

fn cgroup_available(root: &Path, group: &Path) -> Option<u64> {
    let mut group = group.to_path_buf();
    let mut available = None;
    loop {
        let directory = root.join(&group);
        for limit in ["memory.max", "memory.high"] {
            let maximum = std::fs::read_to_string(directory.join(limit))
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok());
            let current = std::fs::read_to_string(directory.join("memory.current"))
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok());
            if let (Some(maximum), Some(current)) = (maximum, current) {
                let free = maximum.saturating_sub(current);
                available = Some(available.map_or(free, |old: u64| old.min(free)));
            }
        }
        if !group.pop() {
            break;
        }
    }
    available
}

pub fn available_memory() -> Option<u64> {
    let host = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let value = line
                    .strip_prefix("MemAvailable:")?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?;
                value.checked_mul(1024)
            })
        });
    let group = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("0::/").map(std::path::PathBuf::from))
        });
    let group = group.and_then(|group| cgroup_available(Path::new("/sys/fs/cgroup"), &group));
    match (host, group) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

pub fn configure(tuning: &crate::tuning::Tuning) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let threads = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let available = available_memory();
        let count = tuning.compile_threads.unwrap_or_else(|| turns(threads, available));
        if metal2vulkan_runtime::set_pipeline_compile_budget(NonZeroUsize::new(count).unwrap()) {
            crate::util::log_once(&format!("[Quadra Metal] pipeline compilation cap: {count}, available host/cgroup memory: {} MiB", available.unwrap_or(0) >> 20));
        }
    });
}

/// One memory heap as the budget policy sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapBudget {
    pub size: u64,
    /// `VK_EXT_memory_budget`'s budget for the heap, 0 where the extension is absent.
    pub budget: u64,
    pub device_local: bool,
}

impl HeapBudget {
    /// What a process may use of the heap: the driver's budget when it gives one, else three
    /// quarters of the heap.
    fn usable(&self) -> u64 {
        if self.budget > 0 {
            self.budget.min(self.size)
        } else {
            self.size.saturating_mul(3) / 4
        }
    }
}

/// `recommendedMaxWorkingSetSize` from a device's heaps.
///
/// A discrete GPU's device-local heaps overlap: without Resizable BAR the driver lists the
/// host-visible window onto VRAM (256 MiB) as a heap beside the full one, so only the largest
/// counts. A unified-memory GPU shares the host's memory, and a driver may list it as several
/// heaps (an APU's small VRAM carve-out beside the larger system-memory heap the GPU also maps),
/// so all of them count.
pub fn working_set(heaps: &[HeapBudget], unified: bool) -> u64 {
    let total = if unified {
        heaps
            .iter()
            .map(HeapBudget::usable)
            .fold(0, u64::saturating_add)
    } else {
        heaps
            .iter()
            .filter(|heap| heap.device_local)
            .map(HeapBudget::usable)
            .max()
            .unwrap_or(0)
    };
    if total > 0 {
        total
    } else {
        512 << 20
    }
}

/// The user's cap on the answer (`MACOBLOX_VRAM_BYTES`), which can only lower it.
pub fn capped(measured: u64, cap: Option<&str>) -> u64 {
    cap.and_then(|value| value.parse::<u64>().ok())
        .filter(|&bytes| bytes > 0)
        .map_or(measured, |value| value.min(measured))
}

pub fn recommended_working_set(device: &Device) -> u64 {
    let raw = unsafe { device.raw() };
    let supported = unsafe {
        raw.instance
            .enumerate_device_extension_properties(raw.physical_device)
    }
    .unwrap_or_default();
    let has_budget = supported.iter().any(|extension| unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) } == ash::ext::memory_budget::NAME);
    let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = ash::vk::PhysicalDeviceMemoryProperties2::default();
    if has_budget {
        properties = properties.push_next(&mut budget);
    }
    unsafe {
        raw.instance
            .get_physical_device_memory_properties2(raw.physical_device, &mut properties);
    }
    let memory = properties.memory_properties;
    let heaps: Vec<HeapBudget> = memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter()
        .enumerate()
        .map(|(index, heap)| HeapBudget {
            size: heap.size,
            budget: if has_budget {
                budget.heap_budget[index]
            } else {
                0
            },
            device_local: heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL),
        })
        .collect();
    let measured = working_set(&heaps, device.properties().has_unified_memory);
    capped(
        measured,
        std::env::var("MACOBLOX_VRAM_BYTES").ok().as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn heap(size: u64, budget: u64, device_local: bool) -> HeapBudget {
        HeapBudget {
            size,
            budget,
            device_local,
        }
    }

    #[test]
    fn a_discrete_card_counts_its_largest_device_local_heap_once() {
        // The RTX 3060 Ti with Resizable BAR (vulkaninfo, 615.78.08): one heap of VRAM.
        let nvidia = [
            heap(8 * GIB, 7_389_708_288, true),
            heap(23 * GIB, 23 * GIB, false),
        ];
        assert_eq!(working_set(&nvidia, false), 7_389_708_288);
        // Without Resizable BAR the host-visible window onto VRAM is a heap of its own.
        let window = [
            heap(16 * GIB, 15 * GIB, true),
            heap(256 * MIB, 256 * MIB, true),
            heap(30 * GIB, 28 * GIB, false),
        ];
        assert_eq!(working_set(&window, false), 15 * GIB);
        // Without the budget extension: three quarters of the heap.
        let plain = [heap(8 * GIB, 0, true), heap(246 * MIB, 0, true)];
        assert_eq!(working_set(&plain, false), 6 * GIB);
    }

    #[test]
    fn a_unified_device_counts_every_heap_it_can_use() {
        // An integrated Intel GPU: its one heap is the system's memory.
        assert_eq!(
            working_set(&[heap(31 * GIB, 22 * GIB, true)], true),
            22 * GIB
        );
        // An APU: a small carve-out of VRAM beside the system memory it also maps.
        let apu = [
            heap(512 * MIB, 512 * MIB, true),
            heap(14 * GIB, 12 * GIB, false),
        ];
        assert_eq!(working_set(&apu, true), 12 * GIB + 512 * MIB);
    }

    #[test]
    fn no_heap_or_budget_keeps_a_floor_and_a_cap_only_lowers() {
        assert_eq!(working_set(&[], false), 512 * MIB);
        assert_eq!(working_set(&[heap(8 * GIB, 9 * GIB, true)], false), 8 * GIB);
        assert_eq!(capped(6 * GIB, Some("4294967296")), 4 * GIB);
        assert_eq!(
            capped(6 * GIB, Some("17179869184")),
            6 * GIB,
            "a cap never raises"
        );
        assert_eq!(capped(6 * GIB, Some("0")), 6 * GIB);
        assert_eq!(capped(6 * GIB, Some("lots")), 6 * GIB);
        assert_eq!(capped(6 * GIB, None), 6 * GIB);
    }

    #[test]
    fn compile_turns_follow_both_cpu_and_memory_pressure() {
        assert_eq!(turns(1, Some(64 << 30)), 1);
        assert_eq!(turns(32, Some(0)), 1);
        assert_eq!(turns(32, Some(2 << 30)), 3);
        assert_eq!(turns(32, Some(16 << 30)), 12);
        assert_eq!(turns(4, None), 2);
    }

    #[test]
    fn cgroup_parent_limits_are_not_ignored() {
        let root = std::env::temp_dir().join(format!("quadra-cgroup-test-{}", std::process::id()));
        std::fs::create_dir_all(root.join("child")).unwrap();
        for (name, value) in [
            ("memory.max", "1000"),
            ("memory.current", "900"),
            ("child/memory.max", "max"),
            ("child/memory.current", "10"),
        ] {
            std::fs::write(root.join(name), value).unwrap();
        }
        assert_eq!(cgroup_available(&root, Path::new("child")), Some(100));
        std::fs::remove_dir_all(root).unwrap();
    }
}
