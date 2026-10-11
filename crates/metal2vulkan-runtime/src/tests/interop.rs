//! Interop with the caller's Vulkan work (`interop.rs`): what the runtime can tell of a semaphore
//! given as the caller's binary semaphore.

use crate::interop::semaphore_problem;
use crate::SemaphoreProblem;
use ash::vk;
use ash::vk::Handle;
use std::collections::HashSet;

fn semaphore(raw: u64) -> vk::Semaphore {
    vk::Semaphore::from_raw(raw)
}

#[test]
fn the_runtimes_own_timelines_and_null_are_refused() {
    let queues = [semaphore(0x10), semaphore(0x20)];
    let events: HashSet<vk::Semaphore> = [semaphore(0x30)].into_iter().collect();
    let problem = |raw| semaphore_problem(semaphore(raw), queues, &events);
    assert_eq!(problem(0), Some(SemaphoreProblem::Null));
    assert_eq!(problem(0x10), Some(SemaphoreProblem::Timeline));
    assert_eq!(problem(0x20), Some(SemaphoreProblem::Timeline));
    assert_eq!(problem(0x30), Some(SemaphoreProblem::Timeline));
    // Any other handle is the caller's to vouch for.
    assert_eq!(problem(0x40), None);
}
