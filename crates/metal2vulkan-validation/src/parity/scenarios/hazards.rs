//! Scenario `hazards`: a word through a dispatch, a blit and a dispatch (`a` to `b` by the
//! [`COPY_WORD`] fixture, `b` to `c` by a copy, `c` to `d` by the fixture again),
//! ordered by Metal's hazard tracking within one command buffer, across three on one queue, and
//! across two queues by a shared event (the waiting command buffer committed first); and, with
//! untracked buffers, by fences. `d` must end as `a` began.

use super::super::api::{Api, Fixture, Observation};
use super::super::COPY_WORD;
use super::steps::{fixture, submit};
use metal2vulkan_runtime::{CpuCacheMode, HazardTrackingMode, ResourceOptions, Size, StorageMode};

/// The word `a` holds, and `d` before it is written.
const WORD: [u8; 4] = [0x67, 0x45, 0x23, 0x01];
const BEFORE: [u8; 4] = [0xab; 4];

pub(super) fn run<A: Api>(api: &A, case: &str, obs: &mut Observation) -> Result<(), String> {
    let order = match case {
        "one_command_buffer" => Order::OneCommandBuffer,
        "a_command_buffer_each" => Order::CommandBufferEach,
        "two_queues_and_an_event" => Order::TwoQueues,
        "untracked_with_fences" => Order::Fences,
        other => return Err(format!("hazards has no case {other}")),
    };
    chain(api, obs, &fixture(COPY_WORD)?, order)
}

/// How the chain's three steps are ordered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    OneCommandBuffer,
    CommandBufferEach,
    TwoQueues,
    Fences,
}

/// The chain's buffers: `a` and `d` `Shared`, `b` and `c` `Private`; untracked for fences.
struct Buffers<B> {
    a: B,
    b: B,
    c: B,
    d: B,
}

fn buffers<A: Api>(api: &A, obs: &mut Observation, order: Order) -> Option<Buffers<A::Buffer>> {
    let hazard = if order == Order::Fences {
        HazardTrackingMode::Untracked
    } else {
        HazardTrackingMode::Default
    };
    let options = |storage| ResourceOptions::new(storage, CpuCacheMode::DefaultCache, hazard);
    let a = obs.call("a", || api.buffer(4, options(StorageMode::Shared)))?;
    api.write(&a, &WORD);
    let b = obs.call("b", || api.buffer(4, options(StorageMode::Private)))?;
    let c = obs.call("c", || api.buffer(4, options(StorageMode::Private)))?;
    let d = obs.call("d", || api.buffer(4, options(StorageMode::Shared)))?;
    api.write(&d, &BEFORE);
    Some(Buffers { a, b, c, d })
}

/// The first dispatch, `a` to `b`, in its own encoder of `cb`, then updating `fence`.
fn first<A: Api>(
    api: &A,
    obs: &mut Observation,
    cb: &A::CommandBuffer,
    pipeline: &A::Pipeline,
    bufs: &Buffers<A::Buffer>,
    fence: Option<&A::Fence>,
) -> Option<()> {
    let mut compute = obs.call("first encoder", || api.compute(cb))?;
    obs.call("a to b", || {
        api.dispatch(
            &mut compute,
            pipeline,
            &[(&bufs.a, 0), (&bufs.b, 1)],
            Size::new(1, 1, 1),
        )
    })?;
    if let Some(fence) = fence {
        obs.call("first updates its fence", || {
            api.compute_update_fence(&mut compute, fence)
        })?;
    }
    obs.call("end first", || api.end_compute(compute))
}

/// The copy, `b` to `c`, in its own encoder of `cb`, after `wait` and before `update`.
fn copy<A: Api>(
    api: &A,
    obs: &mut Observation,
    cb: &A::CommandBuffer,
    bufs: &Buffers<A::Buffer>,
    fences: Option<(&A::Fence, &A::Fence)>,
) -> Option<()> {
    let mut blit = obs.call("blit encoder", || api.blit(cb))?;
    if let Some((wait, _)) = fences {
        obs.call("blit waits", || api.blit_wait_fence(&mut blit, wait))?;
    }
    obs.call("b to c", || {
        api.copy_buffer(&mut blit, (&bufs.b, 0), (&bufs.c, 0), 4)
    })?;
    if let Some((_, update)) = fences {
        obs.call("blit updates", || api.blit_update_fence(&mut blit, update))?;
    }
    obs.call("end blit", || api.end_blit(blit))
}

/// The second dispatch, `c` to `d`, in its own encoder of `cb`, after `fence`.
fn second<A: Api>(
    api: &A,
    obs: &mut Observation,
    cb: &A::CommandBuffer,
    pipeline: &A::Pipeline,
    bufs: &Buffers<A::Buffer>,
    fence: Option<&A::Fence>,
) -> Option<()> {
    let mut compute = obs.call("second encoder", || api.compute(cb))?;
    if let Some(fence) = fence {
        obs.call("second waits", || {
            api.compute_wait_fence(&mut compute, fence)
        })?;
    }
    obs.call("c to d", || {
        api.dispatch(
            &mut compute,
            pipeline,
            &[(&bufs.c, 0), (&bufs.d, 1)],
            Size::new(1, 1, 1),
        )
    })?;
    obs.call("end second", || api.end_compute(compute))
}

fn chain<A: Api>(
    api: &A,
    obs: &mut Observation,
    fixture: &Fixture,
    order: Order,
) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(pipeline) = obs.call("pipeline", || api.pipeline(fixture)) else {
        return Ok(());
    };
    let Some(bufs) = buffers(api, obs, order) else {
        return Ok(());
    };
    let done = match order {
        Order::OneCommandBuffer | Order::Fences => {
            one_command_buffer(api, obs, &queue, &pipeline, &bufs, order)?
        }
        Order::CommandBufferEach => command_buffer_each(api, obs, &queue, &pipeline, &bufs)?,
        Order::TwoQueues => two_queues(api, obs, &queue, &pipeline, &bufs)?,
    };
    if done.is_some() {
        obs.bytes("bytes of d", &api.read(&bufs.d));
    }
    Ok(())
}

fn one_command_buffer<A: Api>(
    api: &A,
    obs: &mut Observation,
    queue: &A::Queue,
    pipeline: &A::Pipeline,
    bufs: &Buffers<A::Buffer>,
    order: Order,
) -> Result<Option<()>, String> {
    let fences = if order == Order::Fences {
        let Some(first_fence) = obs.call("first fence", || api.fence()) else {
            return Ok(None);
        };
        let Some(second_fence) = obs.call("second fence", || api.fence()) else {
            return Ok(None);
        };
        Some((first_fence, second_fence))
    } else {
        None
    };
    let fences = fences.as_ref().map(|(first, second)| (first, second));
    let Some(cb) = obs.call("command buffer", || api.command_buffer(queue)) else {
        return Ok(None);
    };
    if first(api, obs, &cb, pipeline, bufs, fences.map(|(f, _)| f)).is_none()
        || copy(api, obs, &cb, bufs, fences).is_none()
        || second(api, obs, &cb, pipeline, bufs, fences.map(|(_, s)| s)).is_none()
    {
        return Ok(None);
    }
    submit(api, obs, "commit", cb)
}

fn command_buffer_each<A: Api>(
    api: &A,
    obs: &mut Observation,
    queue: &A::Queue,
    pipeline: &A::Pipeline,
    bufs: &Buffers<A::Buffer>,
) -> Result<Option<()>, String> {
    let mut cbs = Vec::new();
    for name in [
        "first command buffer",
        "copy command buffer",
        "second command buffer",
    ] {
        let Some(cb) = obs.call(name, || api.command_buffer(queue)) else {
            return Ok(None);
        };
        cbs.push(cb);
    }
    if first(api, obs, &cbs[0], pipeline, bufs, None).is_none()
        || copy(api, obs, &cbs[1], bufs, None).is_none()
        || second(api, obs, &cbs[2], pipeline, bufs, None).is_none()
    {
        return Ok(None);
    }
    for (index, cb) in cbs.iter().enumerate() {
        if obs
            .call(&format!("commit {index}"), || api.commit(cb))
            .is_none()
        {
            return Ok(None);
        }
    }
    for (index, cb) in cbs.into_iter().enumerate() {
        let completion = api.completion(cb)?;
        obs.completed(&format!("completion {index}"), completion);
    }
    Ok(Some(()))
}

fn two_queues<A: Api>(
    api: &A,
    obs: &mut Observation,
    queue: &A::Queue,
    pipeline: &A::Pipeline,
    bufs: &Buffers<A::Buffer>,
) -> Result<Option<()>, String> {
    let Some(other) = obs.call("other queue", || api.queue()) else {
        return Ok(None);
    };
    let Some(event) = obs.call("event", || api.event()) else {
        return Ok(None);
    };
    let Some(producer) = obs.call("producer", || api.command_buffer(queue)) else {
        return Ok(None);
    };
    let Some(consumer) = obs.call("consumer", || api.command_buffer(&other)) else {
        return Ok(None);
    };
    if first(api, obs, &producer, pipeline, bufs, None).is_none()
        || copy(api, obs, &producer, bufs, None).is_none()
        || obs
            .call("producer signals 1", || api.signal(&producer, &event, 1))
            .is_none()
        || obs
            .call("consumer waits for 1", || api.wait(&consumer, &event, 1))
            .is_none()
        || second(api, obs, &consumer, pipeline, bufs, None).is_none()
        || obs
            .call("commit consumer", || api.commit(&consumer))
            .is_none()
        || obs
            .call("commit producer", || api.commit(&producer))
            .is_none()
    {
        return Ok(None);
    }
    let completion = api.completion(producer)?;
    obs.completed("producer completion", completion);
    let completion = api.completion(consumer)?;
    obs.completed("consumer completion", completion);
    Ok(Some(()))
}
