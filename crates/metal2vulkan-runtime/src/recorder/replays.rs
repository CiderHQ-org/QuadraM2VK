    /// Held through each [`settle_all`](Self::settle_all), so one finds the replays another
    /// took off the queue replayed.
    settling: Mutex<()>,
            settling: Mutex::default(),
                let replayed = replays.drain(functions.get(), true);
    /// it is in, and for another thread's settle: what the release, rebind or detach of host
    /// pages that replays hold does first, so the replays let go of them before it
    /// (`HostPages::let_go_by_buffer`). Once the process exits, what ended is dropped
    /// unreplayed instead ([`Replays::replay_ended`]).
    pub(crate) fn settle_all(&self, device: &Vk) {
        let _settling = lock(&self.settling);
            replayed += replays.drain(device, true);
            replayed += replays.drain(device, false);
    /// ended waits for its replay: an encoder encoded again before its command buffer's first
    /// submission (`reencode.rs`). As at the commit ([`commit`](Self::commit)), the pages it held
    /// last are the answer, for the caller to let go of where it lets go of them without a
    /// recorder. `fill` runs under its lock, which it must not take again.
    pub(crate) fn hold(
        self: &Arc<Self>,
        device: &Vk,
        fill: impl FnOnce(&mut Vec<Arc<HostPages>>),
    ) -> Vec<HostPages> {
        self.settle(device, &mut state)
    /// which it must not take again.
    /// Where the buffer of some of those pages let go of them (released, rebound or detached
    /// while an encoder took them, on any thread), the commit would let go of them last without
    /// a recorder: what ended replays now, on this thread, with `device`, the device's
    /// functions. The pages it held last are the answer, for the caller to let go of where the
    /// commit lets go of them without a recorder. Nothing as a rule.
        let mut state = lock(&self.state);
        Self::fill(&mut state, fill);
        state.open = true;
        self.settle(device, &mut state)
    }

    /// What `state` holds once its command buffer committed (nothing before), under its lock:
    /// queued for the recorder where something ended waits, then each page's buffer's let-go
    /// read, after its hold was counted and the replays queued, so a let-go after the read finds
    /// both (`HostPages::let_go_by_buffer`). Where one came before, everything ended replays now,
    /// on this thread, with `device`; where nothing ended waits, what it holds goes now. The
    /// pages it held last are the answer.
    fn settle(self: &Arc<Self>, device: &Vk, state: &mut ReplayState) -> Vec<HostPages> {
        if !state.open {
            return last;
        }
        if !state.ended.is_empty() {
            if !state.queued {
            if state.held.iter().any(|pages| pages.buffer_let_go()) {
                let replayed = self.replay_ended(device, state);
                self.recorder
                    .by_submitter
                    .fetch_add(replayed, Ordering::Relaxed);
            }
        }
        if state.ended.is_empty() {
            self.let_go(state, Some(&mut last));
    /// Add what `fill` adds to what `state` holds, counted in each page's holds.
    fn fill(state: &mut ReplayState, fill: impl FnOnce(&mut Vec<Arc<HostPages>>)) {
    /// thread (`queued`, which it takes off), a submitting one, or one that settles, with
    fn drain(&self, device: &Vk, queued: bool) -> u64 {
        let replayed = self.replay_ended(device, &mut state);
        // Nothing ended waits now: what it held for the replays goes.
        if state.open {
            self.let_go(&mut state, None);
        }
        replayed
    }

    /// Replay every ended command buffer of `state` not yet replayed, in the order they ended,
    /// inside a section, with `device`, the device's functions: how many. Once the process
    /// exits, outside a section, they are dropped unreplayed instead, and their submission fails
    /// ([`resolve`]), as a submission does then.
    fn replay_ended(&self, device: &Vk, state: &mut ReplayState) -> u64 {
        let Some(_busy) = exit_guard::enter() else {
            state.ended.clear();
            return 0;
        };
            self.replay(device, state, index);
    /// Let go of what `state` holds, under its lock, running nothing of the program's. Where
    /// `last` is given (a commit's, [`settle`](Self::settle)), the pages held last go to it,
    /// for the caller to let go of. Elsewhere none is held last: the release, rebind or detach
    /// of pages held here replays first, and a commit after it replays itself
    /// (`HostPages::let_go_by_buffer`). Were one held last, it would go on the device's handler
    /// threads (`Device::elsewhere`), as its deallocator may run the program's code, which may
    /// not run on the recorder's thread nor under a submission's locks: announced there
    /// (`HeldByReplay`).
            let Some(pages) = HostPages::let_go_by_replay(pages) else {
    let replayed = replays.drain(device, false);
