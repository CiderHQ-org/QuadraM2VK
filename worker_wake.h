/* A full notification pipe already contains a wakeup. Never block the game
 * thread waiting for a worker that might be busy in Xlib or have exited. */
#ifndef MACOBLOX_WORKER_WAKE_H
#define MACOBLOX_WORKER_WAKE_H
int macoblox_wake_pipe(int descriptors[2]);
void macoblox_wake_worker(int descriptor);
#endif
