/* Standalone Mach-O fixture: no shim, UI, timers or live-prefix changes. */
typedef unsigned long long u64;
typedef long long i64;
struct kevent64 { u64 ident; short filter; unsigned short flags; unsigned fflags; i64 data; u64 udata; u64 ext[2]; };
struct timespec { long seconds, nanoseconds; };
struct rlimit { u64 current, maximum; };
extern int kqueue(void), kevent64(int, const struct kevent64 *, int, struct kevent64 *, int, unsigned, const struct timespec *);
extern int pipe(int *), dup(int), dup2(int, int), close(int), open(const char *, int, ...);
extern long write(int, const void *, unsigned long), read(int, void *, unsigned long);
extern int getrlimit(int, struct rlimit *), clock_gettime(int, struct timespec *), printf(const char *, ...);
extern int setrlimit(int, const struct rlimit *);
extern int puts(const char *);
extern unsigned alarm(unsigned);
extern void exit(int);
_Static_assert(sizeof(struct kevent64) == 48, "Darwin kevent64 ABI");
static void require(int yes, const char *message) { if (!yes) { printf("FAIL %s\n", message); exit(1); } }
static void add_read(int queue, int descriptor) {
    struct kevent64 change = {(u64)descriptor, -1, 1|4, 0, 0, 0, {0,0}};
    require(kevent64(queue, &change, 1, 0, 0, 0, 0) == 0, "add read watch");
}
static int poll_read(int queue, int descriptor) {
    struct kevent64 event = {0}; struct timespec timeout={0,0};
    int count=kevent64(queue, 0, 0, &event, 1, 0, &timeout);
    require(count >= 0, "event poll");
    if (count) require(event.ident == (u64)descriptor && event.filter == -1 && !(event.flags & 0x4000), "read event identity");
    return count;
}
static u64 nanos(void) { struct timespec time; require(!clock_gettime(6, &time), "monotonic clock"); return (u64)time.seconds*1000000000ULL+(u64)time.nanoseconds; }
int main(int argc, char **argv) {
    (void)argv; alarm(8);
    int queue=kqueue(); require(queue >= 0, "kqueue");
    struct rlimit limit; require(!getrlimit(8, &limit), "NOFILE");
    if (argc > 1) {
        require(limit.maximum <= 2147483647ULL, "representable descriptor bound");
        /* The fixture changes only its own soft limit, preserving its hard
         * limit. Darling adds/subtracts one around Linux NOFILE. Its driver
         * already occupies Linux fd1023, so the final high fd cannot alias it.
         * Do not lower the hard limit or replace the reserved driver fd. */
        require(limit.maximum > 1023, "boundary safely above reserved driver fd");
        limit.current=limit.maximum;
        require(!setrlimit(8, &limit), "raise own soft NOFILE to existing hard limit");
        int alias=dup2(queue, (int)limit.maximum);
        require(alias == (int)limit.maximum, "inclusive final descriptor alias");
        int endpoints[2]; require(!pipe(endpoints), "boundary pipe");
        add_read(alias, endpoints[0]); require(write(endpoints[1], "x", 1) == 1, "boundary write");
        require(poll_read(queue, endpoints[0]) == 1, "inclusive descriptor alias callback");
        require(!close(endpoints[0]) && !close(endpoints[1]), "boundary watch cleanup");
        require(!close(alias) && !close(queue), "boundary alias cleanup");
        puts("Darling kqueue inclusive descriptor boundary passed"); return 0;
    }
    int alias=dup(queue); require(alias >= 0 && alias != queue, "kqueue alias");
    int first[2], second[2]; require(!pipe(first) && !pipe(second), "pipes");
    add_read(queue, first[0]); require(write(first[1], "x", 1) == 1, "pipe write");
    require(poll_read(alias, first[0]) == 1, "alias sees shared watch");
    int reused=first[0]; require(!close(reused), "watched descriptor close");
    require(dup2(second[0], reused) == reused, "descriptor reuse");
    require(write(second[1], "y", 1) == 1, "reused pipe write");
    require(poll_read(queue, reused) == 0, "closed watch removed before descriptor reuse");
    add_read(alias, reused); require(poll_read(queue, reused) == 1, "reused descriptor fresh watch");
    require(!close(alias), "alias close");
    require(poll_read(queue, reused) == 1, "remaining kqueue alias functional");
    char value; require(read(reused, &value, 1) == 1 && value == 'y', "remaining alias functional");
    require(!close(reused) && !close(second[0]) && !close(second[1]) && !close(first[1]), "pipe cleanup");
    const int count=100; u64 begin=nanos();
    for (int i=0;i<count;i++) { int fd=open("/dev/null", 0); require(fd>=0 && !close(fd), "ordinary fd cleanup"); }
    u64 elapsed=nanos()-begin;
    require(!close(queue), "final kqueue close");
    printf("Darling kqueue alias/watch/reuse passed; NOFILE=%llu; %d open/close=%0.3fms\n", limit.maximum, count, elapsed/1000000.0);
    return 0;
}
