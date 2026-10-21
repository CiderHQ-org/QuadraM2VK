/* Optional isolated AppKit regression: background postEvent must wake a
 * waiting nextEvent call. No windows, network requests or credentials. */
typedef signed char BOOL;
typedef unsigned long long u64;
typedef struct objc_class *Class;
typedef struct objc_ivar *Ivar;
extern Class objc_getClass(const char *);
extern Ivar class_getInstanceVariable(Class, const char *);
extern long ivar_getOffset(Ivar);
extern int pthread_create(void **, const void *, void *(*)(void *), void *);
extern int pthread_join(void *, void **);
extern int printf(const char *, ...);
extern int fflush(void *);
extern int strcmp(const char *, const char *);
extern unsigned int alarm(unsigned int);
extern const void *CFRunLoopGetMain(void);
extern const void *kCFRunLoopCommonModes;
extern const void *CFRunLoopSourceCreate(const void *, long, void *);
extern void CFRunLoopAddSource(const void *, const void *, const void *);
extern void CFRunLoopRemoveSource(const void *, const void *, const void *);
extern void CFRunLoopSourceSignal(const void *);
extern void CFRunLoopWakeUp(const void *);
extern void CFRelease(const void *);
extern double CFAbsoluteTimeGetCurrent(void);
extern const void *CFRunLoopTimerCreate(const void *, double, double, unsigned long, long,
                                      void (*)(const void *, void *), void *);
extern void CFRunLoopAddTimer(const void *, const void *, const void *);
extern void CFRunLoopTimerInvalidate(const void *);
extern id NSDefaultRunLoopMode;
extern id NSModalPanelRunLoopMode;
extern id NSEventTrackingRunLoopMode;
typedef struct { double x, y; } Point;
@interface NSObject
+ (id)alloc;
- (id)init;
- (void)release;
@end
@interface NSAutoreleasePool : NSObject @end
@interface NSDate : NSObject
+ (id)date;
+ (id)dateWithTimeIntervalSinceNow:(double)seconds;
- (double)timeIntervalSinceNow;
@end
@interface NSEvent : NSObject
- (id)initWithType:(unsigned long)type location:(Point)point modifierFlags:(unsigned long)flags window:(id)window;
- (unsigned long)type;
@end
@interface NSApplication : NSObject
+ (id)sharedApplication;
- (void)finishLaunching;
- (void)postEvent:(id)event atStart:(BOOL)first;
- (id)nextEventMatchingMask:(u64)mask untilDate:(id)until inMode:(id)mode dequeue:(BOOL)dequeue;
@end
@interface NSDisplay : NSObject
+ (id)currentDisplay;
- (void)processPendingEvents;
- (id)nextEventMatchingMask:(u64)mask untilDate:(id)until inMode:(id)mode dequeue:(BOOL)dequeue;
@end
@interface NSMutableArray : NSObject
- (void)removeAllObjects;
@end
static NSApplication *app;
static const void *source;
static int control;
static u64 posted;
static u64 now_ns(void) {
    struct { long seconds, nanoseconds; } stamp;
    long result;
    __asm__ volatile("syscall" : "=a"(result) : "a"(228L), "D"(1L), "S"(&stamp)
                     : "rcx", "r11", "memory");
    return result ? 0 : stamp.seconds * 1000000000ULL + stamp.nanoseconds;
}
static void perform(void *unused) { (void)unused; }
static void post_test_event(void) {
    id event = [[NSEvent alloc] initWithType:15 location:(Point){0,0} modifierFlags:0 window:0];
    [app postEvent:event atStart:0]; [event release];
    __atomic_store_n(&posted, now_ns(), __ATOMIC_RELEASE);
}
static void timer_post(const void *timer, void *info) { (void)timer; (void)info; post_test_event(); }
static void *producer(void *unused) {
    (void)unused;
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    struct { long seconds, nanoseconds; } delay = {0, 200000000};
    long result;
    __asm__ volatile("syscall" : "=a"(result) : "a"(35L), "D"(&delay), "S"(0L)
                     : "rcx", "r11", "memory");
    (void)result;
    if (control != 3) post_test_event();
    if (control == 2) CFRunLoopSourceSignal(source);
    if (control == 1 || control == 2) CFRunLoopWakeUp(CFRunLoopGetMain());
    /* A separate, explicit handled-source deadline keeps even a broken
     * Darwin CFRunLoop timeout bounded and measures the missing wake. */
    delay.nanoseconds = 800000000;
    __asm__ volatile("syscall" : "=a"(result) : "a"(35L), "D"(&delay), "S"(0L)
                     : "rcx", "r11", "memory");
    CFRunLoopSourceSignal(source); CFRunLoopWakeUp(CFRunLoopGetMain());
    [pool release];
    return 0;
}
int main(int argc, char **argv) {
    alarm(8);
    [[NSAutoreleasePool alloc] init];
    control = argc > 1 && !strcmp(argv[1], "timer") ? 3 :
              argc > 1 && !strcmp(argv[1], "source") ? 2 :
              argc > 1 && !strcmp(argv[1], "wake") ? 1 : 0;
    app = [NSApplication sharedApplication]; [app finishLaunching];
    NSDisplay *display = [NSDisplay currentDisplay];
    for (int i=0; i<5; i++) {
        [display processPendingEvents];
        [display nextEventMatchingMask:~0ULL untilDate:[NSDate date] inMode:NSDefaultRunLoopMode dequeue:1];
    }
    Ivar ivar = class_getInstanceVariable(objc_getClass("NSDisplay"), "_eventQueue");
    if (!ivar) return 2;
    [(NSMutableArray *)*(id *)((char *)display + ivar_getOffset(ivar)) removeAllObjects];
    struct {
        long version; void *info; void *retain; void *release; void *description;
        void *equal; void *hash; void *schedule; void *cancel; void (*perform)(void *);
    } context = {.perform=perform};
    source = CFRunLoopSourceCreate(0, 0, &context);
    if (!source) return 3;
    CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
    const void *timer = 0;
    if (control == 3) {
        timer = CFRunLoopTimerCreate(0, CFAbsoluteTimeGetCurrent()+0.2, 0, 0, 0, timer_post, 0);
        if (!timer) return 7;
        CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes);
    }
    id mode = argc > 3 && !strcmp(argv[3], "tracking") ? NSEventTrackingRunLoopMode :
              argc > 3 && !strcmp(argv[3], "modal") ? NSModalPanelRunLoopMode : NSDefaultRunLoopMode;
    void *worker;
    if (pthread_create(&worker, 0, producer, 0)) return 4;
    u64 started = now_ns();
    id deadline = [NSDate dateWithTimeIntervalSinceNow:2];
    NSEvent *event = [app nextEventMatchingMask:(1ULL<<15) untilDate:deadline
                                      inMode:mode dequeue:1];
    u64 returned = now_ns();
    pthread_join(worker, 0);
    u64 produced = __atomic_load_n(&posted, __ATOMIC_ACQUIRE);
    if (source) {
        CFRunLoopRemoveSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes); CFRelease(source);
    }
    if (timer) { CFRunLoopTimerInvalidate(timer); CFRelease(timer); }
    double wait_ms = (returned-started)/1e6, delivery_ms = (returned-produced)/1e6;
    printf("Event wakeup control=%s mode=%s type=%lu wait_ms=%.3f delivery_ms=%.3f\n",
           control==3?"timer":control==2?"source":control==1?"wake":"none",
           argc>3?argv[3]:"default", [event type], wait_ms, delivery_ms);
    fflush(0);
    if ([event type] != 15 || !produced || returned < produced) return 5;
    /* Diagnostic baseline accepts a missing wake; --require-ready turns it
     * into a regression check after the production fix is authorized. */
    if ((control == 2 || (argc > 2 && !strcmp(argv[2], "--require-ready"))) && delivery_ms > 500) return 6;
    return 0;
}
