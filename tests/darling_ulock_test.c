/* Standalone Mach-O ABI fixture for Darling's exported ulock wrappers. */
typedef unsigned int u32;
typedef unsigned long long u64;
extern int __ulock_wait(u32, void *, u64, u32), __ulock_wake(u32, void *, u64);
extern int *__error(void);
extern int printf(const char *, ...);
extern unsigned alarm(unsigned);
static int failures;
static void check(const char *name, int actual, int expected, int expected_errno) {
    int error=*__error();
    printf("ulock %s return=%d errno=%d expected=%d/%d\n", name, actual, error, expected, expected_errno);
    if (actual != expected || error != expected_errno) failures++;
}
int main(void) {
    const u32 no_errno=0x01000000;
    const int sentinel=1234;
    u32 word=0;
    alarm(3);
    *__error()=sentinel;
    int result=__ulock_wait(no_errno|1, &word, 0, 1000);
    check("timed NO_ERRNO", result, -60, sentinel);
    *__error()=sentinel;
    result=__ulock_wait(1, &word, 0, 1000);
    check("timed ordinary", result, -1, 60);
    *__error()=sentinel;
    result=__ulock_wait(no_errno|255, &word, 0, 1000);
    check("invalid wait NO_ERRNO", result, -22, sentinel);
    *__error()=sentinel;
    result=__ulock_wait(255, &word, 0, 1000);
    check("invalid wait ordinary", result, -1, 22);
    *__error()=sentinel;
    result=__ulock_wake(no_errno|255, &word, 0);
    check("invalid wake NO_ERRNO", result, -22, sentinel);
    *__error()=sentinel;
    result=__ulock_wait(no_errno|1, &word, 1, 1000);
    check("changed value", result, 1, sentinel);
    return failures ? 1 : 0;
}
