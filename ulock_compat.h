#ifndef MACOBLOX_ULOCK_COMPAT_H
#define MACOBLOX_ULOCK_COMPAT_H

extern int *__error(void);
extern int __ulock_wait(unsigned int, void *, unsigned long long, unsigned int);
extern int __ulock_wake(unsigned int, void *, unsigned long long);
#define MACOBLOX_ULF_NO_ERRNO 0x01000000U

/* Darling's exported wrappers route encoded NO_ERRNO errors through cerror.
 * Use their ordinary error path, then restore Darwin's negative-error ABI.
 * The original functions retain all timeout, futex and successful-return
 * behavior; ordinary callers are forwarded without reading errno. */
static int macoblox_ulock_wait(unsigned int operation, void *address,
        unsigned long long value, unsigned int timeout) {
    if (!(operation & MACOBLOX_ULF_NO_ERRNO))
        return __ulock_wait(operation, address, value, timeout);
    int *error=__error();
    int saved=*error;
    int result=__ulock_wait(operation & ~MACOBLOX_ULF_NO_ERRNO, address, value, timeout);
    if (result == -1) result=-*error;
    *error=saved;
    return result;
}

static int macoblox_ulock_wake(unsigned int operation, void *address,
        unsigned long long value) {
    if (!(operation & MACOBLOX_ULF_NO_ERRNO))
        return __ulock_wake(operation, address, value);
    int *error=__error();
    int saved=*error;
    int result=__ulock_wake(operation & ~MACOBLOX_ULF_NO_ERRNO, address, value);
    if (result == -1) result=-*error;
    *error=saved;
    return result;
}
#endif
