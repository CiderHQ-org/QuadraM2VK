//! A fatal signal's native backtrace (lane tooling).
//!
//! The Mac minis that run the GPU lanes cannot attach a debugger over ssh, and a lane process
//! that dies on a signal there leaves no crash report. So every process that makes a lane device
//! ([`install`], from `validated_device`) handles `SIGSEGV`, `SIGBUS`, `SIGILL` and `SIGABRT`:
//! the handler writes to standard error the signal, the address it faulted at, the interrupted
//! program counter and link register (each with its image, the image's load address and the
//! nearest symbol, so `atos -o <image> -l <load address> <pc>` names a local function the
//! symbol table keeps), and the native backtrace of the thread it hit (`backtrace` and
//! `backtrace_symbols_fd`, which allocate nothing once `backtrace` has been called), puts back the
//! default action, and raises the signal again, so the process dies of it as it would have. It
//! replaces Rust's own `SIGSEGV` and `SIGBUS` handler, whose stack-overflow message the
//! backtrace's repeated frames stand in for.
//!
//! The registers matter: the backtrace walks frame records, and a fault in a function that made
//! no call yet shows only its caller there.
//!
//! It writes the signal's code too (`si_code`: on Linux x86-64 a general-protection fault, such as
//! a non-canonical address, is `SI_KERNEL`, 128, and its fault address reads 0), every general
//! register, and the 16 bytes at the program counter where its page is not the fault's: what a
//! fault in JIT-compiled code, which no image holds, needs to be read back.

use std::sync::Once;

/// The signals handled.
const SIGNALS: [libc::c_int; 4] = [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGABRT];

/// The most frames written.
const FRAMES: usize = 128;

/// Install the handler for every one of [`SIGNALS`], once per process.
pub(super) fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut frames = [std::ptr::null_mut::<libc::c_void>(); 1];
        // SAFETY: `frames` holds one frame. This first call loads what `backtrace` needs, so a
        // call in the handler allocates nothing.
        unsafe { libc::backtrace(frames.as_mut_ptr(), 1) };
        for signal in SIGNALS {
            // SAFETY: a zeroed `sigaction` is a valid value, and every field that matters is set
            // below; `on_fatal` is async-signal-safe (its docs).
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                    on_fatal;
                action.sa_sigaction = handler as libc::sighandler_t;
                action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(signal, &action, std::ptr::null_mut());
            }
        }
    });
}

/// Write `bytes` to standard error, as much as one `write` takes.
fn write_err(bytes: &[u8]) {
    // SAFETY: `bytes` is live for the call; `write` is async-signal-safe.
    unsafe { libc::write(libc::STDERR_FILENO, bytes.as_ptr().cast(), bytes.len()) };
}

/// The decimal digits of `value`, in `buffer`, without allocating.
fn decimal(value: libc::c_int, buffer: &mut [u8; 12]) -> &[u8] {
    let mut at = buffer.len();
    let negative = value < 0;
    let mut rest = value.unsigned_abs();
    loop {
        at -= 1;
        buffer[at] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    if negative {
        at -= 1;
        buffer[at] = b'-';
    }
    &buffer[at..]
}

/// `0x` and the 16 hexadecimal digits of `value`, in `buffer`, without allocating.
fn hex(value: u64, buffer: &mut [u8; 18]) -> &[u8] {
    buffer[0] = b'0';
    buffer[1] = b'x';
    for (at, digit) in buffer[2..].iter_mut().enumerate() {
        let nibble = (value >> (60 - 4 * at)) & 0xf;
        *digit = b"0123456789abcdef"[nibble as usize];
    }
    &buffer[..]
}

/// The interrupted program counter and link register of a signal's context, where this target's
/// context is known (`None` for the link register on x86-64, which has none).
///
/// # Safety
///
/// `context` is the `ucontext_t` the kernel passed the handler.
unsafe fn registers(context: *mut libc::c_void) -> Option<(u64, Option<u64>)> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // SAFETY: the caller's; the kernel's context points at its machine context.
        let state = unsafe { &(*(*context.cast::<libc::ucontext_t>()).uc_mcontext).__ss };
        Some((state.__pc, Some(state.__lr)))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        // SAFETY: the caller's.
        let state = unsafe { &(*context.cast::<libc::ucontext_t>()).uc_mcontext };
        Some((state.pc, Some(state.regs[30])))
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // SAFETY: the caller's.
        let state = unsafe { &(*context.cast::<libc::ucontext_t>()).uc_mcontext };
        Some((state.gregs[libc::REG_RIP as usize] as u64, None))
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
    )))]
    {
        let _ = context;
        None
    }
}

/// The general registers of a signal's context, with their names, where this target's context is
/// known: x86-64's sixteen, or AArch64's `x0` to `x30`.
///
/// # Safety
///
/// `context` is the `ucontext_t` the kernel passed the handler.
unsafe fn general_registers(
    context: *mut libc::c_void,
) -> Option<([u64; 31], &'static [&'static [u8]])> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        const NAMES: [&[u8]; 16] = [
            b"rax", b"rbx", b"rcx", b"rdx", b"rsi", b"rdi", b"rbp", b"rsp", b"r8", b"r9", b"r10",
            b"r11", b"r12", b"r13", b"r14", b"r15",
        ];
        const AT: [libc::c_int; 16] = [
            libc::REG_RAX,
            libc::REG_RBX,
            libc::REG_RCX,
            libc::REG_RDX,
            libc::REG_RSI,
            libc::REG_RDI,
            libc::REG_RBP,
            libc::REG_RSP,
            libc::REG_R8,
            libc::REG_R9,
            libc::REG_R10,
            libc::REG_R11,
            libc::REG_R12,
            libc::REG_R13,
            libc::REG_R14,
            libc::REG_R15,
        ];
        // SAFETY: the caller's.
        let state = unsafe { &(*context.cast::<libc::ucontext_t>()).uc_mcontext };
        let mut values = [0; 31];
        for (value, at) in values.iter_mut().zip(AT) {
            *value = state.gregs[at as usize] as u64;
        }
        Some((values, &NAMES))
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        // SAFETY: the caller's.
        let state = unsafe { &(*context.cast::<libc::ucontext_t>()).uc_mcontext };
        Some((state.regs, &AARCH64_NAMES))
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // SAFETY: the caller's; the kernel's context points at its machine context.
        let state = unsafe { &(*(*context.cast::<libc::ucontext_t>()).uc_mcontext).__ss };
        let mut values = [0; 31];
        values[..29].copy_from_slice(&state.__x);
        values[29] = state.__fp;
        values[30] = state.__lr;
        Some((values, &AARCH64_NAMES))
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
    )))]
    {
        let _ = context;
        None
    }
}

/// AArch64's general registers' names.
#[cfg(target_arch = "aarch64")]
const AARCH64_NAMES: [&[u8]; 31] = [
    b"x0", b"x1", b"x2", b"x3", b"x4", b"x5", b"x6", b"x7", b"x8", b"x9", b"x10", b"x11", b"x12",
    b"x13", b"x14", b"x15", b"x16", b"x17", b"x18", b"x19", b"x20", b"x21", b"x22", b"x23", b"x24",
    b"x25", b"x26", b"x27", b"x28", b"x29", b"x30",
];

/// Write the 16 bytes at `pc` in hexadecimal, unless `pc` lies in the page of `fault` (reading
/// them would fault again).
fn write_code(pc: u64, fault: u64) {
    const PAGE: u64 = 4096;
    if pc / PAGE == fault / PAGE || (pc + 16) / PAGE == fault / PAGE || pc == 0 {
        return;
    }
    write_err(b"runtime_lane: code at pc");
    for at in 0..16 {
        // SAFETY: `pc` is the interrupted instruction's address, in a mapped executable page
        // that is not the faulting one; on the targets handled such pages are readable.
        let byte = unsafe { std::ptr::read_volatile((pc + at) as *const u8) };
        let digits = [
            b' ',
            b"0123456789abcdef"[usize::from(byte >> 4)],
            b"0123456789abcdef"[usize::from(byte & 0xf)],
        ];
        write_err(&digits);
    }
    write_err(b"\n");
}

/// The signal's code (`si_code`), as its information gives it.
///
/// # Safety
///
/// `info` is the `siginfo_t` the kernel passed the handler.
unsafe fn signal_code(info: *mut libc::siginfo_t) -> libc::c_int {
    // SAFETY: the caller's.
    unsafe { (*info).si_code }
}

/// The address a fault was at, as the signal's information gives it.
///
/// # Safety
///
/// `info` is the `siginfo_t` the kernel passed the handler.
unsafe fn fault_address(info: *mut libc::siginfo_t) -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: the caller's.
        unsafe { (*info).si_addr() as u64 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: the caller's.
        unsafe { (*info).si_addr as u64 }
    }
}

/// Write `name`, `address`, and the image that holds it with the image's load address, then the
/// address symbolized.
fn write_address(name: &[u8], address: u64) {
    let mut digits = [0; 18];
    write_err(b"runtime_lane: ");
    write_err(name);
    write_err(b" ");
    write_err(hex(address, &mut digits));
    // SAFETY: a zeroed `Dl_info` is valid, and `dladdr` fills it or answers 0; the image name it
    // gives is a C string of the loaded image, live while the image is.
    unsafe {
        let mut image: libc::Dl_info = std::mem::zeroed();
        if libc::dladdr(address as *const libc::c_void, &mut image) != 0
            && !image.dli_fname.is_null()
        {
            write_err(b" in ");
            let path = std::ffi::CStr::from_ptr(image.dli_fname);
            write_err(path.to_bytes());
            write_err(b" loaded at ");
            write_err(hex(image.dli_fbase as u64, &mut digits));
        }
    }
    write_err(b"\n");
    let frames = [address as *mut libc::c_void];
    // SAFETY: `frames` holds one address, and standard error is open.
    unsafe { libc::backtrace_symbols_fd(frames.as_ptr(), 1, libc::STDERR_FILENO) };
}

/// The handler: the signal, its fault address and registers, and the hit thread's native
/// backtrace to standard error, then the default action, and the signal again.
/// Async-signal-safe in practice: it allocates nothing and calls only `write`, `backtrace`
/// (loaded by [`install`]), `dladdr` and `backtrace_symbols_fd` (which reads the loaded images as
/// `dladdr` does), `signal` and `raise`.
extern "C" fn on_fatal(
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    let mut digits = [0; 12];
    write_err(b"\nruntime_lane: fatal signal ");
    write_err(decimal(signal, &mut digits));
    write_err(b"\n");
    let mut wide = [0; 18];
    write_err(b"runtime_lane: fault address ");
    // SAFETY: the kernel's signal information.
    let fault = unsafe { fault_address(info) };
    write_err(hex(fault, &mut wide));
    write_err(b"\n");
    write_err(b"runtime_lane: signal code ");
    // SAFETY: the kernel's signal information.
    write_err(decimal(unsafe { signal_code(info) }, &mut digits));
    write_err(b"\n");
    // SAFETY: the kernel's context.
    if let Some((pc, lr)) = unsafe { registers(context) } {
        write_address(b"pc", pc);
        if let Some(lr) = lr {
            write_address(b"lr", lr);
        }
        write_code(pc, fault);
    }
    // SAFETY: the kernel's context.
    if let Some((values, names)) = unsafe { general_registers(context) } {
        for (name, &value) in names.iter().zip(&values) {
            write_err(b"runtime_lane: ");
            write_err(name);
            write_err(b" ");
            write_err(hex(value, &mut wide));
            write_err(b"\n");
        }
    }
    write_err(b"runtime_lane: the native backtrace of the thread it hit:\n");
    let mut frames = [std::ptr::null_mut::<libc::c_void>(); FRAMES];
    // SAFETY: `frames` holds `FRAMES` frames, and standard error is open.
    unsafe {
        let count = libc::backtrace(frames.as_mut_ptr(), FRAMES as libc::c_int);
        libc::backtrace_symbols_fd(frames.as_ptr(), count, libc::STDERR_FILENO);
    }
    write_err(b"runtime_lane: end of the backtrace\n");
    // SAFETY: the default action of a signal is valid; the raised signal is blocked until this
    // handler returns, and then ends the process (a fault's instruction faults again too).
    unsafe {
        libc::signal(signal, libc::SIG_DFL);
        libc::raise(signal);
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;

    /// Set in the child process the test below runs.
    const CHILD: &str = "M2V_TEST_FATAL_SIGNAL_CHILD";

    /// Run in a process of its own by the test below: install the handler, then read a page
    /// mapped with no access, and print its address first. Anywhere else, nothing.
    #[test]
    #[ignore = "a child process of a_fatal_signal_writes_the_native_backtrace"]
    fn dies_of_a_fatal_signal() {
        if std::env::var_os(CHILD).is_none() {
            return;
        }
        super::install();
        // SAFETY: a fresh private anonymous mapping of one page, no access.
        let page = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(page, libc::MAP_FAILED);
        let mut digits = [0; 18];
        eprintln!(
            "page {}",
            std::str::from_utf8(super::hex(page as u64, &mut digits)).unwrap()
        );
        // SAFETY: none: the read faults, which is the point, and the process dies of it.
        unsafe { std::ptr::read_volatile(page.cast::<u8>()) };
    }

    /// A process that faults with the handler installed writes the signal, the address it
    /// faulted at, the interrupted program counter, and the hit thread's frames to standard
    /// error, and still dies of the signal.
    #[test]
    fn a_fatal_signal_writes_the_native_backtrace() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "runtime_lane::fatal_signals::tests::dies_of_a_fatal_signal",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        // A page with no access faults with SIGSEGV on Linux and SIGBUS on macOS.
        let signal = output.status.signal().unwrap_or_else(|| panic!("{stderr}"));
        assert!([libc::SIGSEGV, libc::SIGBUS].contains(&signal), "{stderr}");
        let page = stderr
            .lines()
            .find_map(|line| line.strip_prefix("page "))
            .unwrap_or_else(|| panic!("{stderr}"));
        let header = format!("runtime_lane: fatal signal {signal}\n");
        let at = stderr.find(&header).unwrap_or_else(|| panic!("{stderr}"));
        let report = &stderr[at..];
        // A read of a mapped page with no access: an ordinary fault, not a kernel-raised one.
        assert!(
            !report.contains("runtime_lane: signal code 128\n"),
            "{stderr}"
        );
        assert!(
            report.contains(&format!("runtime_lane: fault address {page}\n")),
            "{stderr}"
        );
        if cfg!(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(
                target_os = "linux",
                any(target_arch = "aarch64", target_arch = "x86_64")
            ),
        )) {
            assert!(report.contains("runtime_lane: pc 0x"), "{stderr}");
            assert!(report.contains("runtime_lane: code at pc "), "{stderr}");
            let register = match cfg!(target_arch = "x86_64") {
                true => "runtime_lane: rsp 0x",
                false => "runtime_lane: x30 0x",
            };
            assert!(report.contains(register), "{stderr}");
        }
        let start = report
            .find("runtime_lane: the native backtrace of the thread it hit:\n")
            .unwrap_or_else(|| panic!("{stderr}"));
        let end = report
            .find("runtime_lane: end of the backtrace")
            .unwrap_or_else(|| panic!("{stderr}"));
        let frames = report[start..end].lines().count() - 1;
        assert!(frames >= 2, "{stderr}");
    }

    /// Decimal and hexadecimal digits without allocating.
    #[test]
    fn digits() {
        let mut digits = [0; 12];
        for (value, text) in [(0, "0"), (11, "11"), (-6, "-6"), (i32::MIN, "-2147483648")] {
            assert_eq!(super::decimal(value, &mut digits), text.as_bytes());
        }
        let mut digits = [0; 18];
        assert_eq!(
            super::hex(0x1_0373_a3b8, &mut digits),
            b"0x000000010373a3b8"
        );
        assert_eq!(super::hex(u64::MAX, &mut digits), b"0xffffffffffffffff");
    }
}
