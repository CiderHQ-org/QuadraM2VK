//! The IOSurfaces of the Metal side's surface cells (`create.iosurface`): a surface laid out as
//! the runtime side's buffer stands for one, so a texture over it reads the same bytes. The
//! surface is 16 x 16, its allocation 64 KiB, filled with the cell's pattern from its base.
//!
//! - **Without planes** (`planar=false`): rows 256 bytes apart, each element the format's unit
//!   (its texel, or its block with the block's size in texels), no OSType.
//! - **With planes** (`planar=true`): plane info (`kIOSurfacePlaneInfo`). A format the runtime
//!   side gives surface facts (a two-plane YCbCr or extended-range format: `surface_facts`) has
//!   their OSType and two planes, plane 0 at byte 0 and plane 1 at the facts' chroma offset, rows
//!   256 bytes apart, each plane's size in texels and its elements as CoreVideo makes a surface of
//!   that OSType (`CVPixelBufferCreate`, read back plane by plane); any other format one plane, at
//!   byte 0, laid out as without planes.
//!
//! The surface's frameworks are reached through their C functions (CoreFoundation, IOSurface,
//! CoreVideo); each object made here is released here.

use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::encode::{Encoding, RefEncode};

/// `IOSurfaceRef`'s pointee, opaque, encoded as Metal's methods declare it.
#[repr(C)]
pub(super) struct IoSurfaceOpaque {
    _private: [u8; 0],
}

// SAFETY: `IOSurfaceRef` is `struct __IOSurface *`, which Objective-C encodes so.
unsafe impl RefEncode for IoSurfaceOpaque {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__IOSurface", &[]));
}

type CfType = *const c_void;
type CfAllocator = *const c_void;
type CfMutableDictionary = *mut c_void;

/// `kCFNumberSInt64Type`.
const NUMBER_SINT64: isize = 4;
/// `kIOSurfaceLockReadOnly`.
const LOCK_READ_ONLY: u32 = 1;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    static kCFTypeDictionaryKeyCallBacks: u8;
    static kCFTypeDictionaryValueCallBacks: u8;
    static kCFTypeArrayCallBacks: u8;
    fn CFDictionaryCreateMutable(
        allocator: CfAllocator,
        capacity: isize,
        keys: *const u8,
        values: *const u8,
    ) -> CfMutableDictionary;
    fn CFDictionarySetValue(dictionary: CfMutableDictionary, key: CfType, value: CfType);
    fn CFNumberCreate(allocator: CfAllocator, number_type: isize, value: *const c_void) -> CfType;
    fn CFArrayCreate(
        allocator: CfAllocator,
        values: *const CfType,
        count: isize,
        callbacks: *const u8,
    ) -> CfType;
    fn CFRelease(object: CfType);
}

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    static kIOSurfaceWidth: CfType;
    static kIOSurfaceHeight: CfType;
    static kIOSurfaceBytesPerRow: CfType;
    static kIOSurfaceBytesPerElement: CfType;
    static kIOSurfaceElementWidth: CfType;
    static kIOSurfaceElementHeight: CfType;
    static kIOSurfacePixelFormat: CfType;
    static kIOSurfaceAllocSize: CfType;
    static kIOSurfacePlaneInfo: CfType;
    static kIOSurfacePlaneWidth: CfType;
    static kIOSurfacePlaneHeight: CfType;
    static kIOSurfacePlaneBytesPerRow: CfType;
    static kIOSurfacePlaneOffset: CfType;
    static kIOSurfacePlaneSize: CfType;
    static kIOSurfacePlaneBytesPerElement: CfType;
    static kIOSurfacePlaneElementWidth: CfType;
    static kIOSurfacePlaneElementHeight: CfType;
    fn IOSurfaceCreate(properties: CfType) -> *mut IoSurfaceOpaque;
    fn IOSurfaceLock(surface: *mut IoSurfaceOpaque, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut IoSurfaceOpaque, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(surface: *mut IoSurfaceOpaque) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: *mut IoSurfaceOpaque) -> usize;
    fn IOSurfaceGetPlaneCount(surface: *mut IoSurfaceOpaque) -> usize;
    fn IOSurfaceGetWidthOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerElementOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetElementWidthOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetElementHeightOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferIOSurfacePropertiesKey: CfType;
    fn CVPixelBufferCreate(
        allocator: CfAllocator,
        width: usize,
        height: usize,
        pixel_format: u32,
        attributes: CfType,
        out: *mut CfType,
    ) -> i32;
    fn CVPixelBufferGetIOSurface(buffer: CfType) -> *mut IoSurfaceOpaque;
}

/// The surface's size in texels, a side.
const SIDE: u64 = 16;
/// The bytes between rows of every plane.
const ROW: u64 = 256;
/// The surface's allocation: the runtime side's buffer's length.
pub(super) const BYTES: usize = 1 << 16;

/// One plane's layout: its size in texels, and its elements' bytes and size in texels.
#[derive(Clone, Copy, Debug)]
pub(super) struct Plane {
    pub width: u64,
    pub height: u64,
    pub bytes_per_element: u64,
    pub element_width: u64,
    pub element_height: u64,
}

/// What a two-plane surface states beyond its planes' layout: its OSType and plane 1's first
/// byte (the runtime side's surface facts).
#[derive(Clone, Copy, Debug)]
pub(super) struct TwoPlanes {
    pub pixel_format: u32,
    pub chroma_offset: u64,
}

/// A CoreFoundation object made here, released as it goes.
struct Owned(CfType);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: an object this module made, released once.
            unsafe { CFRelease(self.0) };
        }
    }
}

fn number(value: u64) -> Owned {
    let value = value as i64;
    // SAFETY: a 64-bit integer read through its pointer during the call.
    Owned(unsafe {
        CFNumberCreate(
            core::ptr::null(),
            NUMBER_SINT64,
            (&raw const value).cast::<c_void>(),
        )
    })
}

/// A mutable dictionary of `pairs`, its keys IOSurface's (or CoreVideo's) string constants.
fn dictionary(pairs: &[(CfType, CfType)]) -> Owned {
    // SAFETY: CoreFoundation's own callbacks; each key and value is a live CF object, which
    // the dictionary retains.
    unsafe {
        let d = CFDictionaryCreateMutable(
            core::ptr::null(),
            0,
            &raw const kCFTypeDictionaryKeyCallBacks,
            &raw const kCFTypeDictionaryValueCallBacks,
        );
        for &(key, value) in pairs {
            CFDictionarySetValue(d, key, value);
        }
        Owned(d.cast_const())
    }
}

fn array(values: &[CfType]) -> Owned {
    // SAFETY: live CF objects, which the array retains.
    Owned(unsafe {
        CFArrayCreate(
            core::ptr::null(),
            values.as_ptr(),
            values.len() as isize,
            &raw const kCFTypeArrayCallBacks,
        )
    })
}

/// The planes CoreVideo lays out for a 16 x 16 surface of OSType `pixel_format`, or `None` where
/// it makes none.
pub(super) fn core_video_planes(pixel_format: u32) -> Option<Vec<Plane>> {
    let empty = dictionary(&[]);
    // SAFETY: CoreVideo's key constant, read once.
    let attributes = dictionary(&[(unsafe { kCVPixelBufferIOSurfacePropertiesKey }, empty.0)]);
    let mut buffer: CfType = core::ptr::null();
    // SAFETY: a 16 x 16 buffer; `buffer` receives a retained object or stays null.
    let status = unsafe {
        CVPixelBufferCreate(
            core::ptr::null(),
            SIDE as usize,
            SIDE as usize,
            pixel_format,
            attributes.0,
            &raw mut buffer,
        )
    };
    let buffer = Owned(buffer);
    if status != 0 || buffer.0.is_null() {
        return None;
    }
    // SAFETY: the buffer's surface, which the buffer keeps while it lives (borrowed).
    let surface = unsafe { CVPixelBufferGetIOSurface(buffer.0) };
    if surface.is_null() {
        return None;
    }
    // SAFETY: plane queries of a live surface, each plane below its count.
    unsafe {
        let count = IOSurfaceGetPlaneCount(surface);
        Some(
            (0..count)
                .map(|p| Plane {
                    width: IOSurfaceGetWidthOfPlane(surface, p) as u64,
                    height: IOSurfaceGetHeightOfPlane(surface, p) as u64,
                    bytes_per_element: IOSurfaceGetBytesPerElementOfPlane(surface, p) as u64,
                    element_width: IOSurfaceGetElementWidthOfPlane(surface, p) as u64,
                    element_height: IOSurfaceGetElementHeightOfPlane(surface, p) as u64,
                })
                .collect(),
        )
    }
}

/// A surface made here, released as it goes.
pub(super) struct Surface(NonNull<IoSurfaceOpaque>);

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: the surface `IOSurfaceCreate` made, released once.
        unsafe { CFRelease(self.0.as_ptr().cast_const().cast::<c_void>()) };
    }
}

impl Surface {
    /// A 16 x 16 surface without planes, its elements `element` (bytes, texels across and down),
    /// rows 256 bytes apart.
    pub(super) fn plain(element: (u64, u64, u64)) -> Option<Surface> {
        let (bytes, across, down) = element;
        // SAFETY: IOSurface's key constants, read once each.
        let keys = unsafe {
            [
                kIOSurfaceWidth,
                kIOSurfaceHeight,
                kIOSurfaceBytesPerRow,
                kIOSurfaceBytesPerElement,
                kIOSurfaceElementWidth,
                kIOSurfaceElementHeight,
                kIOSurfaceAllocSize,
            ]
        };
        let values = [SIDE, SIDE, ROW, bytes, across, down, BYTES as u64].map(number);
        let pairs: Vec<_> = keys.iter().zip(&values).map(|(&k, v)| (k, v.0)).collect();
        Surface::create(&dictionary(&pairs))
    }

    /// A 16 x 16 surface with `planes` (plane 0 at byte 0, plane 1 at `two`'s chroma offset),
    /// rows 256 bytes apart, of `two`'s OSType where it has one.
    pub(super) fn planar(planes: &[Plane], two: Option<TwoPlanes>) -> Option<Surface> {
        // SAFETY: IOSurface's key constants, read once each.
        let plane_keys = unsafe {
            [
                kIOSurfacePlaneWidth,
                kIOSurfacePlaneHeight,
                kIOSurfacePlaneBytesPerRow,
                kIOSurfacePlaneOffset,
                kIOSurfacePlaneSize,
                kIOSurfacePlaneBytesPerElement,
                kIOSurfacePlaneElementWidth,
                kIOSurfacePlaneElementHeight,
            ]
        };
        let mut held = Vec::new();
        let mut infos = Vec::new();
        for (i, p) in planes.iter().enumerate() {
            let offset = match (i, two) {
                (0, _) => 0,
                (_, Some(t)) => t.chroma_offset,
                (_, None) => return None,
            };
            let rows = p.height.div_ceil(p.element_height.max(1));
            let values = [
                p.width,
                p.height,
                ROW,
                offset,
                ROW * rows,
                p.bytes_per_element,
                p.element_width,
                p.element_height,
            ]
            .map(number);
            let pairs: Vec<_> = plane_keys
                .iter()
                .zip(&values)
                .map(|(&k, v)| (k, v.0))
                .collect();
            let info = dictionary(&pairs);
            infos.push(info.0);
            held.push((info, values));
        }
        let info = array(&infos);
        let first = planes.first()?;
        // SAFETY: IOSurface's key constants, read once each.
        let mut pairs = unsafe {
            vec![
                (kIOSurfaceWidth, number(SIDE)),
                (kIOSurfaceHeight, number(SIDE)),
                (kIOSurfaceBytesPerElement, number(first.bytes_per_element)),
                (kIOSurfaceAllocSize, number(BYTES as u64)),
            ]
        };
        if let Some(t) = two {
            // SAFETY: IOSurface's key constant, read once.
            pairs.push((
                unsafe { kIOSurfacePixelFormat },
                number(u64::from(t.pixel_format)),
            ));
        }
        let mut raw: Vec<_> = pairs.iter().map(|(k, v)| (*k, v.0)).collect();
        // SAFETY: IOSurface's key constant, read once.
        raw.push((unsafe { kIOSurfacePlaneInfo }, info.0));
        Surface::create(&dictionary(&raw))
    }

    fn create(properties: &Owned) -> Option<Surface> {
        // SAFETY: a dictionary of IOSurface's properties; the surface comes back retained.
        NonNull::new(unsafe { IOSurfaceCreate(properties.0) }).map(Surface)
    }

    /// The surface, as Metal's `newTextureWithDescriptor:iosurface:plane:` takes it.
    pub(super) fn as_ptr(&self) -> *mut IoSurfaceOpaque {
        self.0.as_ptr()
    }

    /// Run `f` over the surface's first [`BYTES`] bytes, locked (read-only where `read_only`).
    fn with_bytes<T>(&self, read_only: bool, f: impl FnOnce(*mut u8) -> T) -> Option<T> {
        let options = if read_only { LOCK_READ_ONLY } else { 0 };
        let s = self.as_ptr();
        // SAFETY: the surface is live; its allocation holds at least `BYTES` (checked), and the
        // lock keeps its base address valid until the unlock.
        unsafe {
            if IOSurfaceGetAllocSize(s) < BYTES
                || IOSurfaceLock(s, options, core::ptr::null_mut()) != 0
            {
                return None;
            }
            let base = IOSurfaceGetBaseAddress(s).cast::<u8>();
            let out = (!base.is_null()).then(|| f(base));
            IOSurfaceUnlock(s, options, core::ptr::null_mut());
            out
        }
    }

    /// Write `bytes` (at most [`BYTES`]) from the surface's base.
    pub(super) fn fill(&self, bytes: &[u8]) -> Option<()> {
        let n = bytes.len().min(BYTES);
        // SAFETY: `n` bytes inside the locked allocation.
        self.with_bytes(false, |base| unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, n)
        })
    }

    /// The surface's first [`BYTES`] bytes.
    pub(super) fn bytes(&self) -> Option<Vec<u8>> {
        // SAFETY: `BYTES` bytes inside the locked allocation.
        self.with_bytes(true, |base| unsafe {
            std::slice::from_raw_parts(base, BYTES).to_vec()
        })
    }
}

/// The planes of a two-plane surface of OSType `pixel_format` where CoreVideo makes none (used
/// only then): for `'b3a8'` (`kCVPixelFormatType_30RGBLE_8A_BiPlanar`), a 32-bit color plane and
/// an 8-bit alpha plane, both 16 x 16, the elements of `RGB10A8_2P_XR10`'s two planes.
pub(super) fn known_planes(pixel_format: u32) -> Option<Vec<Plane>> {
    let plane = |bytes_per_element| Plane {
        width: SIDE,
        height: SIDE,
        bytes_per_element,
        element_width: 1,
        element_height: 1,
    };
    (pixel_format == u32::from_be_bytes(*b"b3a8")).then(|| vec![plane(4), plane(1)])
}

/// `pixel_format` as its four characters.
pub(super) fn four_cc(pixel_format: u32) -> String {
    pixel_format
        .to_be_bytes()
        .iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
        .collect()
}
