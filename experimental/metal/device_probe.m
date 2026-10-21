// SPDX-License-Identifier: MIT
typedef struct objc_object *id;
typedef struct objc_class *Class;
typedef struct objc_selector *SEL;
extern Class objc_getClass(const char *);
extern SEL sel_registerName(const char *);
extern id objc_msgSend(id, SEL, ...);
extern id MTLCreateSystemDefaultDevice(void);
extern long write(int, const void *, unsigned long);
#define STEP(text) write(2, text "\n", sizeof(text "\n") - 1)
static void leave(int code) {
    __asm__ volatile("syscall" : : "a"(231L), "D"((long)code) : "rcx", "r11", "memory");
}
int main(void) {
    STEP("Native Metal probe: entered main");
    ((id (*)(id, SEL))objc_msgSend)((id)objc_getClass("NSAutoreleasePool"), sel_registerName("new"));
    id device = MTLCreateSystemDefaultDevice();
    if(!device) { STEP("Native Metal probe: no Vulkan-backed Metal device"); leave(2); }
    STEP("Native Metal probe: created Vulkan-backed Metal device");
    id buffer = ((id (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
        device, sel_registerName("newBufferWithLength:options:"), 4096, 0);
    unsigned char *pixels = buffer ? ((void *(*)(id, SEL))objc_msgSend)(buffer, sel_registerName("contents")) : 0;
    unsigned long length = buffer ? ((unsigned long (*)(id, SEL))objc_msgSend)(buffer, sel_registerName("length")) : 0;
    unsigned long long address = buffer ? ((unsigned long long (*)(id, SEL))objc_msgSend)(buffer, sel_registerName("gpuAddress")) : 0;
    if(!pixels || length != 4096 || !address) { STEP("Native Metal probe: failed buffer allocation/map/address"); leave(3); }
    for(unsigned long i = 0; i < length; ++i) pixels[i] = (unsigned char)i;
    STEP("Native Metal probe: allocated and mapped 4096 bytes of Vulkan memory");
    id queue = ((id (*)(id, SEL))objc_msgSend)(device, sel_registerName("newCommandQueue"));
    id commands = queue ? ((id (*)(id, SEL))objc_msgSend)(queue, sel_registerName("commandBuffer")) : 0;
    if(!commands) { STEP("Native Metal probe: failed command buffer allocation"); leave(4); }
    ((void (*)(id, SEL))objc_msgSend)(commands, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(commands, sel_registerName("waitUntilCompleted"));
    STEP("Native Metal probe: Vulkan command buffer submitted and completed");
    leave(0);
    return 0;
}
