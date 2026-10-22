/* GPL-3.0-only. Require a physical Vulkan GPU without opening a display connection. */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void quoted(const char *value) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\');
        if (*p < 32) printf("\\u%04x", *p); else putchar(*p);
    }
    putchar('"');
}

int main(void) {
    void *library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "Vulkan loader: %s\n", dlerror()); return 1; }
    PFN_vkGetInstanceProcAddr get = (PFN_vkGetInstanceProcAddr)dlsym(library, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance create = get ? (PFN_vkCreateInstance)get(VK_NULL_HANDLE, "vkCreateInstance") : NULL;
    if (!create) { fputs("Vulkan loader has no vkCreateInstance\n", stderr); return 1; }
    VkApplicationInfo application = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="QuadraM2VK hardware preflight", .apiVersion=VK_API_VERSION_1_0};
    VkInstanceCreateInfo info = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&application};
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = create(&info, NULL, &instance);
    if (result != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance: %d\n", result); return 1; }
    PFN_vkDestroyInstance destroy = (PFN_vkDestroyInstance)get(instance, "vkDestroyInstance");
    PFN_vkEnumeratePhysicalDevices enumerate = (PFN_vkEnumeratePhysicalDevices)get(instance, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties properties = (PFN_vkGetPhysicalDeviceProperties)get(instance, "vkGetPhysicalDeviceProperties");
    if (!destroy || !enumerate || !properties) { fputs("Vulkan instance is missing required functions\n", stderr); return 1; }
    uint32_t count = 0;
    result = enumerate(instance, &count, NULL);
    if (result != VK_SUCCESS || count > 64) {
        fprintf(stderr, "vkEnumeratePhysicalDevices: %d, count %u\n", result, count);
        destroy(instance, NULL); return 1;
    }
    VkPhysicalDevice devices[64];
    if (count) result = enumerate(instance, &count, devices);
    if (result != VK_SUCCESS) { fprintf(stderr, "device enumeration: %d\n", result); destroy(instance, NULL); return 1; }
    int hardware = 0;
    puts("{\"devices\":[");
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties p;
        properties(devices[i], &p);
        int physical = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        hardware += physical;
        if (i) puts(",");
        fputs("{\"name\":", stdout); quoted(p.deviceName);
        printf(",\"vendor_id\":%u,\"device_id\":%u,\"type\":%u,\"driver_version\":%u,\"api_version\":%u,\"hardware\":%s}",
            p.vendorID, p.deviceID, p.deviceType, p.driverVersion, p.apiVersion, physical ? "true" : "false");
    }
    printf("\n],\"hardware_devices\":%d}\n", hardware);
    destroy(instance, NULL);
    dlclose(library);
    if (!hardware) { fputs("No discrete or integrated Vulkan GPU is accessible; hardware tests stopped.\n", stderr); return 2; }
    return 0;
}
