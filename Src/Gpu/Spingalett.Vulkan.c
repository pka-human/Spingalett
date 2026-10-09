/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Vulkan compute for the GPU backend (Spingalett.Vulkan.h). The loader (libvulkan.so.1,
 * vulkan-1.dll, libvulkan.1.dylib or MoltenVK) is opened at run time and every function is fetched
 * through vkGetInstanceProcAddr / vkGetDeviceProcAddr, so nothing links against Vulkan. The first
 * discrete GPU with Vulkan 1.2, a compute queue and buffer device addresses is used, else an
 * integrated one, else any other (a CPU implementation such as lavapipe). SPINGALETT_GPU_DEVICE
 * picks one by its index instead. Kernels are SPIR-V compiled from Src/Gpu/Shaders and embedded
 * (Spingalett.Kernels.h); their pipelines are made when first dispatched.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include "Spingalett.Vulkan.h"
#include "Spingalett.Thread.h"
#include "Spingalett.Kernels.h"         /* spg_kernel_spirv[], spg_kernel_spirv_size[], spg_kernel_names[] */
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static void *open_loader(void) { return (void *)LoadLibraryA("vulkan-1.dll"); }
static PFN_vkVoidFunction loader_symbol(void *lib, const char *name) {
    return (PFN_vkVoidFunction)(void (*)(void))GetProcAddress((HMODULE)lib, name);
}
#else
#include <dlfcn.h>
static void *open_loader(void) {
    static const char *const names[] = {
#if defined(__APPLE__)
        /* the Vulkan SDK's or Homebrew's loader (Homebrew's directories are not searched by default),
           or MoltenVK itself */
        "libvulkan.1.dylib", "libvulkan.dylib", "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib", "libMoltenVK.dylib", "/opt/homebrew/lib/libMoltenVK.dylib",
        "/usr/local/lib/libMoltenVK.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
    };
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
        void *lib = dlopen(names[k], RTLD_NOW | RTLD_LOCAL);
        if (lib) return lib;
    }
    return NULL;
}
static PFN_vkVoidFunction loader_symbol(void *lib, const char *name) {
    PFN_vkVoidFunction f;
    void *symbol = dlsym(lib, name);
    memcpy(&f, &symbol, sizeof f);      /* object to function pointer, as POSIX allows */
    return f;
}
#endif

/* ------------------------------------------------------------------------- the functions used */

#define VK_GLOBAL_FUNCTIONS(X) X(vkCreateInstance) X(vkEnumerateInstanceVersion)
#define VK_INSTANCE_FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkDestroyInstance)
#define VK_DEVICE_FUNCTIONS(X) \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkGetBufferDeviceAddress) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkCreateComputePipelines) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdPipelineBarrier) \
    X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCreateFence) X(vkDestroyFence) X(vkResetFences) \
    X(vkWaitForFences) X(vkQueueSubmit) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)

/* VK_KHR_cooperative_matrix and VK_KHR_shader_bfloat16, declared here for headers that predate them */
typedef struct {
    VkStructureType sType;
    void *pNext;
    VkBool32 cooperativeMatrix, cooperativeMatrixRobustBufferAccess;
} CoopFeatures;
typedef struct {
    VkStructureType sType;
    void *pNext;
    VkBool32 type, dot_product, cooperative_matrix;
} Bf16Features;
typedef struct {
    VkStructureType sType;
    void *pNext;
    uint32_t M, N, K;
    int32_t a, b, c, result;
    VkBool32 saturating;
    int32_t scope;
} CoopProperties;
typedef VkResult (VKAPI_PTR *CoopPropertiesFn)(VkPhysicalDevice, uint32_t *, CoopProperties *);
#define COOP_FEATURES_TYPE   ((VkStructureType)1000506000)
#define COOP_PROPERTIES_TYPE ((VkStructureType)1000506001)
#define BF16_FEATURES_TYPE   ((VkStructureType)1000141000)
#define COMPONENT_FLOAT32    1
#define COMPONENT_BFLOAT16   1000141000
#define SCOPE_SUBGROUP       3
#define SPARE_COMMANDS       64      /* freed command buffers kept for reuse */

#define DECLARE(name) static PFN_##name name;
VK_GLOBAL_FUNCTIONS(DECLARE)
VK_INSTANCE_FUNCTIONS(DECLARE)
VK_DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

/* ------------------------------------------------------------------------- the device */

typedef struct {
    SpgKernel kernel;
    uint32_t count, spec[SPG_SPEC_MAX];
    VkPipeline pipeline;
} Pipeline;

static struct {
    atomic_int state;               /* 0: not tried, 1: opening, 2: open, 3: unavailable */
    void *loader;
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t family;
    VkPhysicalDeviceMemoryProperties memory;
    uint64_t heap;
    bool host_writes;               /* the largest device-local heap has host-visible memory */
    uint32_t shared;
    uint32_t max_groups[3];         /* workgroups a dispatch may have in x, y and z */
    uint32_t subgroup;              /* the subgroup size */
    bool mma_bf16;                  /* cooperative matrices of bfloat16, 16 x 16 x 16, sums in float */
    bool storage16;                 /* 16-bit values in storage buffers (bfloat16 activations) */
    bool full_subgroups;            /* the matrix units' kernel can ask for subgroups without inactive lanes */
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    VkPipelineLayout layout;
    VkShaderModule modules[SPG_KERNEL_COUNT];
    Pipeline *pipelines;            /* made so far, in the order they were */
    size_t pipeline_count, pipeline_cap;
    SpgSignal *lock;                /* the pipelines, the queue and the spare command buffers */
    struct SpgGpuCommands *spare[SPARE_COMMANDS];   /* freed, for the next spg_gpu_commands_create() */
    uint32_t spare_count;
    float tick_ns;                  /* timestamp period */
    uint32_t stamp_bits;            /* valid bits of the queue's timestamps (0: none) */
    bool profile;                   /* SPINGALETT_GPU_PROFILE: time every dispatch */
    struct { char label[48]; double ms; uint64_t calls; } stats[256];
    uint32_t stat_count;
    double busy_ms;                 /* profiling: first to last timestamp of every submission */
    uint64_t submissions;
} gpu;

/* ------------------------------------------------------------------------- profiling */

/* With SPINGALETT_GPU_PROFILE set, every dispatch and copy is timed (timestamps around it) and the
   time per kernel and modes is printed when the process exits, with the time from the first
   timestamp of each submission to its last (what the device spent on it, gaps between dispatches
   included). */
#define PROFILE_QUERIES 8192u

static void profile_report(void) {
    if (gpu.stat_count == 0) return;
    double total = 0.0;
    for (uint32_t k = 0; k < gpu.stat_count; k++) total += gpu.stats[k].ms;
    /* largest first */
    for (uint32_t a = 0; a < gpu.stat_count; a++)
        for (uint32_t b = a + 1; b < gpu.stat_count; b++)
            if (gpu.stats[b].ms > gpu.stats[a].ms) {
                __typeof__(gpu.stats[0]) t = gpu.stats[a];
                gpu.stats[a] = gpu.stats[b];
                gpu.stats[b] = t;
            }
    fprintf(stderr, "GPU time by kernel (%.1f ms in all, %.1f ms from first to last timestamp of %llu submissions):\n",
            total, gpu.busy_ms, (unsigned long long)gpu.submissions);
    for (uint32_t k = 0; k < gpu.stat_count; k++)
        fprintf(stderr, "  %-46s %10.2f ms %5.1f%% %9llu calls %8.1f us each\n", gpu.stats[k].label, gpu.stats[k].ms,
                100.0 * gpu.stats[k].ms / total, (unsigned long long)gpu.stats[k].calls,
                1000.0 * gpu.stats[k].ms / (double)gpu.stats[k].calls);
}

static void profile_add(const char *label, double ms) {
    for (uint32_t k = 0; k < gpu.stat_count; k++)
        if (!strcmp(gpu.stats[k].label, label)) {
            gpu.stats[k].ms += ms;
            gpu.stats[k].calls++;
            return;
        }
    if (gpu.stat_count < sizeof gpu.stats / sizeof gpu.stats[0]) {
        snprintf(gpu.stats[gpu.stat_count].label, sizeof gpu.stats[0].label, "%s", label);
        gpu.stats[gpu.stat_count].ms = ms;
        gpu.stats[gpu.stat_count++].calls = 1;
    }
}

static bool has_extension(const VkExtensionProperties *list, uint32_t count, const char *name) {
    for (uint32_t k = 0; k < count; k++)
        if (!strcmp(list[k].extensionName, name)) return true;
    return false;
}

static bool open_device(void) {
    gpu.loader = open_loader();
    if (!gpu.loader) return false;
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)loader_symbol(gpu.loader, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return false;
#define LOAD_GLOBAL(name) name = (PFN_##name)vkGetInstanceProcAddr(NULL, #name);
    VK_GLOBAL_FUNCTIONS(LOAD_GLOBAL)
#undef LOAD_GLOBAL
    uint32_t api = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&api);
    if (!vkCreateInstance || api < VK_API_VERSION_1_2) return false;

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "Spingalett", 1, "Spingalett", 1,
                             VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
#if defined(__APPLE__)
    /* MoltenVK is listed as a portability implementation only */
    static const char *const instance_extensions[] = {"VK_KHR_portability_enumeration"};
    ici.flags = 0x00000001;         /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = instance_extensions;
#endif
    if (vkCreateInstance(&ici, NULL, &gpu.instance) != VK_SUCCESS) return false;
#define LOAD_INSTANCE(name) name = (PFN_##name)vkGetInstanceProcAddr(gpu.instance, #name); if (!name) return false;
    VK_INSTANCE_FUNCTIONS(LOAD_INSTANCE)
#undef LOAD_INSTANCE

    VkPhysicalDevice devices[16];
    uint32_t count = 16;
    if (vkEnumeratePhysicalDevices(gpu.instance, &count, devices) < 0 || count == 0) return false;
    const char *chosen = getenv("SPINGALETT_GPU_DEVICE");
    int best = -1, best_rank = 0;
    for (uint32_t d = 0; d < count; d++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[d], &props);
        if (props.apiVersion < VK_API_VERSION_1_2) continue;      /* before its 1.2 features may be asked for */
        VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
        vkGetPhysicalDeviceFeatures2(devices[d], &f);
        if (!f12.bufferDeviceAddress) continue;
        int rank = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4
                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? 1 : 2;
        if (chosen && *chosen) rank = (uint32_t)atoi(chosen) == d ? 5 : 0;
        if (rank > best_rank) { best_rank = rank; best = (int)d; }
    }
    if (best < 0) return false;
    gpu.physical = devices[best];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu.physical, &props);
    memcpy(gpu.name, props.deviceName, sizeof gpu.name);
    gpu.shared = props.limits.maxComputeSharedMemorySize;
    memcpy(gpu.max_groups, props.limits.maxComputeWorkGroupCount, sizeof gpu.max_groups);
    vkGetPhysicalDeviceMemoryProperties(gpu.physical, &gpu.memory);
    uint32_t largest = 0;
    for (uint32_t h = 0; h < gpu.memory.memoryHeapCount; h++)
        if ((gpu.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && gpu.memory.memoryHeaps[h].size > gpu.heap) {
            gpu.heap = gpu.memory.memoryHeaps[h].size;
            largest = h;
        }
    /* resizable BAR or unified memory: not the window of 256 MB into device memory some have */
    const VkMemoryPropertyFlags writable = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t t = 0; gpu.heap && t < gpu.memory.memoryTypeCount; t++)
        if (gpu.memory.memoryTypes[t].heapIndex == largest &&
            (gpu.memory.memoryTypes[t].propertyFlags & writable) == writable)
            gpu.host_writes = !getenv("SPINGALETT_GPU_NO_HOST_WRITES");

    VkQueueFamilyProperties families[32];
    uint32_t nf = 32;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &nf, families);
    gpu.family = UINT32_MAX;
    for (uint32_t q = 0; q < nf && gpu.family == UINT32_MAX; q++)
        if (families[q].queueFlags & VK_QUEUE_COMPUTE_BIT) gpu.family = q;
    if (gpu.family == UINT32_MAX) return false;
    gpu.stamp_bits = families[gpu.family].timestampValidBits;

    /* a portability implementation (MoltenVK) must have its subset extension enabled */
    VkExtensionProperties *extensions = NULL;
    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(gpu.physical, NULL, &extension_count, NULL) == VK_SUCCESS &&
        extension_count > 0) {
        extensions = (VkExtensionProperties *)malloc(extension_count * sizeof *extensions);
        if (!extensions || vkEnumerateDeviceExtensionProperties(gpu.physical, NULL, &extension_count, extensions) < 0)
            extension_count = 0;
    }
    const char *device_extensions[4];
    uint32_t enabled = 0;
    if (has_extension(extensions, extension_count, "VK_KHR_portability_subset"))
        device_extensions[enabled++] = "VK_KHR_portability_subset";
    /* the matrix units: bfloat16 cooperative matrices of 16 x 16 x 16 that add in float, in subgroups;
       their shader declares the Vulkan memory model and 16-bit floats, which must be enabled too */
    CoopFeatures coop = {.sType = COOP_FEATURES_TYPE};
    Bf16Features bf16 = {.sType = BF16_FEATURES_TYPE};
    VkPhysicalDeviceVulkan12Features have12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features have11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkBool32 int16 = VK_FALSE;
    /* (a build whose glslc could not compile the kernel holds an empty module for it) */
    if (spg_kernel_spirv_size[SPG_KERNEL_gemm_mma] > 20u &&
        has_extension(extensions, extension_count, "VK_KHR_cooperative_matrix") &&
        has_extension(extensions, extension_count, "VK_KHR_shader_bfloat16")) {
        coop.pNext = &bf16;
        bf16.pNext = &have12;
        have12.pNext = &have11;
        VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &coop};
        vkGetPhysicalDeviceFeatures2(gpu.physical, &f);
        bf16.pNext = NULL;
        have12.pNext = NULL;
        int16 = f.features.shaderInt16;
        CoopPropertiesFn properties = (CoopPropertiesFn)vkGetInstanceProcAddr(gpu.instance,
                                                                              "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        uint32_t count = 0;
        if (coop.cooperativeMatrix && bf16.type && bf16.cooperative_matrix && have12.vulkanMemoryModel &&
            have12.shaderFloat16 && properties &&
            properties(gpu.physical, &count, NULL) == VK_SUCCESS && count > 0) {
            CoopProperties *list = (CoopProperties *)calloc(count, sizeof *list);
            for (uint32_t k = 0; list && k < count; k++) list[k].sType = COOP_PROPERTIES_TYPE;
            if (list && properties(gpu.physical, &count, list) >= 0)
                for (uint32_t k = 0; k < count; k++)
                    if (list[k].M == 16 && list[k].N == 16 && list[k].K == 16 && list[k].a == COMPONENT_BFLOAT16 &&
                        list[k].b == COMPONENT_BFLOAT16 && list[k].c == COMPONENT_FLOAT32 &&
                        list[k].result == COMPONENT_FLOAT32 && list[k].scope == SCOPE_SUBGROUP)
                        gpu.mma_bf16 = true;
            free(list);
        }
        /* the kernel keeps bfloat16 values in memory too, and moves their bits as 16-bit integers (16-bit
           storage and integers, which every device with such matrix units has) */
        gpu.mma_bf16 = gpu.mma_bf16 && have11.storageBuffer16BitAccess && int16;
        gpu.storage16 = gpu.mma_bf16;
    }
    /* subgroups of the matrix units' kernel complete, as cooperative matrices need them
       (VK_EXT_subgroup_size_control, core in Vulkan 1.3) */
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT size_control = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
    if (gpu.mma_bf16 && has_extension(extensions, extension_count, "VK_EXT_subgroup_size_control")) {
        VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &size_control};
        vkGetPhysicalDeviceFeatures2(gpu.physical, &f);
        gpu.full_subgroups = size_control.computeFullSubgroups;
        size_control.subgroupSizeControl = VK_FALSE;
    }
    if (gpu.mma_bf16) {
        device_extensions[enabled++] = "VK_KHR_cooperative_matrix";
        device_extensions[enabled++] = "VK_KHR_shader_bfloat16";
        coop.cooperativeMatrixRobustBufferAccess = VK_FALSE;
        bf16.dot_product = VK_FALSE;
        if (gpu.full_subgroups) {
            device_extensions[enabled++] = "VK_EXT_subgroup_size_control";
            bf16.pNext = &size_control;
        }
    }
    free(extensions);
    VkPhysicalDeviceSubgroupProperties subgroup = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
    vkGetPhysicalDeviceProperties2(gpu.physical, &props2);
    gpu.subgroup = subgroup.subgroupSize ? subgroup.subgroupSize : 32u;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, gpu.family, 1, &priority};
    VkPhysicalDeviceVulkan11Features enable11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                                 .storageBuffer16BitAccess = gpu.storage16,
                                                 .pNext = gpu.mma_bf16 ? (void *)&coop : NULL};
    VkPhysicalDeviceVulkan12Features enable12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                 .bufferDeviceAddress = VK_TRUE,
                                                 .shaderFloat16 = gpu.mma_bf16, .vulkanMemoryModel = gpu.mma_bf16,
                                                 .pNext = &enable11};
    VkPhysicalDeviceFeatures2 enable = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &enable12};
    enable.features.shaderInt16 = gpu.storage16;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable, 0, 1, &qci, 0, NULL, enabled,
                              device_extensions, NULL};
    if (vkCreateDevice(gpu.physical, &dci, NULL, &gpu.device) != VK_SUCCESS) return false;
#define LOAD_DEVICE(name) name = (PFN_##name)vkGetDeviceProcAddr(gpu.device, #name); if (!name) return false;
    VK_DEVICE_FUNCTIONS(LOAD_DEVICE)
#undef LOAD_DEVICE
    vkGetDeviceQueue(gpu.device, gpu.family, 0, &gpu.queue);

    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, SPG_PUSH_BYTES};
    VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &range};
    if (vkCreatePipelineLayout(gpu.device, &lci, NULL, &gpu.layout) != VK_SUCCESS) return false;
    gpu.lock = spg_signal_create();
    gpu.tick_ns = props.limits.timestampPeriod;
    const char *profile = getenv("SPINGALETT_GPU_PROFILE");
    gpu.profile = profile && *profile && *profile != '0' && props.limits.timestampComputeAndGraphics;
    if (gpu.profile) atexit(profile_report);
    return gpu.lock != NULL;
}

bool spg_gpu_open(void) {
    int state = atomic_load(&gpu.state);
    if (state == 2) return true;
    if (state == 3) return false;
    int expected = 0;
    if (atomic_compare_exchange_strong(&gpu.state, &expected, 1)) {
        bool ok = open_device();
        atomic_store(&gpu.state, ok ? 2 : 3);
        return ok;
    }
    while ((state = atomic_load(&gpu.state)) == 1) {}       /* another thread is opening it */
    return state == 2;
}

const char *spg_gpu_device_name(void) {
    return spg_gpu_open() ? gpu.name : NULL;
}

uint64_t spg_gpu_memory(void) {
    return spg_gpu_open() ? gpu.heap : 0;
}

uint32_t spg_gpu_shared_memory(void) {
    return spg_gpu_open() ? gpu.shared : 0;
}

uint32_t spg_gpu_max_workgroups(uint32_t axis) {
    return spg_gpu_open() && axis < 3 ? gpu.max_groups[axis] : 0;
}

uint32_t spg_gpu_subgroup_size(void) {
    return spg_gpu_open() ? gpu.subgroup : 32u;
}

bool spg_gpu_mma_bf16(void) {
    return spg_gpu_open() && gpu.mma_bf16;
}

bool spg_gpu_bf16_storage(void) {
    static int off = -1;                /* SPINGALETT_GPU_NO_BF16_STORAGE=1: single precision in memory */
    if (off < 0) off = getenv("SPINGALETT_GPU_NO_BF16_STORAGE") != NULL;
    return spg_gpu_open() && gpu.storage16 && !off;
}

bool spg_gpu_host_writes(void) {
    return spg_gpu_open() && gpu.host_writes;
}

/* The pipeline of a kernel with these specialization constants, made on first use. */
static VkPipeline pipeline(SpgKernel kernel, const uint32_t *spec, uint32_t count) {
    VkPipeline found = VK_NULL_HANDLE;
    spg_lock(gpu.lock);
    for (size_t k = 0; k < gpu.pipeline_count && !found; k++) {
        const Pipeline *e = &gpu.pipelines[k];
        if (e->kernel == kernel && e->count == count && (!count || !memcmp(e->spec, spec, count * sizeof(uint32_t))))
            found = e->pipeline;
    }
    if (!found && gpu.pipeline_count == gpu.pipeline_cap) {
        size_t cap = gpu.pipeline_cap ? 2 * gpu.pipeline_cap : 64;
        Pipeline *grown = (Pipeline *)realloc(gpu.pipelines, cap * sizeof *grown);
        if (grown) { gpu.pipelines = grown; gpu.pipeline_cap = cap; }
    }
    if (!found && gpu.pipeline_count < gpu.pipeline_cap) {
        if (!gpu.modules[kernel]) {
            VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0,
                                            spg_kernel_spirv_size[kernel], spg_kernel_spirv[kernel]};
            if (vkCreateShaderModule(gpu.device, &mci, NULL, &gpu.modules[kernel]) != VK_SUCCESS)
                gpu.modules[kernel] = VK_NULL_HANDLE;
        }
        VkSpecializationMapEntry entries[SPG_SPEC_MAX];
        for (uint32_t k = 0; k < count; k++) entries[k] = (VkSpecializationMapEntry){k, 4u * k, 4u};
        VkSpecializationInfo si = {count, entries, count * sizeof(uint32_t), spec};
        VkComputePipelineCreateInfo cci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL,
                      kernel == SPG_KERNEL_gemm_mma && gpu.full_subgroups
                          ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT : 0u,
                      VK_SHADER_STAGE_COMPUTE_BIT, gpu.modules[kernel], "main", count ? &si : NULL},
            .layout = gpu.layout,
        };
        VkPipeline made;
        if (gpu.modules[kernel] && vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &cci, NULL, &made) == VK_SUCCESS) {
            Pipeline *e = &gpu.pipelines[gpu.pipeline_count++];
            e->kernel = kernel;
            e->count = count;
            if (count) memcpy(e->spec, spec, count * sizeof(uint32_t));
            e->pipeline = found = made;
        }
    }
    spg_unlock(gpu.lock);
    return found;
}

/* ------------------------------------------------------------------------- buffers */

static int32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < gpu.memory.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (gpu.memory.memoryTypes[i].propertyFlags & want) == want) return (int32_t)i;
    return -1;
}

/* A buffer with memory of its own of the given kind (SPG_MEMORY_HOST_WRITES: false without such). */
static bool create_buffer(SpgGpuBuffer *b, size_t bytes, SpgMemory kind) {
    memset(b, 0, sizeof *b);
    if (!spg_gpu_open()) return false;
    if (bytes == 0) bytes = 16;
    bytes = (bytes + 15u) & ~(size_t)15u;
    const bool host_visible = kind != SPG_MEMORY_DEVICE;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer buffer;
    if (vkCreateBuffer(gpu.device, &bci, NULL, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu.device, buffer, &req);
    /* host-visible memory is cached where the device has such, for reading results back */
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int32_t type = kind == SPG_MEMORY_HOST_WRITES ? memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                 : host_visible ? memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                                : memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0 && kind != SPG_MEMORY_HOST_WRITES) type = memory_type(req.memoryTypeBits, host_visible ? visible : 0);
    VkMemoryAllocateFlagsInfo flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL,
                                       VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags, req.size, (uint32_t)type};
    VkDeviceMemory memory;
    if (type < 0 || vkAllocateMemory(gpu.device, &mai, NULL, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(gpu.device, buffer, NULL);
        return false;
    }
    void *mapped = NULL;
    if (vkBindBufferMemory(gpu.device, buffer, memory, 0) != VK_SUCCESS ||
        (host_visible && vkMapMemory(gpu.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)) {
        vkDestroyBuffer(gpu.device, buffer, NULL);
        vkFreeMemory(gpu.device, memory, NULL);
        return false;
    }
    VkBufferDeviceAddressInfo ai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, buffer};
    b->buffer = (void *)buffer;
    b->memory = (void *)memory;
    b->address = vkGetBufferDeviceAddress(gpu.device, &ai);
    b->mapped = mapped;
    b->size = bytes;
    return true;
}

bool spg_gpu_buffer_create(SpgGpuBuffer *b, size_t bytes, bool host_visible) {
    return create_buffer(b, bytes, host_visible ? SPG_MEMORY_HOST : SPG_MEMORY_DEVICE);
}

void spg_gpu_buffer_free(SpgGpuBuffer *b) {
    if (!b || !b->buffer) return;
    if (b->memory) {                    /* an arena's buffers leave their memory to it */
        vkDestroyBuffer(gpu.device, (VkBuffer)b->buffer, NULL);
        vkFreeMemory(gpu.device, (VkDeviceMemory)b->memory, NULL);   /* unmaps it too */
    }
    memset(b, 0, sizeof *b);
}

/* ------------------------------------------------------------------------- arenas */

/* An arena's buffers are 256-byte ranges of blocks of up to ARENA_BLOCK bytes, one block per run of
   buffers of a kind in the order they were asked for (a larger buffer is a block of its own). */
#define ARENA_BLOCK (256ull << 20)
#define ARENA_ALIGN 256u

typedef struct { SpgGpuBuffer *buffer; size_t bytes, offset; SpgMemory kind; uint32_t block; } ArenaAsk;

struct SpgGpuArena {
    ArenaAsk *asks;
    uint32_t count, cap;
    SpgGpuBuffer *blocks;
    uint32_t block_count;
    bool failed;                        /* an ask could not be kept */
};

SpgGpuArena *spg_gpu_arena_create(void) {
    return (SpgGpuArena *)calloc(1, sizeof(SpgGpuArena));
}

void spg_gpu_arena_add(SpgGpuArena *a, SpgGpuBuffer *buffer, size_t bytes, SpgMemory kind) {
    memset(buffer, 0, sizeof *buffer);
    if (bytes == 0) return;
    if (a->count == a->cap) {
        uint32_t cap = a->cap ? 2u * a->cap : 64u;
        ArenaAsk *grown = (ArenaAsk *)realloc(a->asks, cap * sizeof *grown);
        if (!grown) { a->failed = true; return; }
        a->asks = grown;
        a->cap = cap;
    }
    a->asks[a->count++] = (ArenaAsk){buffer, (bytes + 15u) & ~(size_t)15u, 0, kind, 0};
}

bool spg_gpu_arena_commit(SpgGpuArena *a) {
    if (a->failed || !spg_gpu_open()) return false;
    /* the blocks (at most one per ask): their kinds and the bytes they need */
    const uint32_t most = a->count ? a->count : 1u;
    a->blocks = (SpgGpuBuffer *)calloc(most, sizeof(SpgGpuBuffer));
    size_t *need = (size_t *)calloc(most, sizeof(size_t));
    SpgMemory *kinds = (SpgMemory *)calloc(most, sizeof(SpgMemory));
    bool ok = a->blocks && need && kinds;
    for (int kind = SPG_MEMORY_DEVICE; ok && kind <= SPG_MEMORY_HOST_WRITES; kind++) {
        uint32_t open = UINT32_MAX;     /* the block being filled */
        for (uint32_t k = 0; k < a->count; k++) {
            ArenaAsk *ask = &a->asks[k];
            if ((int)ask->kind != kind) continue;
            const size_t bytes = (ask->bytes + ARENA_ALIGN - 1u) & ~(size_t)(ARENA_ALIGN - 1u);
            if (open == UINT32_MAX || need[open] + bytes > ARENA_BLOCK) {
                open = a->block_count++;
                kinds[open] = (SpgMemory)kind;
            }
            ask->block = open;
            ask->offset = need[open];
            need[open] += bytes;
        }
    }
    uint32_t made = 0;
    while (ok && made < a->block_count) {
        ok = create_buffer(&a->blocks[made], need[made], kinds[made]);
        made += ok;
    }
    a->block_count = made;              /* the blocks to free */
    free(need);
    free(kinds);
    if (!ok) return false;
    for (uint32_t k = 0; k < a->count; k++) {
        const ArenaAsk *ask = &a->asks[k];
        const SpgGpuBuffer *block = &a->blocks[ask->block];
        *ask->buffer = (SpgGpuBuffer){block->buffer, NULL, block->address + ask->offset,
                                      block->mapped ? (char *)block->mapped + ask->offset : NULL, ask->bytes,
                                      ask->offset};
    }
    return true;
}

void spg_gpu_arena_free(SpgGpuArena *a) {
    if (!a) return;
    for (uint32_t b = 0; b < a->block_count; b++) spg_gpu_buffer_free(&a->blocks[b]);
    free(a->blocks);
    free(a->asks);
    free(a);
}

/* ------------------------------------------------------------------------- commands */

/* Each command buffer has a pool of its own: recording into buffers of one pool needs the pool
   synchronized, and networks may record on several threads at once. */
struct SpgGpuCommands {
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    bool pending;                   /* submitted and not waited for */
    bool ok;                        /* every dispatch found its pipeline */
    VkQueryPool queries;            /* profiling: two timestamps per dispatch */
    uint32_t timed;
    char (*labels)[48];
    VkQueryPool stamps;             /* spg_gpu_commands_stamps() */
    uint32_t stamp_count;
};

SpgGpuCommands *spg_gpu_commands_create(void) {
    if (!spg_gpu_open()) return NULL;
    /* a command buffer freed before, which a fresh one would take a third of a millisecond to make (its
       pool, buffer and fence, on NVIDIA's driver): recorded again from the start, as a fresh one */
    spg_lock(gpu.lock);
    SpgGpuCommands *c = gpu.spare_count ? gpu.spare[--gpu.spare_count] : NULL;
    spg_unlock(gpu.lock);
    if (c) return c;
    c = (SpgGpuCommands *)calloc(1, sizeof *c);
    if (!c) return NULL;
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                   VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, gpu.family};
    bool ok = vkCreateCommandPool(gpu.device, &pci, NULL, &c->pool) == VK_SUCCESS;
    if (!ok) c->pool = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, c->pool,
                                      VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    ok = ok && vkAllocateCommandBuffers(gpu.device, &ai, &c->cb) == VK_SUCCESS;
    if (!ok || vkCreateFence(gpu.device, &fci, NULL, &c->fence) != VK_SUCCESS) {
        spg_gpu_commands_free(c);
        return NULL;
    }
    if (gpu.profile) {
        VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP,
                                     2u * PROFILE_QUERIES, 0};
        c->labels = (char (*)[48])calloc(PROFILE_QUERIES, sizeof *c->labels);
        if (!c->labels || vkCreateQueryPool(gpu.device, &qci, NULL, &c->queries) != VK_SUCCESS) c->queries = VK_NULL_HANDLE;
    }
    return c;
}

void spg_gpu_commands_untimed(SpgGpuCommands *c) {
    if (c->queries) vkDestroyQueryPool(gpu.device, c->queries, NULL);
    c->queries = VK_NULL_HANDLE;
}

bool spg_gpu_commands_stamps(SpgGpuCommands *c, uint32_t count) {
    if (gpu.stamp_bits == 0 || gpu.tick_ns <= 0.0f || count == 0) return false;
    VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, count, 0};
    if (vkCreateQueryPool(gpu.device, &qci, NULL, &c->stamps) != VK_SUCCESS) {
        c->stamps = VK_NULL_HANDLE;
        return false;
    }
    c->stamp_count = count;
    return true;
}

void spg_gpu_timestamp(SpgGpuCommands *c, uint32_t index) {
    if (c->stamps && index < c->stamp_count)
        vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->stamps, index);
}

bool spg_gpu_timestamps(SpgGpuCommands *c, double *ns, uint32_t count) {
    if (!c->stamps || count > c->stamp_count) return false;
    uint64_t *ticks = (uint64_t *)malloc(count * sizeof(uint64_t));
    bool ok = ticks && vkGetQueryPoolResults(gpu.device, c->stamps, 0, count, count * sizeof(uint64_t), ticks,
                                             sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
    for (uint32_t k = 0; ok && k < count; k++) ns[k] = (double)ticks[k] * gpu.tick_ns;
    free(ticks);
    return ok;
}

void spg_gpu_commands_free(SpgGpuCommands *c) {
    if (!c) return;
    if (c->pending) spg_gpu_wait(c);
    if (c->stamps) vkDestroyQueryPool(gpu.device, c->stamps, NULL);
    c->stamps = VK_NULL_HANDLE;
    c->stamp_count = 0;
    /* kept for the next spg_gpu_commands_create() while there is room (with its profiling queries) */
    if (c->cb && c->fence && (c->queries || !gpu.profile)) {
        spg_lock(gpu.lock);
        const bool kept = gpu.spare_count < SPARE_COMMANDS;
        if (kept) gpu.spare[gpu.spare_count++] = c;
        spg_unlock(gpu.lock);
        if (kept) return;
    }
    if (c->fence) vkDestroyFence(gpu.device, c->fence, NULL);
    if (c->queries) vkDestroyQueryPool(gpu.device, c->queries, NULL);
    free(c->labels);
    if (c->pool) vkDestroyCommandPool(gpu.device, c->pool, NULL);       /* its buffer with it */
    free(c);
}

bool spg_gpu_record_begin(SpgGpuCommands *c) {
    if (c->pending && !spg_gpu_wait(c)) return false;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0, NULL};
    c->ok = vkBeginCommandBuffer(c->cb, &bi) == VK_SUCCESS;
    c->timed = 0;
    if (c->ok && c->queries) vkCmdResetQueryPool(c->cb, c->queries, 0, 2u * PROFILE_QUERIES);
    if (c->ok && c->stamps) vkCmdResetQueryPool(c->cb, c->stamps, 0, c->stamp_count);
    return c->ok;
}

void spg_gpu_dispatch(SpgGpuCommands *c, SpgKernel kernel, const uint32_t *spec, uint32_t spec_count, const void *push,
                      uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    VkPipeline p = spec_count <= SPG_SPEC_MAX && push_size <= SPG_PUSH_BYTES ? pipeline(kernel, spec, spec_count)
                                                                              : VK_NULL_HANDLE;
    if (!p) { c->ok = false; return; }
    vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    if (push_size) vkCmdPushConstants(c->cb, gpu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    const bool timed = c->queries && c->timed < PROFILE_QUERIES;
    if (timed) {
        /* the kernel and its modes: for gemm the operand modes and tile, for the others the first constants */
        char *label = c->labels[c->timed];
        int len = snprintf(label, 48, "%s", spg_kernel_names[kernel]);
        if ((kernel == SPG_KERNEL_gemm || kernel == SPG_KERNEL_gemm_mma) && spec_count >= 8)
            snprintf(label + len, 48u - (size_t)len, " A%u B%u E%u %ux%u", spec[5], spec[6], spec[7], spec[0], spec[1]);
        else
            for (uint32_t k = 0; k < spec_count && k < 2 && len < 40; k++)
                len += snprintf(label + len, 48u - (size_t)len, " %u", spec[k]);
        vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c->queries, 2u * c->timed);
    }
    vkCmdDispatch(c->cb, gx, gy, gz);
    if (timed) vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->queries, 2u * c->timed++ + 1u);
}

void spg_gpu_barrier(SpgGpuCommands *c) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

void spg_gpu_barrier_host(SpgGpuCommands *c) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

void spg_gpu_copy(SpgGpuCommands *c, const SpgGpuBuffer *src, size_t src_offset, const SpgGpuBuffer *dst,
                  size_t dst_offset, size_t bytes) {
    if (bytes == 0) return;
    VkBufferCopy region = {src->offset + src_offset, dst->offset + dst_offset, bytes};
    const bool timed = c->queries && c->timed < PROFILE_QUERIES && bytes >= 4096u;
    if (timed) {
        snprintf(c->labels[c->timed], 48, "copy%s", src->mapped ? " from host" : dst->mapped ? " to host" : "");
        vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c->queries, 2u * c->timed);
    }
    vkCmdCopyBuffer(c->cb, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &region);
    if (timed) vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->queries, 2u * c->timed++ + 1u);
}

void spg_gpu_fill(SpgGpuCommands *c, const SpgGpuBuffer *dst, size_t offset, size_t bytes, uint32_t value) {
    if (bytes == 0) return;
    vkCmdFillBuffer(c->cb, (VkBuffer)dst->buffer, dst->offset + offset, bytes, value);
}

bool spg_gpu_record_end(SpgGpuCommands *c) {
    return vkEndCommandBuffer(c->cb) == VK_SUCCESS && c->ok;
}

bool spg_gpu_submit(SpgGpuCommands *c) {
    if (c->pending && !spg_gpu_wait(c)) return false;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &c->cb, 0, NULL};
    spg_lock(gpu.lock);
    bool ok = vkResetFences(gpu.device, 1, &c->fence) == VK_SUCCESS &&
              vkQueueSubmit(gpu.queue, 1, &si, c->fence) == VK_SUCCESS;
    spg_unlock(gpu.lock);
    c->pending = ok;
    return ok;
}

bool spg_gpu_wait(SpgGpuCommands *c) {
    if (!c->pending) return true;
    c->pending = false;
    bool ok = vkWaitForFences(gpu.device, 1, &c->fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (ok && c->queries && c->timed) {
        uint64_t *ticks = (uint64_t *)malloc(2u * c->timed * sizeof(uint64_t));
        if (ticks && vkGetQueryPoolResults(gpu.device, c->queries, 0, 2u * c->timed, 2u * c->timed * sizeof(uint64_t),
                                           ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            spg_lock(gpu.lock);
            for (uint32_t k = 0; k < c->timed; k++)
                profile_add(c->labels[k], (double)(ticks[2u * k + 1] - ticks[2u * k]) * gpu.tick_ns * 1e-6);
            gpu.busy_ms += (double)(ticks[2u * c->timed - 1u] - ticks[0]) * gpu.tick_ns * 1e-6;
            gpu.submissions++;
            spg_unlock(gpu.lock);
        }
        free(ticks);
    }
    return ok;
}
