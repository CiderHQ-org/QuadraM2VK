// Generated from Vulkan-Headers vulkan-sdk-1.4.341.0/registry/vk.xml.
// Registry SHA-256: 006807a47518e9ab23ecf7f4c95d936aab477a20d27a9a7ac23cedefb7210178
// Regenerate with scripts/regen-vulkan-requirements/registry.py; do not edit.
//
// vk.xml is licensed under Apache-2.0 OR MIT; this table uses it under the MIT license:
//
// Copyright 2015-2026 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Exposure {
    Core(u8, u8),
    Extension(&'static str),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Enablement {
    Exposed(Exposure),
    Feature(&'static str),
    Property(&'static str),
}
pub(crate) fn feature_exposure(feature: &str) -> Option<&'static [Exposure]> {
    Some(match feature {
        "AttachmentFragmentShadingRate" => &[Exposure::Extension("KhrFragmentShadingRate")],
        "BufferDeviceAddress" => &[
            Exposure::Extension("KhrBufferDeviceAddress"),
            Exposure::Core(1, 2),
        ],
        "BufferDeviceAddressEXT" => &[Exposure::Extension("ExtBufferDeviceAddress")],
        "ClusterAccelerationStructure" => &[Exposure::Extension("NvClusterAccelerationStructure")],
        "ClustercullingShader" => &[Exposure::Extension("HuaweiClusterCullingShader")],
        "ComputeDerivativeGroupLinear" => &[
            Exposure::Extension("KhrComputeShaderDerivatives"),
            Exposure::Extension("NvComputeShaderDerivatives"),
        ],
        "ComputeDerivativeGroupQuads" => &[
            Exposure::Extension("KhrComputeShaderDerivatives"),
            Exposure::Extension("NvComputeShaderDerivatives"),
        ],
        "CooperativeMatrix" => &[Exposure::Extension("KhrCooperativeMatrix")],
        "CooperativeMatrixBlockLoads" => &[Exposure::Extension("NvCooperativeMatrix2")],
        "CooperativeMatrixConversions" => &[Exposure::Extension("NvCooperativeMatrix2")],
        "CooperativeMatrixNV" => &[Exposure::Extension("NvCooperativeMatrix")],
        "CooperativeMatrixPerElementOperations" => &[Exposure::Extension("NvCooperativeMatrix2")],
        "CooperativeMatrixReductions" => &[Exposure::Extension("NvCooperativeMatrix2")],
        "CooperativeMatrixTensorAddressing" => &[Exposure::Extension("NvCooperativeMatrix2")],
        "CooperativeVector" => &[Exposure::Extension("NvCooperativeVector")],
        "CooperativeVectorTraining" => &[Exposure::Extension("NvCooperativeVector")],
        "DataGraph" => &[Exposure::Extension("ArmDataGraph")],
        "DescriptorHeap" => &[Exposure::Extension("ExtDescriptorHeap")],
        "FragmentDensityMap" => &[Exposure::Extension("ExtFragmentDensityMap")],
        "FragmentShaderBarycentric" => &[
            Exposure::Extension("KhrFragmentShaderBarycentric"),
            Exposure::Extension("NvFragmentShaderBarycentric"),
        ],
        "FragmentShaderPixelInterlock" => &[Exposure::Extension("ExtFragmentShaderInterlock")],
        "FragmentShaderSampleInterlock" => &[Exposure::Extension("ExtFragmentShaderInterlock")],
        "FragmentShaderShadingRateInterlock" => {
            &[Exposure::Extension("ExtFragmentShaderInterlock")]
        }
        "FragmentStoresAndAtomics" => &[Exposure::Core(1, 0)],
        "GeometryShader" => &[Exposure::Core(1, 0)],
        "GeometryStreams" => &[Exposure::Extension("ExtTransformFeedback")],
        "ImageCubeArray" => &[Exposure::Core(1, 0)],
        "ImageFootprint" => &[Exposure::Extension("NvShaderImageFootprint")],
        "LinearSweptSpheres" => &[Exposure::Extension("NvRayTracingLinearSweptSpheres")],
        "LongVector" => &[Exposure::Extension("ExtShaderLongVector")],
        "MultiViewport" => &[Exposure::Core(1, 0)],
        "Multiview" => &[Exposure::Extension("KhrMultiview"), Exposure::Core(1, 2)],
        "PipelineFragmentShadingRate" => &[Exposure::Extension("KhrFragmentShadingRate")],
        "PrimitiveFragmentShadingRate" => &[Exposure::Extension("KhrFragmentShadingRate")],
        "PushConstantBank" => &[Exposure::Extension("NvPushConstantBank")],
        "RayQuery" => &[Exposure::Extension("KhrRayQuery")],
        "RayTracingMaintenance1" => &[Exposure::Extension("KhrRayTracingMaintenance1")],
        "RayTracingMotionBlur" => &[Exposure::Extension("NvRayTracingMotionBlur")],
        "RayTracingPipeline" => &[Exposure::Extension("KhrRayTracingPipeline")],
        "RayTracingPositionFetch" => &[Exposure::Extension("KhrRayTracingPositionFetch")],
        "RayTraversalPrimitiveCulling" => &[Exposure::Extension("KhrRayTracingPipeline")],
        "RuntimeDescriptorArray" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "SampleRateShading" => &[Exposure::Core(1, 0)],
        "Shader64BitIndexing" => &[Exposure::Extension("ExtShader64BitIndexing")],
        "ShaderBFloat16CooperativeMatrix" => &[Exposure::Extension("KhrShaderBfloat16")],
        "ShaderBFloat16DotProduct" => &[Exposure::Extension("KhrShaderBfloat16")],
        "ShaderBFloat16Type" => &[Exposure::Extension("KhrShaderBfloat16")],
        "ShaderBufferFloat16AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderBufferFloat16AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderBufferFloat16Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderBufferFloat32AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderBufferFloat32AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderBufferFloat32Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderBufferFloat64AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderBufferFloat64AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderBufferFloat64Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderBufferInt64Atomics" => &[
            Exposure::Extension("KhrShaderAtomicInt64"),
            Exposure::Core(1, 2),
        ],
        "ShaderClipDistance" => &[Exposure::Core(1, 0)],
        "ShaderCoreBuiltins" => &[Exposure::Extension("ArmShaderCoreBuiltins")],
        "ShaderCullDistance" => &[Exposure::Core(1, 0)],
        "ShaderDemoteToHelperInvocation" => &[
            Exposure::Extension("ExtShaderDemoteToHelperInvocation"),
            Exposure::Core(1, 3),
        ],
        "ShaderDrawParameters" => &[Exposure::Core(1, 1), Exposure::Core(1, 2)],
        "ShaderEnqueue" => &[Exposure::Extension("AmdxShaderEnqueue")],
        "ShaderExpectAssume" => &[
            Exposure::Extension("KhrShaderExpectAssume"),
            Exposure::Core(1, 4),
        ],
        "ShaderFloat16" => &[
            Exposure::Extension("KhrShaderFloat16Int8"),
            Exposure::Core(1, 2),
        ],
        "ShaderFloat16VectorAtomics" => &[Exposure::Extension("NvShaderAtomicFloat16Vector")],
        "ShaderFloat64" => &[Exposure::Core(1, 0)],
        "ShaderFloat8" => &[Exposure::Extension("ExtShaderFloat8")],
        "ShaderFloat8CooperativeMatrix" => &[Exposure::Extension("ExtShaderFloat8")],
        "ShaderFloatControls2" => &[
            Exposure::Extension("KhrShaderFloatControls2"),
            Exposure::Core(1, 4),
        ],
        "ShaderFmaFloat16" => &[Exposure::Extension("KhrShaderFma")],
        "ShaderFmaFloat32" => &[Exposure::Extension("KhrShaderFma")],
        "ShaderFmaFloat64" => &[Exposure::Extension("KhrShaderFma")],
        "ShaderImageFloat32AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderImageFloat32AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderImageFloat32Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderImageGatherExtended" => &[Exposure::Core(1, 0)],
        "ShaderImageInt64Atomics" => &[Exposure::Extension("ExtShaderImageAtomicInt64")],
        "ShaderInputAttachmentArrayDynamicIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderInputAttachmentArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderInt16" => &[Exposure::Core(1, 0)],
        "ShaderInt64" => &[Exposure::Core(1, 0)],
        "ShaderInt8" => &[
            Exposure::Extension("KhrShaderFloat16Int8"),
            Exposure::Core(1, 2),
        ],
        "ShaderIntegerDotProduct" => &[
            Exposure::Extension("KhrShaderIntegerDotProduct"),
            Exposure::Core(1, 3),
        ],
        "ShaderIntegerFunctions2" => &[Exposure::Extension("IntelShaderIntegerFunctions2")],
        "ShaderOutputLayer" => &[Exposure::Core(1, 2)],
        "ShaderOutputViewportIndex" => &[Exposure::Core(1, 2)],
        "ShaderQuadControl" => &[Exposure::Extension("KhrShaderQuadControl")],
        "ShaderRawAccessChains" => &[Exposure::Extension("NvRawAccessChains")],
        "ShaderReplicatedComposites" => &[Exposure::Extension("ExtShaderReplicatedComposites")],
        "ShaderResourceMinLod" => &[Exposure::Core(1, 0)],
        "ShaderResourceResidency" => &[Exposure::Core(1, 0)],
        "ShaderSMBuiltins" => &[Exposure::Extension("NvShaderSmBuiltins")],
        "ShaderSampledImageArrayDynamicIndexing" => &[Exposure::Core(1, 0)],
        "ShaderSampledImageArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderSharedFloat16AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderSharedFloat16AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderSharedFloat16Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderSharedFloat32AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderSharedFloat32AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderSharedFloat32Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderSharedFloat64AtomicAdd" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderSharedFloat64AtomicMinMax" => &[Exposure::Extension("ExtShaderAtomicFloat2")],
        "ShaderSharedFloat64Atomics" => &[Exposure::Extension("ExtShaderAtomicFloat")],
        "ShaderSharedInt64Atomics" => &[
            Exposure::Extension("KhrShaderAtomicInt64"),
            Exposure::Core(1, 2),
        ],
        "ShaderStorageBufferArrayDynamicIndexing" => &[Exposure::Core(1, 0)],
        "ShaderStorageBufferArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderStorageImageArrayDynamicIndexing" => &[Exposure::Core(1, 0)],
        "ShaderStorageImageArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderStorageImageMultisample" => &[Exposure::Core(1, 0)],
        "ShaderStorageImageReadWithoutFormat" => &[Exposure::Core(1, 0)],
        "ShaderStorageImageWriteWithoutFormat" => &[Exposure::Core(1, 0)],
        "ShaderStorageTensorArrayDynamicIndexing" => &[Exposure::Extension("ArmTensors")],
        "ShaderStorageTensorArrayNonUniformIndexing" => &[Exposure::Extension("ArmTensors")],
        "ShaderStorageTexelBufferArrayDynamicIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderStorageTexelBufferArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderSubgroupPartitioned" => &[Exposure::Extension("ExtShaderSubgroupPartitioned")],
        "ShaderSubgroupRotate" => &[
            Exposure::Extension("KhrShaderSubgroupRotate"),
            Exposure::Core(1, 4),
        ],
        "ShaderTensorAccess" => &[Exposure::Extension("ArmTensors")],
        "ShaderTessellationAndGeometryPointSize" => &[Exposure::Core(1, 0)],
        "ShaderTileImageColorReadAccess" => &[Exposure::Extension("ExtShaderTileImage")],
        "ShaderTileImageDepthReadAccess" => &[Exposure::Extension("ExtShaderTileImage")],
        "ShaderTileImageStencilReadAccess" => &[Exposure::Extension("ExtShaderTileImage")],
        "ShaderUniformBufferArrayDynamicIndexing" => &[Exposure::Core(1, 0)],
        "ShaderUniformBufferArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderUniformTexelBufferArrayDynamicIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderUniformTexelBufferArrayNonUniformIndexing" => &[
            Exposure::Extension("ExtDescriptorIndexing"),
            Exposure::Core(1, 2),
        ],
        "ShaderUntypedPointers" => &[Exposure::Extension("KhrShaderUntypedPointers")],
        "ShadingRateImage" => &[Exposure::Extension("NvShadingRateImage")],
        "Spheres" => &[Exposure::Extension("NvRayTracingLinearSweptSpheres")],
        "StorageBuffer16BitAccess" => {
            &[Exposure::Extension("Khr16BitStorage"), Exposure::Core(1, 2)]
        }
        "StorageBuffer8BitAccess" => &[Exposure::Extension("Khr8BitStorage"), Exposure::Core(1, 2)],
        "StorageInputOutput16" => &[Exposure::Extension("Khr16BitStorage"), Exposure::Core(1, 2)],
        "StoragePushConstant16" => &[Exposure::Extension("Khr16BitStorage"), Exposure::Core(1, 2)],
        "StoragePushConstant8" => &[Exposure::Extension("Khr8BitStorage"), Exposure::Core(1, 2)],
        "TessellationShader" => &[Exposure::Core(1, 0)],
        "TextureBlockMatch" => &[Exposure::Extension("QcomImageProcessing")],
        "TextureBlockMatch2" => &[Exposure::Extension("QcomImageProcessing2")],
        "TextureBoxFilter" => &[Exposure::Extension("QcomImageProcessing")],
        "TextureSampleWeighted" => &[Exposure::Extension("QcomImageProcessing")],
        "TileShading" => &[Exposure::Extension("QcomTileShading")],
        "TransformFeedback" => &[Exposure::Extension("ExtTransformFeedback")],
        "UniformAndStorageBuffer16BitAccess" => {
            &[Exposure::Extension("Khr16BitStorage"), Exposure::Core(1, 2)]
        }
        "UniformAndStorageBuffer8BitAccess" => {
            &[Exposure::Extension("Khr8BitStorage"), Exposure::Core(1, 2)]
        }
        "VariablePointers" => &[
            Exposure::Extension("KhrVariablePointers"),
            Exposure::Core(1, 2),
        ],
        "VariablePointersStorageBuffer" => &[
            Exposure::Extension("KhrVariablePointers"),
            Exposure::Core(1, 2),
        ],
        "VertexPipelineStoresAndAtomics" => &[Exposure::Core(1, 0)],
        "VulkanMemoryModel" => &[
            Exposure::Extension("KhrVulkanMemoryModel"),
            Exposure::Core(1, 2),
        ],
        "VulkanMemoryModelDeviceScope" => &[
            Exposure::Extension("KhrVulkanMemoryModel"),
            Exposure::Core(1, 2),
        ],
        "WorkgroupMemoryExplicitLayout" => {
            &[Exposure::Extension("KhrWorkgroupMemoryExplicitLayout")]
        }
        "WorkgroupMemoryExplicitLayout16BitAccess" => {
            &[Exposure::Extension("KhrWorkgroupMemoryExplicitLayout")]
        }
        "WorkgroupMemoryExplicitLayout8BitAccess" => {
            &[Exposure::Extension("KhrWorkgroupMemoryExplicitLayout")]
        }
        _ => return None,
    })
}
pub(crate) fn capability_enablement(name: &str) -> Option<&'static [Enablement]> {
    Some(match name {
        "AtomicFloat16AddEXT" => &[
            Enablement::Feature("ShaderBufferFloat16AtomicAdd"),
            Enablement::Feature("ShaderSharedFloat16AtomicAdd"),
        ],
        "AtomicFloat16MinMaxEXT" => &[
            Enablement::Feature("ShaderBufferFloat16AtomicMinMax"),
            Enablement::Feature("ShaderSharedFloat16AtomicMinMax"),
        ],
        "AtomicFloat16VectorNV" => &[Enablement::Feature("ShaderFloat16VectorAtomics")],
        "AtomicFloat32AddEXT" => &[
            Enablement::Feature("ShaderBufferFloat32AtomicAdd"),
            Enablement::Feature("ShaderSharedFloat32AtomicAdd"),
            Enablement::Feature("ShaderImageFloat32AtomicAdd"),
        ],
        "AtomicFloat32MinMaxEXT" => &[
            Enablement::Feature("ShaderBufferFloat32AtomicMinMax"),
            Enablement::Feature("ShaderSharedFloat32AtomicMinMax"),
            Enablement::Feature("ShaderImageFloat32AtomicMinMax"),
        ],
        "AtomicFloat64AddEXT" => &[
            Enablement::Feature("ShaderBufferFloat64AtomicAdd"),
            Enablement::Feature("ShaderSharedFloat64AtomicAdd"),
        ],
        "AtomicFloat64MinMaxEXT" => &[
            Enablement::Feature("ShaderBufferFloat64AtomicMinMax"),
            Enablement::Feature("ShaderSharedFloat64AtomicMinMax"),
        ],
        "BFloat16CooperativeMatrixKHR" => &[Enablement::Feature("ShaderBFloat16CooperativeMatrix")],
        "BFloat16DotProductKHR" => &[Enablement::Feature("ShaderBFloat16DotProduct")],
        "BFloat16TypeKHR" => &[Enablement::Feature("ShaderBFloat16Type")],
        "ClipDistance" => &[Enablement::Feature("ShaderClipDistance")],
        "ClusterCullingShadingHUAWEI" => &[Enablement::Feature("ClustercullingShader")],
        "ComputeDerivativeGroupLinearKHR" => &[
            Enablement::Feature("ComputeDerivativeGroupLinear"),
            Enablement::Feature("ComputeDerivativeGroupLinear"),
        ],
        "ComputeDerivativeGroupQuadsKHR" => &[
            Enablement::Feature("ComputeDerivativeGroupQuads"),
            Enablement::Feature("ComputeDerivativeGroupQuads"),
        ],
        "CooperativeMatrixBlockLoadsNV" => &[Enablement::Feature("CooperativeMatrixBlockLoads")],
        "CooperativeMatrixConversionsNV" => &[Enablement::Feature("CooperativeMatrixConversions")],
        "CooperativeMatrixKHR" => &[Enablement::Feature("CooperativeMatrix")],
        "CooperativeMatrixNV" => &[Enablement::Feature("CooperativeMatrixNV")],
        "CooperativeMatrixPerElementOperationsNV" => {
            &[Enablement::Feature("CooperativeMatrixPerElementOperations")]
        }
        "CooperativeMatrixReductionsNV" => &[Enablement::Feature("CooperativeMatrixReductions")],
        "CooperativeMatrixTensorAddressingNV" => {
            &[Enablement::Feature("CooperativeMatrixTensorAddressing")]
        }
        "CooperativeVectorNV" => &[Enablement::Feature("CooperativeVector")],
        "CooperativeVectorTrainingNV" => &[Enablement::Feature("CooperativeVectorTraining")],
        "CoreBuiltinsARM" => &[Enablement::Feature("ShaderCoreBuiltins")],
        "CullDistance" => &[Enablement::Feature("ShaderCullDistance")],
        "DemoteToHelperInvocation" => &[
            Enablement::Feature("ShaderDemoteToHelperInvocation"),
            Enablement::Feature("ShaderDemoteToHelperInvocation"),
        ],
        "DenormFlushToZero" => &[
            Enablement::Property("ShaderDenormFlushToZeroFloat16"),
            Enablement::Property("ShaderDenormFlushToZeroFloat32"),
            Enablement::Property("ShaderDenormFlushToZeroFloat64"),
        ],
        "DenormPreserve" => &[
            Enablement::Property("ShaderDenormPreserveFloat16"),
            Enablement::Property("ShaderDenormPreserveFloat32"),
            Enablement::Property("ShaderDenormPreserveFloat64"),
        ],
        "DerivativeControl" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "DescriptorHeapEXT" => &[Enablement::Feature("DescriptorHeap")],
        "DeviceGroup" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("KhrDeviceGroup")),
        ],
        "DotProduct" => &[
            Enablement::Feature("ShaderIntegerDotProduct"),
            Enablement::Feature("ShaderIntegerDotProduct"),
        ],
        "DotProductInput4x8Bit" => &[
            Enablement::Feature("ShaderIntegerDotProduct"),
            Enablement::Feature("ShaderIntegerDotProduct"),
        ],
        "DotProductInput4x8BitPacked" => &[
            Enablement::Feature("ShaderIntegerDotProduct"),
            Enablement::Feature("ShaderIntegerDotProduct"),
        ],
        "DotProductInputAll" => &[
            Enablement::Feature("ShaderIntegerDotProduct"),
            Enablement::Feature("ShaderIntegerDotProduct"),
        ],
        "DrawParameters" => &[
            Enablement::Feature("ShaderDrawParameters"),
            Enablement::Feature("ShaderDrawParameters"),
            Enablement::Exposed(Exposure::Extension("KhrShaderDrawParameters")),
        ],
        "ExpectAssumeKHR" => &[
            Enablement::Feature("ShaderExpectAssume"),
            Enablement::Feature("ShaderExpectAssume"),
        ],
        "FMAKHR" => &[
            Enablement::Feature("ShaderFmaFloat16"),
            Enablement::Feature("ShaderFmaFloat32"),
            Enablement::Feature("ShaderFmaFloat64"),
        ],
        "Float16" => &[
            Enablement::Feature("ShaderFloat16"),
            Enablement::Exposed(Exposure::Extension("AmdGpuShaderHalfFloat")),
        ],
        "Float64" => &[Enablement::Feature("ShaderFloat64")],
        "Float8CooperativeMatrixEXT" => &[Enablement::Feature("ShaderFloat8CooperativeMatrix")],
        "Float8EXT" => &[Enablement::Feature("ShaderFloat8")],
        "FloatControls2" => &[
            Enablement::Feature("ShaderFloatControls2"),
            Enablement::Feature("ShaderFloatControls2"),
        ],
        "FragmentBarycentricKHR" => &[
            Enablement::Feature("FragmentShaderBarycentric"),
            Enablement::Feature("FragmentShaderBarycentric"),
        ],
        "FragmentDensityEXT" => &[
            Enablement::Feature("FragmentDensityMap"),
            Enablement::Feature("ShadingRateImage"),
        ],
        "FragmentFullyCoveredEXT" => &[Enablement::Exposed(Exposure::Extension(
            "ExtConservativeRasterization",
        ))],
        "FragmentMaskAMD" => &[Enablement::Exposed(Exposure::Extension(
            "AmdShaderFragmentMask",
        ))],
        "FragmentShaderPixelInterlockEXT" => &[Enablement::Feature("FragmentShaderPixelInterlock")],
        "FragmentShaderSampleInterlockEXT" => {
            &[Enablement::Feature("FragmentShaderSampleInterlock")]
        }
        "FragmentShaderShadingRateInterlockEXT" => &[
            Enablement::Feature("FragmentShaderShadingRateInterlock"),
            Enablement::Feature("ShadingRateImage"),
        ],
        "FragmentShadingRateKHR" => &[
            Enablement::Feature("PipelineFragmentShadingRate"),
            Enablement::Feature("PrimitiveFragmentShadingRate"),
            Enablement::Feature("AttachmentFragmentShadingRate"),
        ],
        "Geometry" => &[Enablement::Feature("GeometryShader")],
        "GeometryPointSize" => &[Enablement::Feature(
            "ShaderTessellationAndGeometryPointSize",
        )],
        "GeometryShaderPassthroughNV" => &[Enablement::Exposed(Exposure::Extension(
            "NvGeometryShaderPassthrough",
        ))],
        "GeometryStreams" => &[Enablement::Feature("GeometryStreams")],
        "GraphARM" => &[Enablement::Feature("DataGraph")],
        "GroupNonUniform" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureBasicBit",
        )],
        "GroupNonUniformArithmetic" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureArithmeticBit",
        )],
        "GroupNonUniformBallot" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureBallotBit",
        )],
        "GroupNonUniformClustered" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureClusteredBit",
        )],
        "GroupNonUniformPartitionedEXT" => &[
            Enablement::Feature("ShaderSubgroupPartitioned"),
            Enablement::Property("SubgroupSupportedOperationsSubgroupFeaturePartitionedBitNv"),
        ],
        "GroupNonUniformQuad" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureQuadBit",
        )],
        "GroupNonUniformRotateKHR" => &[
            Enablement::Feature("ShaderSubgroupRotate"),
            Enablement::Feature("ShaderSubgroupRotate"),
        ],
        "GroupNonUniformShuffle" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureShuffleBit",
        )],
        "GroupNonUniformShuffleRelative" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureShuffleRelativeBit",
        )],
        "GroupNonUniformVote" => &[Enablement::Property(
            "SubgroupSupportedOperationsSubgroupFeatureVoteBit",
        )],
        "Image1D" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "ImageBuffer" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "ImageCubeArray" => &[Enablement::Feature("ImageCubeArray")],
        "ImageFootprintNV" => &[Enablement::Feature("ImageFootprint")],
        "ImageGatherBiasLodAMD" => &[Enablement::Exposed(Exposure::Extension(
            "AmdTextureGatherBiasLod",
        ))],
        "ImageGatherExtended" => &[Enablement::Feature("ShaderImageGatherExtended")],
        "ImageMSArray" => &[Enablement::Feature("ShaderStorageImageMultisample")],
        "ImageQuery" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "ImageReadWriteLodAMD" => &[Enablement::Exposed(Exposure::Extension(
            "AmdShaderImageLoadStoreLod",
        ))],
        "InputAttachment" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "InputAttachmentArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderInputAttachmentArrayDynamicIndexing",
        )],
        "InputAttachmentArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderInputAttachmentArrayNonUniformIndexing",
        )],
        "Int16" => &[Enablement::Feature("ShaderInt16")],
        "Int64" => &[Enablement::Feature("ShaderInt64")],
        "Int64Atomics" => &[
            Enablement::Feature("ShaderBufferInt64Atomics"),
            Enablement::Feature("ShaderSharedInt64Atomics"),
            Enablement::Feature("ShaderImageInt64Atomics"),
        ],
        "Int64ImageEXT" => &[Enablement::Feature("ShaderImageInt64Atomics")],
        "Int8" => &[Enablement::Feature("ShaderInt8")],
        "IntegerFunctions2INTEL" => &[Enablement::Feature("ShaderIntegerFunctions2")],
        "InterpolationFunction" => &[Enablement::Feature("SampleRateShading")],
        "LongVectorEXT" => &[Enablement::Feature("LongVector")],
        "Matrix" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "MeshShadingEXT" => &[Enablement::Exposed(Exposure::Extension("ExtMeshShader"))],
        "MeshShadingNV" => &[Enablement::Exposed(Exposure::Extension("NvMeshShader"))],
        "MinLod" => &[Enablement::Feature("ShaderResourceMinLod")],
        "MultiView" => &[
            Enablement::Feature("Multiview"),
            Enablement::Feature("Multiview"),
        ],
        "MultiViewport" => &[Enablement::Feature("MultiViewport")],
        "PerViewAttributesNV" => &[Enablement::Exposed(Exposure::Extension(
            "NvxMultiviewPerViewAttributes",
        ))],
        "PhysicalStorageBufferAddresses" => &[
            Enablement::Feature("BufferDeviceAddress"),
            Enablement::Feature("BufferDeviceAddressEXT"),
        ],
        "PushConstantBanksNV" => &[Enablement::Feature("PushConstantBank")],
        "QuadControlKHR" => &[Enablement::Feature("ShaderQuadControl")],
        "RawAccessChainsNV" => &[Enablement::Feature("ShaderRawAccessChains")],
        "RayCullMaskKHR" => &[Enablement::Feature("RayTracingMaintenance1")],
        "RayQueryKHR" => &[Enablement::Feature("RayQuery")],
        "RayQueryPositionFetchKHR" => &[Enablement::Feature("RayTracingPositionFetch")],
        "RayTracingClusterAccelerationStructureNV" => {
            &[Enablement::Feature("ClusterAccelerationStructure")]
        }
        "RayTracingKHR" => &[Enablement::Feature("RayTracingPipeline")],
        "RayTracingLinearSweptSpheresGeometryNV" => &[Enablement::Feature("LinearSweptSpheres")],
        "RayTracingMotionBlurNV" => &[Enablement::Feature("RayTracingMotionBlur")],
        "RayTracingNV" => &[Enablement::Exposed(Exposure::Extension("NvRayTracing"))],
        "RayTracingOpacityMicromapEXT" => &[Enablement::Exposed(Exposure::Extension(
            "ExtOpacityMicromap",
        ))],
        "RayTracingPositionFetchKHR" => &[Enablement::Feature("RayTracingPositionFetch")],
        "RayTracingSpheresGeometryNV" => &[Enablement::Feature("Spheres")],
        "RayTraversalPrimitiveCullingKHR" => &[
            Enablement::Feature("RayTraversalPrimitiveCulling"),
            Enablement::Feature("RayQuery"),
        ],
        "ReplicatedCompositesEXT" => &[Enablement::Feature("ShaderReplicatedComposites")],
        "RoundingModeRTE" => &[
            Enablement::Property("ShaderRoundingModeRTEFloat16"),
            Enablement::Property("ShaderRoundingModeRTEFloat32"),
            Enablement::Property("ShaderRoundingModeRTEFloat64"),
        ],
        "RoundingModeRTZ" => &[
            Enablement::Property("ShaderRoundingModeRTZFloat16"),
            Enablement::Property("ShaderRoundingModeRTZFloat32"),
            Enablement::Property("ShaderRoundingModeRTZFloat64"),
        ],
        "RuntimeDescriptorArray" => &[Enablement::Feature("RuntimeDescriptorArray")],
        "SampleMaskOverrideCoverageNV" => &[Enablement::Exposed(Exposure::Extension(
            "NvSampleMaskOverrideCoverage",
        ))],
        "SampleMaskPostDepthCoverage" => &[Enablement::Exposed(Exposure::Extension(
            "ExtPostDepthCoverage",
        ))],
        "SampleRateShading" => &[Enablement::Feature("SampleRateShading")],
        "Sampled1D" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "SampledBuffer" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "SampledCubeArray" => &[Enablement::Feature("ImageCubeArray")],
        "SampledImageArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderSampledImageArrayDynamicIndexing",
        )],
        "SampledImageArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderSampledImageArrayNonUniformIndexing",
        )],
        "Shader" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "Shader64BitIndexingEXT" => &[Enablement::Feature("Shader64BitIndexing")],
        "ShaderClockKHR" => &[Enablement::Exposed(Exposure::Extension("KhrShaderClock"))],
        "ShaderEnqueueAMDX" => &[Enablement::Feature("ShaderEnqueue")],
        "ShaderInvocationReorderEXT" => &[Enablement::Exposed(Exposure::Extension(
            "ExtRayTracingInvocationReorder",
        ))],
        "ShaderInvocationReorderNV" => &[Enablement::Exposed(Exposure::Extension(
            "NvRayTracingInvocationReorder",
        ))],
        "ShaderLayer" => &[Enablement::Feature("ShaderOutputLayer")],
        "ShaderNonUniform" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("ExtDescriptorIndexing")),
        ],
        "ShaderSMBuiltinsNV" => &[Enablement::Feature("ShaderSMBuiltins")],
        "ShaderViewportIndex" => &[Enablement::Feature("ShaderOutputViewportIndex")],
        "ShaderViewportIndexLayerEXT" => &[
            Enablement::Exposed(Exposure::Extension("ExtShaderViewportIndexLayer")),
            Enablement::Exposed(Exposure::Extension("NvViewportArray2")),
        ],
        "ShaderViewportMaskNV" => &[Enablement::Exposed(Exposure::Extension("NvViewportArray2"))],
        "SignedZeroInfNanPreserve" => &[
            Enablement::Property("ShaderSignedZeroInfNanPreserveFloat16"),
            Enablement::Property("ShaderSignedZeroInfNanPreserveFloat32"),
            Enablement::Property("ShaderSignedZeroInfNanPreserveFloat64"),
        ],
        "SparseResidency" => &[Enablement::Feature("ShaderResourceResidency")],
        "StencilExportEXT" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderStencilExport",
        ))],
        "StorageBuffer16BitAccess" => &[
            Enablement::Feature("StorageBuffer16BitAccess"),
            Enablement::Feature("StorageBuffer16BitAccess"),
        ],
        "StorageBuffer8BitAccess" => &[Enablement::Feature("StorageBuffer8BitAccess")],
        "StorageBufferArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderStorageBufferArrayDynamicIndexing",
        )],
        "StorageBufferArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderStorageBufferArrayNonUniformIndexing",
        )],
        "StorageImageArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderStorageImageArrayDynamicIndexing",
        )],
        "StorageImageArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderStorageImageArrayNonUniformIndexing",
        )],
        "StorageImageExtendedFormats" => &[Enablement::Exposed(Exposure::Core(1, 0))],
        "StorageImageMultisample" => &[Enablement::Feature("ShaderStorageImageMultisample")],
        "StorageImageReadWithoutFormat" => &[
            Enablement::Feature("ShaderStorageImageReadWithoutFormat"),
            Enablement::Exposed(Exposure::Core(1, 3)),
            Enablement::Exposed(Exposure::Extension("KhrFormatFeatureFlags2")),
        ],
        "StorageImageWriteWithoutFormat" => &[
            Enablement::Feature("ShaderStorageImageWriteWithoutFormat"),
            Enablement::Exposed(Exposure::Core(1, 3)),
            Enablement::Exposed(Exposure::Extension("KhrFormatFeatureFlags2")),
        ],
        "StorageInputOutput16" => &[
            Enablement::Feature("StorageInputOutput16"),
            Enablement::Feature("StorageInputOutput16"),
        ],
        "StoragePushConstant16" => &[
            Enablement::Feature("StoragePushConstant16"),
            Enablement::Feature("StoragePushConstant16"),
        ],
        "StoragePushConstant8" => &[Enablement::Feature("StoragePushConstant8")],
        "StorageTensorArrayDynamicIndexingARM" => &[Enablement::Feature(
            "ShaderStorageTensorArrayDynamicIndexing",
        )],
        "StorageTensorArrayNonUniformIndexingARM" => &[Enablement::Feature(
            "ShaderStorageTensorArrayNonUniformIndexing",
        )],
        "StorageTexelBufferArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderStorageTexelBufferArrayDynamicIndexing",
        )],
        "StorageTexelBufferArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderStorageTexelBufferArrayNonUniformIndexing",
        )],
        "SubgroupBallotKHR" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderSubgroupBallot",
        ))],
        "SubgroupVoteKHR" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderSubgroupVote",
        ))],
        "TensorAddressingNV" => &[Enablement::Feature("CooperativeMatrixTensorAddressing")],
        "TensorsARM" => &[Enablement::Feature("ShaderTensorAccess")],
        "Tessellation" => &[Enablement::Feature("TessellationShader")],
        "TessellationPointSize" => &[Enablement::Feature(
            "ShaderTessellationAndGeometryPointSize",
        )],
        "TextureBlockMatch2QCOM" => &[Enablement::Feature("TextureBlockMatch2")],
        "TextureBlockMatchQCOM" => &[Enablement::Feature("TextureBlockMatch")],
        "TextureBoxFilterQCOM" => &[Enablement::Feature("TextureBoxFilter")],
        "TextureSampleWeightedQCOM" => &[Enablement::Feature("TextureSampleWeighted")],
        "TileImageColorReadAccessEXT" => &[Enablement::Feature("ShaderTileImageColorReadAccess")],
        "TileImageDepthReadAccessEXT" => &[Enablement::Feature("ShaderTileImageDepthReadAccess")],
        "TileImageStencilReadAccessEXT" => {
            &[Enablement::Feature("ShaderTileImageStencilReadAccess")]
        }
        "TileShadingQCOM" => &[Enablement::Feature("TileShading")],
        "TransformFeedback" => &[Enablement::Feature("TransformFeedback")],
        "UniformAndStorageBuffer16BitAccess" => &[
            Enablement::Feature("UniformAndStorageBuffer16BitAccess"),
            Enablement::Feature("UniformAndStorageBuffer16BitAccess"),
        ],
        "UniformAndStorageBuffer8BitAccess" => {
            &[Enablement::Feature("UniformAndStorageBuffer8BitAccess")]
        }
        "UniformBufferArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderUniformBufferArrayDynamicIndexing",
        )],
        "UniformBufferArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderUniformBufferArrayNonUniformIndexing",
        )],
        "UniformTexelBufferArrayDynamicIndexing" => &[Enablement::Feature(
            "ShaderUniformTexelBufferArrayDynamicIndexing",
        )],
        "UniformTexelBufferArrayNonUniformIndexing" => &[Enablement::Feature(
            "ShaderUniformTexelBufferArrayNonUniformIndexing",
        )],
        "UntypedPointersKHR" => &[Enablement::Feature("ShaderUntypedPointers")],
        "VariablePointers" => &[
            Enablement::Feature("VariablePointers"),
            Enablement::Feature("VariablePointers"),
        ],
        "VariablePointersStorageBuffer" => &[
            Enablement::Feature("VariablePointersStorageBuffer"),
            Enablement::Feature("VariablePointersStorageBuffer"),
        ],
        "VulkanMemoryModel" => &[Enablement::Feature("VulkanMemoryModel")],
        "VulkanMemoryModelDeviceScope" => &[Enablement::Feature("VulkanMemoryModelDeviceScope")],
        "WorkgroupMemoryExplicitLayout16BitAccessKHR" => &[Enablement::Feature(
            "WorkgroupMemoryExplicitLayout16BitAccess",
        )],
        "WorkgroupMemoryExplicitLayout8BitAccessKHR" => &[Enablement::Feature(
            "WorkgroupMemoryExplicitLayout8BitAccess",
        )],
        "WorkgroupMemoryExplicitLayoutKHR" => {
            &[Enablement::Feature("WorkgroupMemoryExplicitLayout")]
        }
        _ => return None,
    })
}
pub(crate) fn extension_enablement(name: &str) -> Option<&'static [Enablement]> {
    Some(match name {
        "SPV_AMD_gcn_shader" => &[Enablement::Exposed(Exposure::Extension("AmdGcnShader"))],
        "SPV_AMD_gpu_shader_half_float" => &[Enablement::Exposed(Exposure::Extension(
            "AmdGpuShaderHalfFloat",
        ))],
        "SPV_AMD_gpu_shader_int16" => &[Enablement::Exposed(Exposure::Extension(
            "AmdGpuShaderInt16",
        ))],
        "SPV_AMD_shader_ballot" => &[Enablement::Exposed(Exposure::Extension("AmdShaderBallot"))],
        "SPV_AMD_shader_early_and_late_fragment_tests" => &[Enablement::Exposed(
            Exposure::Extension("AmdShaderEarlyAndLateFragmentTests"),
        )],
        "SPV_AMD_shader_explicit_vertex_parameter" => &[Enablement::Exposed(Exposure::Extension(
            "AmdShaderExplicitVertexParameter",
        ))],
        "SPV_AMD_shader_fragment_mask" => &[Enablement::Exposed(Exposure::Extension(
            "AmdShaderFragmentMask",
        ))],
        "SPV_AMD_shader_image_load_store_lod" => &[Enablement::Exposed(Exposure::Extension(
            "AmdShaderImageLoadStoreLod",
        ))],
        "SPV_AMD_shader_trinary_minmax" => &[Enablement::Exposed(Exposure::Extension(
            "AmdShaderTrinaryMinmax",
        ))],
        "SPV_AMD_texture_gather_bias_lod" => &[Enablement::Exposed(Exposure::Extension(
            "AmdTextureGatherBiasLod",
        ))],
        "SPV_ARM_core_builtins" => &[Enablement::Exposed(Exposure::Extension(
            "ArmShaderCoreBuiltins",
        ))],
        "SPV_ARM_graph" => &[Enablement::Exposed(Exposure::Extension("ArmDataGraph"))],
        "SPV_ARM_tensors" => &[Enablement::Exposed(Exposure::Extension("ArmTensors"))],
        "SPV_EXT_demote_to_helper_invocation" => &[
            Enablement::Exposed(Exposure::Core(1, 3)),
            Enablement::Exposed(Exposure::Extension("ExtShaderDemoteToHelperInvocation")),
        ],
        "SPV_EXT_descriptor_heap" => &[Enablement::Exposed(Exposure::Extension(
            "ExtDescriptorHeap",
        ))],
        "SPV_EXT_descriptor_indexing" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("ExtDescriptorIndexing")),
        ],
        "SPV_EXT_float8" => &[Enablement::Exposed(Exposure::Extension("ExtShaderFloat8"))],
        "SPV_EXT_fragment_fully_covered" => &[Enablement::Exposed(Exposure::Extension(
            "ExtConservativeRasterization",
        ))],
        "SPV_EXT_fragment_invocation_density" => &[Enablement::Exposed(Exposure::Extension(
            "ExtFragmentDensityMap",
        ))],
        "SPV_EXT_fragment_shader_interlock" => &[Enablement::Exposed(Exposure::Extension(
            "ExtFragmentShaderInterlock",
        ))],
        "SPV_EXT_long_vector" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderLongVector",
        ))],
        "SPV_EXT_mesh_shader" => &[Enablement::Exposed(Exposure::Extension("ExtMeshShader"))],
        "SPV_EXT_opacity_micromap" => &[Enablement::Exposed(Exposure::Extension(
            "ExtOpacityMicromap",
        ))],
        "SPV_EXT_physical_storage_buffer" => &[Enablement::Exposed(Exposure::Extension(
            "ExtBufferDeviceAddress",
        ))],
        "SPV_EXT_replicated_composites" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderReplicatedComposites",
        ))],
        "SPV_EXT_shader_64bit_indexing" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShader64BitIndexing",
        ))],
        "SPV_EXT_shader_atomic_float16_add" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderAtomicFloat2",
        ))],
        "SPV_EXT_shader_atomic_float_add" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderAtomicFloat",
        ))],
        "SPV_EXT_shader_atomic_float_min_max" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderAtomicFloat2",
        ))],
        "SPV_EXT_shader_image_int64" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderImageAtomicInt64",
        ))],
        "SPV_EXT_shader_invocation_reorder" => &[Enablement::Exposed(Exposure::Extension(
            "ExtRayTracingInvocationReorder",
        ))],
        "SPV_EXT_shader_stencil_export" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderStencilExport",
        ))],
        "SPV_EXT_shader_subgroup_partitioned" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderSubgroupPartitioned",
        ))],
        "SPV_EXT_shader_tile_image" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderTileImage",
        ))],
        "SPV_EXT_shader_viewport_index_layer" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("ExtShaderViewportIndexLayer")),
        ],
        "SPV_GOOGLE_decorate_string" => &[Enablement::Exposed(Exposure::Extension(
            "GoogleDecorateString",
        ))],
        "SPV_GOOGLE_hlsl_functionality1" => &[Enablement::Exposed(Exposure::Extension(
            "GoogleHlslFunctionality1",
        ))],
        "SPV_GOOGLE_user_type" => &[Enablement::Exposed(Exposure::Extension("GoogleUserType"))],
        "SPV_HUAWEI_cluster_culling_shader" => &[Enablement::Exposed(Exposure::Extension(
            "HuaweiClusterCullingShader",
        ))],
        "SPV_HUAWEI_subpass_shading" => &[Enablement::Exposed(Exposure::Extension(
            "HuaweiSubpassShading",
        ))],
        "SPV_INTEL_shader_integer_functions2" => &[Enablement::Exposed(Exposure::Extension(
            "IntelShaderIntegerFunctions2",
        ))],
        "SPV_KHR_16bit_storage" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("Khr16BitStorage")),
        ],
        "SPV_KHR_8bit_storage" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("Khr8BitStorage")),
        ],
        "SPV_KHR_bfloat16" => &[Enablement::Exposed(Exposure::Extension(
            "KhrShaderBfloat16",
        ))],
        "SPV_KHR_compute_shader_derivatives" => &[Enablement::Exposed(Exposure::Extension(
            "KhrComputeShaderDerivatives",
        ))],
        "SPV_KHR_cooperative_matrix" => &[Enablement::Exposed(Exposure::Extension(
            "KhrCooperativeMatrix",
        ))],
        "SPV_KHR_device_group" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("KhrDeviceGroup")),
        ],
        "SPV_KHR_expect_assume" => &[
            Enablement::Exposed(Exposure::Core(1, 4)),
            Enablement::Exposed(Exposure::Extension("KhrShaderExpectAssume")),
        ],
        "SPV_KHR_float_controls" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("KhrShaderFloatControls")),
        ],
        "SPV_KHR_float_controls2" => &[
            Enablement::Exposed(Exposure::Core(1, 4)),
            Enablement::Exposed(Exposure::Extension("KhrShaderFloatControls2")),
        ],
        "SPV_KHR_fma" => &[Enablement::Exposed(Exposure::Extension("KhrShaderFma"))],
        "SPV_KHR_fragment_shader_barycentric" => &[Enablement::Exposed(Exposure::Extension(
            "KhrFragmentShaderBarycentric",
        ))],
        "SPV_KHR_fragment_shading_rate" => &[Enablement::Exposed(Exposure::Extension(
            "KhrFragmentShadingRate",
        ))],
        "SPV_KHR_integer_dot_product" => &[
            Enablement::Exposed(Exposure::Core(1, 3)),
            Enablement::Exposed(Exposure::Extension("KhrShaderIntegerDotProduct")),
        ],
        "SPV_KHR_maximal_reconvergence" => &[Enablement::Exposed(Exposure::Extension(
            "KhrShaderMaximalReconvergence",
        ))],
        "SPV_KHR_multiview" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("KhrMultiview")),
        ],
        "SPV_KHR_non_semantic_info" => &[
            Enablement::Exposed(Exposure::Core(1, 3)),
            Enablement::Exposed(Exposure::Extension("KhrShaderNonSemanticInfo")),
        ],
        "SPV_KHR_physical_storage_buffer" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("KhrBufferDeviceAddress")),
        ],
        "SPV_KHR_post_depth_coverage" => &[Enablement::Exposed(Exposure::Extension(
            "ExtPostDepthCoverage",
        ))],
        "SPV_KHR_quad_control" => &[Enablement::Exposed(Exposure::Extension(
            "KhrShaderQuadControl",
        ))],
        "SPV_KHR_ray_cull_mask" => &[Enablement::Exposed(Exposure::Extension(
            "KhrRayTracingMaintenance1",
        ))],
        "SPV_KHR_ray_query" => &[Enablement::Exposed(Exposure::Extension("KhrRayQuery"))],
        "SPV_KHR_ray_tracing" => &[Enablement::Exposed(Exposure::Extension(
            "KhrRayTracingPipeline",
        ))],
        "SPV_KHR_ray_tracing_position_fetch" => &[Enablement::Exposed(Exposure::Extension(
            "KhrRayTracingPositionFetch",
        ))],
        "SPV_KHR_relaxed_extended_instruction" => &[Enablement::Exposed(Exposure::Extension(
            "KhrShaderRelaxedExtendedInstruction",
        ))],
        "SPV_KHR_shader_ballot" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderSubgroupBallot",
        ))],
        "SPV_KHR_shader_clock" => &[Enablement::Exposed(Exposure::Extension("KhrShaderClock"))],
        "SPV_KHR_shader_draw_parameters" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("KhrShaderDrawParameters")),
        ],
        "SPV_KHR_storage_buffer_storage_class" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("KhrStorageBufferStorageClass")),
        ],
        "SPV_KHR_subgroup_rotate" => &[
            Enablement::Exposed(Exposure::Core(1, 4)),
            Enablement::Exposed(Exposure::Extension("KhrShaderSubgroupRotate")),
        ],
        "SPV_KHR_subgroup_uniform_control_flow" => &[Enablement::Exposed(Exposure::Extension(
            "KhrShaderSubgroupUniformControlFlow",
        ))],
        "SPV_KHR_subgroup_vote" => &[Enablement::Exposed(Exposure::Extension(
            "ExtShaderSubgroupVote",
        ))],
        "SPV_KHR_terminate_invocation" => &[
            Enablement::Exposed(Exposure::Core(1, 3)),
            Enablement::Exposed(Exposure::Extension("KhrShaderTerminateInvocation")),
        ],
        "SPV_KHR_untyped_pointers" => &[Enablement::Exposed(Exposure::Extension(
            "KhrShaderUntypedPointers",
        ))],
        "SPV_KHR_variable_pointers" => &[
            Enablement::Exposed(Exposure::Core(1, 1)),
            Enablement::Exposed(Exposure::Extension("KhrVariablePointers")),
        ],
        "SPV_KHR_vulkan_memory_model" => &[
            Enablement::Exposed(Exposure::Core(1, 2)),
            Enablement::Exposed(Exposure::Extension("KhrVulkanMemoryModel")),
        ],
        "SPV_KHR_workgroup_memory_explicit_layout" => &[Enablement::Exposed(Exposure::Extension(
            "KhrWorkgroupMemoryExplicitLayout",
        ))],
        "SPV_NVX_multiview_per_view_attributes" => &[Enablement::Exposed(Exposure::Extension(
            "NvxMultiviewPerViewAttributes",
        ))],
        "SPV_NV_cluster_acceleration_structure" => &[Enablement::Exposed(Exposure::Extension(
            "NvClusterAccelerationStructure",
        ))],
        "SPV_NV_compute_shader_derivatives" => &[Enablement::Exposed(Exposure::Extension(
            "NvComputeShaderDerivatives",
        ))],
        "SPV_NV_cooperative_matrix" => &[Enablement::Exposed(Exposure::Extension(
            "NvCooperativeMatrix",
        ))],
        "SPV_NV_cooperative_matrix2" => &[Enablement::Exposed(Exposure::Extension(
            "NvCooperativeMatrix2",
        ))],
        "SPV_NV_cooperative_vector" => &[Enablement::Exposed(Exposure::Extension(
            "NvCooperativeVector",
        ))],
        "SPV_NV_fragment_shader_barycentric" => &[Enablement::Exposed(Exposure::Extension(
            "NvFragmentShaderBarycentric",
        ))],
        "SPV_NV_geometry_shader_passthrough" => &[Enablement::Exposed(Exposure::Extension(
            "NvGeometryShaderPassthrough",
        ))],
        "SPV_NV_linear_swept_spheres" => &[Enablement::Exposed(Exposure::Extension(
            "NvRayTracingLinearSweptSpheres",
        ))],
        "SPV_NV_mesh_shader" => &[Enablement::Exposed(Exposure::Extension("NvMeshShader"))],
        "SPV_NV_push_constant_bank" => &[Enablement::Exposed(Exposure::Extension(
            "NvPushConstantBank",
        ))],
        "SPV_NV_raw_access_chains" => &[Enablement::Exposed(Exposure::Extension(
            "NvRawAccessChains",
        ))],
        "SPV_NV_ray_tracing" => &[Enablement::Exposed(Exposure::Extension("NvRayTracing"))],
        "SPV_NV_ray_tracing_motion_blur" => &[Enablement::Exposed(Exposure::Extension(
            "NvRayTracingMotionBlur",
        ))],
        "SPV_NV_sample_mask_override_coverage" => &[Enablement::Exposed(Exposure::Extension(
            "NvSampleMaskOverrideCoverage",
        ))],
        "SPV_NV_shader_atomic_fp16_vector" => &[Enablement::Exposed(Exposure::Extension(
            "NvShaderAtomicFloat16Vector",
        ))],
        "SPV_NV_shader_image_footprint" => &[Enablement::Exposed(Exposure::Extension(
            "NvShaderImageFootprint",
        ))],
        "SPV_NV_shader_invocation_reorder" => &[Enablement::Exposed(Exposure::Extension(
            "NvRayTracingInvocationReorder",
        ))],
        "SPV_NV_shader_sm_builtins" => &[Enablement::Exposed(Exposure::Extension(
            "NvShaderSmBuiltins",
        ))],
        "SPV_NV_shader_subgroup_partitioned" => &[
            Enablement::Exposed(Exposure::Extension("NvShaderSubgroupPartitioned")),
            Enablement::Exposed(Exposure::Extension("ExtShaderSubgroupPartitioned")),
        ],
        "SPV_NV_shading_rate" => &[Enablement::Exposed(Exposure::Extension(
            "NvShadingRateImage",
        ))],
        "SPV_NV_tensor_addressing" => &[Enablement::Exposed(Exposure::Extension(
            "NvCooperativeMatrix2",
        ))],
        "SPV_NV_viewport_array2" => &[Enablement::Exposed(Exposure::Extension("NvViewportArray2"))],
        "SPV_QCOM_image_processing" => &[Enablement::Exposed(Exposure::Extension(
            "QcomImageProcessing",
        ))],
        "SPV_QCOM_image_processing2" => &[Enablement::Exposed(Exposure::Extension(
            "QcomImageProcessing2",
        ))],
        "SPV_QCOM_tile_shading" => &[Enablement::Exposed(Exposure::Extension("QcomTileShading"))],
        _ => return None,
    })
}
