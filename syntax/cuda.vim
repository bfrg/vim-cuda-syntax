" ==============================================================================
" Vim syntax file
" Language:     CUDA (NVIDIA Compute Unified Device Architecture)
" Maintainer:   bfrg <bfrg@users.noreply.github.com>
" Website:      https://github.cim/bfrg/vim-cuda-syntax
" Last Change:  Sep 10, 2018
"
" Enhanced CUDA syntax highlighting including highlighting of CUDA kernel calls.
"
" This syntax file fully replaces Vim's default CUDA syntax file. Hence, put it
" into the ~/.vim/syntax/ directory.
"
" Keywords were accumulated from the CUDA Toolkit Documentation:
" http://docs.nvidia.com/cuda/index.html
"
" Notes:
" CUDA data fields are not highlighted because many keywords have familiar names
" which could collide with either user-defined variables (like ptr, x, y, z), or
" with standard library types (like function or array).
" ==============================================================================


" Exit when a syntax file was already loaded
if exists('b:current_syntax')
    finish
endif

" CUDA supports a lot of C++ syntax nowadays, so let's read the C++ syntax file
runtime! syntax/cpp.vim
unlet b:current_syntax


" Highlight CUDA kernel calls {{{1
if !exists('g:cuda_no_kernel_highlight')
    " Thanks to lopid from freenode/#vim
    " Match <<< and >>> only when they appear as a pair, must be on the same line
    " Look back only 128 bytes before '>>>' too speed up parsing
    syntax match cudaKernelAngles "<<<\(.\{-}>>>\)\@=\|\(<<<.\{-}\)\@128<=>>>"
    " Highlight names of kernel calls, like 'foo' in foo<<<grid, threads>>>(var)
    syntax match cudaKernel "\<\h\w*\>\(\s\|\n\)*<<<"me=e-3
    hi default link cudaKernel       Function
    hi default link cudaKernelAngles Operator
endif


" C language extensions {{{1
" Based on: http://docs.nvidia.com/cuda/cuda-c-programming-guide

" B.1/B.2
syntax keyword cudaStorageClass __device__ __global__ __host__
syntax keyword cudaStorageClass __noinline__ __forceinline__
syntax keyword cudaStorageClass __shared__ __constant__ __managed__ __restrict__
syntax keyword cudaConstant     __CUDA_ARCH__

" B.3 Built-in Vector Types
syntax keyword cudaType     char1 char2 char3 char4
syntax keyword cudaType     uchar1 uchar2 uchar3 uchar4
syntax keyword cudaType     short1 short2 short3 short4
syntax keyword cudaType     ushort1 ushort2 ushort3 ushort4
syntax keyword cudaType     int1 int2 int3 int4
syntax keyword cudaType     uint1 uint2 uint3 uint4
syntax keyword cudaType     long1 long2 long3 long4
syntax keyword cudaType     ulong1 ulong2 ulong3 ulong4
syntax keyword cudaType     float1 float2 float3 float4
syntax keyword cudaType     ufloat1 ufloat2 ufloat3 ufloat4
syntax keyword cudaType     longlong1 longlong2 ulonglong1 ulonglong2
syntax keyword cudaType     longlong3 longlong4 ulonglong3 ulonglong4
syntax keyword cudaType     double1 double2 double3 double4
syntax keyword cudaType     dim3
syntax keyword cudaFunction make_char1 make_char2 make_char3 make_char4
syntax keyword cudaFunction make_uchar1 make_uchar2 make_uchar3 make_uchar4
syntax keyword cudaFunction make_short1 make_short2 make_short3 make_short4
syntax keyword cudaFunction make_ushort1 make_ushort2 make_ushort3 make_ushort4
syntax keyword cudaFunction make_int1 make_int2 make_int3 make_int4
syntax keyword cudaFunction make_uint1 make_uint2 make_uint3 make_uint4
syntax keyword cudaFunction make_long1 make_long2 make_long3 make_long4
syntax keyword cudaFunction make_ulong1 make_ulong2 make_ulong3 make_ulong4
syntax keyword cudaFunction make_float1 make_float2 make_float3 make_float4
syntax keyword cudaFunction make_ufloat1 make_ufloat2 make_ufloat3 make_ufloat4
syntax keyword cudaFunction make_longlong1 make_longlong2 make_longlong3 make_longlong4
syntax keyword cudaFunction make_ulonglong1 make_ulonglong2 make_ulonglong3 make_ulonglong4
syntax keyword cudaFunction make_double1 make_double2 make_double3 make_double4

" B.4 Built-in variables
syntax keyword cudaVariable gridDim blockIdx blockDim threadIdx warpSize

" B.5/B.6
syntax keyword cudaFunction __threadfence_block __threadfence __threadfence_system
syntax keyword cudaFunction __syncthreads __syncthreads_count __syncthreads_and __syncthreads_or
syntax keyword cudaFunction __syncwarp

" B.8 Texture functions and objects
syntax keyword cudaType     texture
syntax keyword cudaConstant cudaTextureType1D cudaTextureType2D cudaTextureType3D
syntax keyword cudaConstant cudaTextureType1DLayered cudaTextureType2DLayered
syntax keyword cudaConstant cudaTextureTypeCubemap cudaTextureTypeCubemapLayered
syntax keyword cudaFunction tex1Dfetch
syntax keyword cudaFunction tex1D tex1DLod tex1DGrad
syntax keyword cudaFunction tex2D tex2DLod tex2DGrad
syntax keyword cudaFunction tex3D tex3DLod tex3DGrad
syntax keyword cudaFunction tex1DLayered tex1DLayeredLod tex1DLayeredGrad
syntax keyword cudaFunction tex2DLayered tex2DLayeredLod tex2DLayeredGrad
syntax keyword cudaFunction tex3DLayered tex3DLayeredLod tex3DLayeredGrad
syntax keyword cudaFunction texCubemap
syntax keyword cudaFunction texCubemapLod
syntax keyword cudaFunction texCubemapLayered
syntax keyword cudaFunction texCubemapLayeredLod
syntax keyword cudaFunction tex2Dgather

" B.9 Surface functions and objects
syntax keyword cudaType     surface
syntax keyword cudaConstant cudaSurfaceType1D cudaSurfaceType2D cudaSurfaceType3D
syntax keyword cudaConstant cudaSurfaceTypeCubemap cudaSurfaceTypeCubemapLayered
syntax keyword cudaConstant cudaSurfaceType1DLayered cudaSurfaceType2DLayered
syntax keyword cudaFunction surf1Dread surf1Dwrite
syntax keyword cudaFunction surf2Dread surf2Dwrite
syntax keyword cudaFunction surf3Dread surf3Dwrite
syntax keyword cudaFunction surf1DLayeredread surf1DLayeredwrite
syntax keyword cudaFunction surf2DLayeredread surf2DLayeredwrite
syntax keyword cudaFunction surfCubemapread
syntax keyword cudaFunction surfCubemapwrite
syntax keyword cudaFunction surfCubemapLayeredread
syntax keyword cudaFunction surfCubemapLayeredwrite

" B.10/B.11/B.12/B.13/B.14/B.15
syntax keyword cudaFunction __ldg
syntax keyword cudaFunction atomicAdd_system
syntax keyword cudaFunction atomicAdd atomicSub
syntax keyword cudaFunction atomicMin atomicMax
syntax keyword cudaFunction atomicExch atomicInc atomicDec atomicCAS
syntax keyword cudaFunction atomicAnd atomicOr atomicXor
syntax keyword cudaFunction __all_sync
syntax keyword cudaFunction __any_sync
syntax keyword cudaFunction __ballot_sync
syntax keyword cudaFunction __activemask
syntax keyword cudaFunction __match_any_sync __match_all_sync
syntax keyword cudaFunction __shfl_sync __shfl_up_sync __shfl_down_sync __shfl_xor_sync

" B.16 Warp matrix functions
syntax keyword cudaNamespace nvcuda wmma
syntax keyword cudaType      fragment
syntax keyword cudaFunction  load_matrix_sync
syntax keyword cudaFunction  store_matrix_sync
syntax keyword cudaFunction  fill_fragment
syntax keyword cudaFunction  mma_sync

" B.17/B.18/B.19
syntax keyword cudaFunction __prof_trigger
syntax keyword cudaFunction __launch_bounds__


" Cooperative Groups <cooperative_groups.h> {{{1
" C.2/C.3/C.4
syntax keyword cudaNamespace cooperative_groups
syntax keyword cudaType      thread_block
syntax keyword cudaType      thread_group
syntax keyword cudaType      thread_block_tile
syntax keyword cudaType      coalesced_group
syntax keyword cudaType      grid_group
syntax keyword cudaFunction  thread_rank
syntax keyword cudaFunction  group_index thread_index
syntax keyword cudaFunction  this_thread_block
syntax keyword cudaFunction  this_grid
syntax keyword cudaFunction  tiled_partition


" Polymorphic function wrappers <nvfunctional> {{{1
" F.4/F.5
syntax keyword cudaNamespace nvstd
syntax keyword cudaType      function
syntax keyword cudaType      __nv_is_extended_device_lambda_closure_type
syntax keyword cudaType      __nv_is_extended_host_device_lambda_closure_type


" CUDA Runtime API {{{1
" Based on: http://docs.nvidia.com/cuda/cuda-runtime-api (v9.1.85, Jan 24, 2018)

if exists('g:cuda_runtime_api_highlight') && g:cuda_runtime_api_highlight
    " 4. Modules -- 4.1. Device Management
    syntax keyword cudaFunction cudaChooseDevice
    syntax keyword cudaFunction cudaDeviceGetAttribute
    syntax keyword cudaFunction cudaDeviceGetByPCIBusId
    syntax keyword cudaFunction cudaDeviceGetCacheConfig
    syntax keyword cudaFunction cudaDeviceGetLimit
    syntax keyword cudaFunction cudaDeviceGetP2PAttribute
    syntax keyword cudaFunction cudaDeviceGetPCIBusId
    syntax keyword cudaFunction cudaDeviceGetSharedMemConfig
    syntax keyword cudaFunction cudaDeviceGetStreamPriorityRange
    syntax keyword cudaFunction cudaDeviceReset
    syntax keyword cudaFunction cudaDeviceSetCacheConfig
    syntax keyword cudaFunction cudaDeviceSetLimit
    syntax keyword cudaFunction cudaDeviceSetSharedMemConfig
    syntax keyword cudaFunction cudaDeviceSynchronize
    syntax keyword cudaFunction cudaGetDevice
    syntax keyword cudaFunction cudaGetDeviceCount
    syntax keyword cudaFunction cudaGetDeviceFlags
    syntax keyword cudaFunction cudaGetDeviceProperties
    syntax keyword cudaFunction cudaIpcCloseMemHandle
    syntax keyword cudaFunction cudaIpcGetEventHandle
    syntax keyword cudaFunction cudaIpcGetMemHandle
    syntax keyword cudaFunction cudaIpcOpenEventHandle
    syntax keyword cudaFunction cudaIpcOpenMemHandle
    syntax keyword cudaFunction cudaSetDevice
    syntax keyword cudaFunction cudaSetDeviceFlags
    syntax keyword cudaFunction cudaSetValidDevices

    " 4.3. Error Handling
    syntax keyword cudaFunction cudaGetErrorName
    syntax keyword cudaFunction cudaGetErrorString
    syntax keyword cudaFunction cudaGetLastError
    syntax keyword cudaFunction cudaPeekAtLastError

    " 4.4. Stream Management
    syntax keyword cudaType     cudaStreamCallback_t
    syntax keyword cudaFunction cudaStreamAddCallback
    syntax keyword cudaFunction cudaStreamAttachMemAsync
    syntax keyword cudaFunction cudaStreamCreate
    syntax keyword cudaFunction cudaStreamCreateWithFlags
    syntax keyword cudaFunction cudaStreamCreateWithPriority
    syntax keyword cudaFunction cudaStreamDestroy
    syntax keyword cudaFunction cudaStreamGetFlags
    syntax keyword cudaFunction cudaStreamGetPriority
    syntax keyword cudaFunction cudaStreamQuery
    syntax keyword cudaFunction cudaStreamSynchronize
    syntax keyword cudaFunction cudaStreamWaitEvent

    " 4.5. Event Management
    syntax keyword cudaFunction cudaEventCreate
    syntax keyword cudaFunction cudaEventCreateWithFlags
    syntax keyword cudaFunction cudaEventDestroy
    syntax keyword cudaFunction cudaEventElapsedTime
    syntax keyword cudaFunction cudaEventQuery
    syntax keyword cudaFunction cudaEventRecord
    syntax keyword cudaFunction cudaEventSynchronize

    " 4.6. Execution Control
    syntax keyword cudaFunction cudaFuncGetAttributes
    syntax keyword cudaFunction cudaFuncSetAttribute
    syntax keyword cudaFunction cudaFuncSetCacheConfig
    syntax keyword cudaFunction cudaFuncSetSharedMemConfig
    syntax keyword cudaFunction cudaGetParameterBuffer
    syntax keyword cudaFunction cudaGetParameterBufferV2
    syntax keyword cudaFunction cudaLaunchCooperativeKernel
    syntax keyword cudaFunction cudaLaunchCooperativeKernelMultiDevice
    syntax keyword cudaFunction cudaLaunchKernel

    " 4.7. Occupancy
    syntax keyword cudaFunction cudaOccupancyMaxActiveBlocksPerMultiprocessor
    syntax keyword cudaFunction cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

    " 4.9. Memory Management
    syntax keyword cudaFunction cudaArrayGetInfo
    syntax keyword cudaFunction cudaFree
    syntax keyword cudaFunction cudaFreeArray
    syntax keyword cudaFunction cudaFreeHost
    syntax keyword cudaFunction cudaFreeMipmappedArray
    syntax keyword cudaFunction cudaGetMipmappedArrayLevel
    syntax keyword cudaFunction cudaGetSymbolAddress
    syntax keyword cudaFunction cudaGetSymbolSize
    syntax keyword cudaFunction cudaHostAlloc
    syntax keyword cudaFunction cudaHostGetDevicePointer
    syntax keyword cudaFunction cudaHostGetFlags
    syntax keyword cudaFunction cudaHostRegister
    syntax keyword cudaFunction cudaHostUnregister
    syntax keyword cudaFunction cudaMalloc
    syntax keyword cudaFunction cudaMalloc3D
    syntax keyword cudaFunction cudaMalloc3DArray
    syntax keyword cudaFunction cudaMallocArray
    syntax keyword cudaFunction cudaMallocHost
    syntax keyword cudaFunction cudaMallocManaged
    syntax keyword cudaFunction cudaMallocMipmappedArray
    syntax keyword cudaFunction cudaMallocPitch
    syntax keyword cudaFunction cudaMemAdvise
    syntax keyword cudaFunction cudaMemGetInfo
    syntax keyword cudaFunction cudaMemPrefetchAsync
    syntax keyword cudaFunction cudaMemRangeGetAttribute
    syntax keyword cudaFunction cudaMemRangeGetAttributes
    syntax keyword cudaFunction cudaMemcpy
    syntax keyword cudaFunction cudaMemcpy2D
    syntax keyword cudaFunction cudaMemcpy2DArrayToArray
    syntax keyword cudaFunction cudaMemcpy2DAsync
    syntax keyword cudaFunction cudaMemcpy2DFromArray
    syntax keyword cudaFunction cudaMemcpy2DFromArrayAsync
    syntax keyword cudaFunction cudaMemcpy2DToArray
    syntax keyword cudaFunction cudaMemcpy2DToArrayAsync
    syntax keyword cudaFunction cudaMemcpy3D
    syntax keyword cudaFunction cudaMemcpy3DAsync
    syntax keyword cudaFunction cudaMemcpy3DPeer
    syntax keyword cudaFunction cudaMemcpy3DPeerAsync
    syntax keyword cudaFunction cudaMemcpyArrayToArray
    syntax keyword cudaFunction cudaMemcpyAsync
    syntax keyword cudaFunction cudaMemcpyFromArray
    syntax keyword cudaFunction cudaMemcpyFromArrayAsync
    syntax keyword cudaFunction cudaMemcpyFromSymbol
    syntax keyword cudaFunction cudaMemcpyFromSymbolAsync
    syntax keyword cudaFunction cudaMemcpyPeer
    syntax keyword cudaFunction cudaMemcpyPeerAsync
    syntax keyword cudaFunction cudaMemcpyToArray
    syntax keyword cudaFunction cudaMemcpyToArrayAsync
    syntax keyword cudaFunction cudaMemcpyToSymbol
    syntax keyword cudaFunction cudaMemcpyToSymbolAsync
    syntax keyword cudaFunction cudaMemset
    syntax keyword cudaFunction cudaMemset2D
    syntax keyword cudaFunction cudaMemset2DAsync
    syntax keyword cudaFunction cudaMemset3D
    syntax keyword cudaFunction cudaMemset3DAsync
    syntax keyword cudaFunction cudaMemsetAsync
    syntax keyword cudaFunction make_cudaExtent
    syntax keyword cudaFunction make_cudaPitchedPtr
    syntax keyword cudaFunction make_cudaPos

    " 4.10. Unified Addressing
    syntax keyword cudaFunction cudaPointerGetAttributes

    " 4.11. Peer Device Memory Access
    syntax keyword cudaFunction cudaDeviceCanAccessPeer
    syntax keyword cudaFunction cudaDeviceDisablePeerAccess
    syntax keyword cudaFunction cudaDeviceEnablePeerAccess

    " 4.12. OpenGL Interoperability
    syntax keyword cudaType     cudaGLDeviceList
    syntax keyword cudaConstant cudaGLDeviceListAll cudaGLDeviceListCurrentFrame cudaGLDeviceListNextFrame
    syntax keyword cudaFunction cudaGLGetDevices
    syntax keyword cudaFunction cudaGraphicsGLRegisterBuffer
    syntax keyword cudaFunction cudaGraphicsGLRegisterImage
    syntax keyword cudaFunction cudaWGLGetDevice

    " 4.14. Direct3D 9 Interoperability
    syntax keyword cudaType     cudaD3D9DeviceList
    syntax keyword cudaConstant cudaD3D9DeviceListAll cudaD3D9DeviceListCurrentFrame cudaD3D9DeviceListNextFrame
    syntax keyword cudaFunction cudaD3D9GetDevice
    syntax keyword cudaFunction cudaD3D9GetDevices
    syntax keyword cudaFunction cudaD3D9GetDirect3DDevice
    syntax keyword cudaFunction cudaD3D9SetDirect3DDevice
    syntax keyword cudaFunction cudaGraphicsD3D9RegisterResource

    " 4.16. Direct3D 10 Interoperability
    syntax keyword cudaType     cudaD3D10DeviceList
    syntax keyword cudaConstant cudaD3D10DeviceListAll cudaD3D10DeviceListCurrentFrame cudaD3D10DeviceListNextFrame
    syntax keyword cudaFunction cudaD3D10GetDevice
    syntax keyword cudaFunction cudaD3D10GetDevices
    syntax keyword cudaFunction cudaGraphicsD3D10RegisterResource

    " 4.18. Direct3D 11 Interoperability
    syntax keyword cudaType     cudaD3D11DeviceList
    syntax keyword cudaConstant cudaD3D11DeviceListAll cudaD3D11DeviceListCurrentFrame cudaD3D11DeviceListNextFrame
    syntax keyword cudaFunction cudaD3D11GetDevice
    syntax keyword cudaFunction cudaD3D11GetDevices
    syntax keyword cudaFunction cudaGraphicsD3D11RegisterResource

    " 4.20. VDPAU Interoperability
    syntax keyword cudaFunction cudaGraphicsVDPAURegisterOutputSurface
    syntax keyword cudaFunction cudaGraphicsVDPAURegisterVideoSurface
    syntax keyword cudaFunction cudaVDPAUGetDevice
    syntax keyword cudaFunction cudaVDPAUSetVDPAUDevice

    " 4.21. EGL Interoperability
    syntax keyword cudaFunction cudaEGLStreamConsumerAcquireFrame
    syntax keyword cudaFunction cudaEGLStreamConsumerConnect
    syntax keyword cudaFunction cudaEGLStreamConsumerConnectWithFlags
    syntax keyword cudaFunction cudaEGLStreamConsumerDisconnect
    syntax keyword cudaFunction cudaEGLStreamConsumerReleaseFrame
    syntax keyword cudaFunction cudaEGLStreamProducerConnect
    syntax keyword cudaFunction cudaEGLStreamProducerDisconnect
    syntax keyword cudaFunction cudaEGLStreamProducerPresentFrame
    syntax keyword cudaFunction cudaEGLStreamProducerReturnFrame
    syntax keyword cudaFunction cudaEventCreateFromEGLSync
    syntax keyword cudaFunction cudaGraphicsEGLRegisterImage
    syntax keyword cudaFunction cudaGraphicsResourceGetMappedEglFrame

    " 4.22. Graphics Interoperability
    syntax keyword cudaFunction cudaGraphicsMapResources
    syntax keyword cudaFunction cudaGraphicsResourceGetMappedMipmappedArray
    syntax keyword cudaFunction cudaGraphicsResourceGetMappedPointer
    syntax keyword cudaFunction cudaGraphicsResourceSetMapFlags
    syntax keyword cudaFunction cudaGraphicsSubResourceGetMappedArray
    syntax keyword cudaFunction cudaGraphicsUnmapResources
    syntax keyword cudaFunction cudaGraphicsUnregisterResource

    " 4.23. Texture Reference Management
    syntax keyword cudaFunction cudaBindTexture
    syntax keyword cudaFunction cudaBindTexture2D
    syntax keyword cudaFunction cudaBindTextureToArray
    syntax keyword cudaFunction cudaBindTextureToMipmappedArray
    syntax keyword cudaFunction cudaCreateChannelDesc
    syntax keyword cudaFunction cudaGetChannelDesc
    syntax keyword cudaFunction cudaGetTextureAlignmentOffset
    syntax keyword cudaFunction cudaGetTextureReference
    syntax keyword cudaFunction cudaUnbindTexture

    " 4.24. Surface Reference Management
    syntax keyword cudaFunction cudaBindSurfaceToArray
    syntax keyword cudaFunction cudaGetSurfaceReference

    " 4.25. Texture Object Management
    syntax keyword cudaFunction cudaCreateTextureObject
    syntax keyword cudaFunction cudaDestroyTextureObject
    syntax keyword cudaFunction cudaGetTextureObjectResourceDesc
    syntax keyword cudaFunction cudaGetTextureObjectResourceViewDesc
    syntax keyword cudaFunction cudaGetTextureObjectTextureDesc

    " 4.26. Surface Object Management
    syntax keyword cudaFunction cudaCreateSurfaceObject
    syntax keyword cudaFunction cudaDestroySurfaceObject
    syntax keyword cudaFunction cudaGetSurfaceObjectResourceDesc

    " 4.27. Version Management
    syntax keyword cudaFunction cudaDriverGetVersion
    syntax keyword cudaFunction cudaRuntimeGetVersion

    " 4.28. C++ API Routines
    syntax keyword cudaType __cudaOccupancyB2DHelper
    syntax keyword cudaFunction cudaEventCreate
    syntax keyword cudaFunction cudaMallocHost

    " Note: following functions are already listed above
    " The one listed here are the overloaded template versions
    " syntax keyword cudaFunction cudaBindSurfaceToArray
    " syntax keyword cudaFunction cudaBindTexture
    " syntax keyword cudaFunction cudaBindTexture2D
    " syntax keyword cudaFunction cudaBindTextureToArray
    " syntax keyword cudaFunction cudaBindTextureToMipmappedArray
    " syntax keyword cudaFunction cudaCreateChannelDesc
    " syntax keyword cudaFunction cudaFuncGetAttributes
    " syntax keyword cudaFunction cudaFuncSetAttribute
    " syntax keyword cudaFunction cudaFuncSetCacheConfig
    " syntax keyword cudaFunction cudaGetSymbolAddress
    " syntax keyword cudaFunction cudaGetSymbolSize
    " syntax keyword cudaFunction cudaGetTextureAlignmentOffset
    " syntax keyword cudaFunction cudaLaunchCooperativeKernel
    " syntax keyword cudaFunction cudaLaunchKernel
    " syntax keyword cudaFunction cudaMallocManaged
    " syntax keyword cudaFunction cudaMemcpyFromSymbol
    " syntax keyword cudaFunction cudaMemcpyFromSymbolAsync
    " syntax keyword cudaFunction cudaMemcpyToSymbol
    " syntax keyword cudaFunction cudaMemcpyToSymbolAsync
    " syntax keyword cudaFunction cudaOccupancyMaxActiveBlocksPerMultiprocessor
    " syntax keyword cudaFunction cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    " syntax keyword cudaFunction cudaStreamAttachMemAsync
    " syntax keyword cudaFunction cudaUnbindTexture
    syntax keyword cudaFunction cudaLaunch
    syntax keyword cudaFunction cudaSetupArgument
    syntax keyword cudaFunction cudaOccupancyMaxPotentialBlockSize
    syntax keyword cudaFunction cudaOccupancyMaxPotentialBlockSizeVariableSMem
    syntax keyword cudaFunction cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
    syntax keyword cudaFunction cudaOccupancyMaxPotentialBlockSizeWithFlags

    " 4.30. Profiler Control
    syntax keyword cudaFunction cudaProfilerInitialize
    syntax keyword cudaFunction cudaProfilerStart
    syntax keyword cudaFunction cudaProfilerStop

    " 4.31. Data types used by CUDA Runtime
    syntax keyword cudaType     cudaChannelFormatDesc
    syntax keyword cudaType     cudaDeviceProp
    syntax keyword cudaType     cudaEglFrame
    syntax keyword cudaType     cudaEglPlaneDesc
    syntax keyword cudaType     cudaExtent
    syntax keyword cudaType     cudaFuncAttributes
    syntax keyword cudaType     cudaIpcEventHandle_t
    syntax keyword cudaType     cudaIpcMemHandle_t
    syntax keyword cudaType     cudaLaunchParams
    syntax keyword cudaType     cudaMemcpy3DParms
    syntax keyword cudaType     cudaMemcpy3DPeerParms
    syntax keyword cudaType     cudaPitchedPtr
    syntax keyword cudaType     cudaPointerAttributes
    syntax keyword cudaType     cudaPos
    syntax keyword cudaType     cudaResourceDesc
    syntax keyword cudaType     cudaResourceViewDesc
    syntax keyword cudaType     cudaTextureDesc
    syntax keyword cudaType     surfaceReference
    syntax keyword cudaType     textureReference

    syntax keyword cudaConstant CUDA_EGL_MAX_PLANES
    syntax keyword cudaConstant CUDA_IPC_HANDLE_SIZE
    syntax keyword cudaConstant cudaArrayCubemap
    syntax keyword cudaConstant cudaArrayDefault
    syntax keyword cudaConstant cudaArrayLayered
    syntax keyword cudaConstant cudaArraySurfaceLoadStore
    syntax keyword cudaConstant cudaArrayTextureGather
    syntax keyword cudaConstant cudaCooperativeLaunchMultiDeviceNoPostSync
    syntax keyword cudaConstant cudaCooperativeLaunchMultiDeviceNoPreSync
    syntax keyword cudaConstant cudaCpuDeviceId
    syntax keyword cudaConstant cudaDeviceBlockingSync
    syntax keyword cudaConstant cudaDeviceLmemResizeToMax
    syntax keyword cudaConstant cudaDeviceMapHost
    syntax keyword cudaConstant cudaDeviceMask
    syntax keyword cudaConstant cudaDevicePropDontCare
    syntax keyword cudaConstant cudaDeviceScheduleAuto
    syntax keyword cudaConstant cudaDeviceScheduleBlockingSync
    syntax keyword cudaConstant cudaDeviceScheduleMask
    syntax keyword cudaConstant cudaDeviceScheduleSpin
    syntax keyword cudaConstant cudaDeviceScheduleYield
    syntax keyword cudaConstant cudaEventBlockingSync
    syntax keyword cudaConstant cudaEventDefault
    syntax keyword cudaConstant cudaEventDisableTiming
    syntax keyword cudaConstant cudaEventInterprocess
    syntax keyword cudaConstant cudaHostAllocDefault
    syntax keyword cudaConstant cudaHostAllocMapped
    syntax keyword cudaConstant cudaHostAllocPortable
    syntax keyword cudaConstant cudaHostAllocWriteCombined
    syntax keyword cudaConstant cudaHostRegisterDefault
    syntax keyword cudaConstant cudaHostRegisterIoMemory
    syntax keyword cudaConstant cudaHostRegisterMapped
    syntax keyword cudaConstant cudaHostRegisterPortable
    syntax keyword cudaConstant cudaInvalidDeviceId
    syntax keyword cudaConstant cudaIpcMemLazyEnablePeerAccess
    syntax keyword cudaConstant cudaMemAttachGlobal
    syntax keyword cudaConstant cudaMemAttachHost
    syntax keyword cudaConstant cudaMemAttachSingle
    syntax keyword cudaConstant cudaOccupancyDefault
    syntax keyword cudaConstant cudaOccupancyDisableCachingOverride
    syntax keyword cudaConstant cudaPeerAccessDefault
    syntax keyword cudaConstant cudaStreamDefault
    syntax keyword cudaConstant cudaStreamLegacy
    syntax keyword cudaConstant cudaStreamNonBlocking
    syntax keyword cudaConstant cudaStreamPerThread
    syntax keyword cudaType     cudaArray_const_t
    syntax keyword cudaType     cudaArray_t
    syntax keyword cudaType     cudaEglStreamConnection
    syntax keyword cudaType     cudaError_t
    syntax keyword cudaType     cudaEvent_t
    syntax keyword cudaType     cudaGraphicsResource_t
    syntax keyword cudaType     cudaMipmappedArray_const_t
    syntax keyword cudaType     cudaMipmappedArray_t
    syntax keyword cudaType     cudaOutputMode_t
    syntax keyword cudaType     cudaStream_t
    syntax keyword cudaType     cudaSurfaceObject_t
    syntax keyword cudaType     cudaTextureObject_t
    syntax keyword cudaType     cudaUUID_t

    syntax keyword cudaType     cudaArray
    syntax keyword cudaType     CUeglStreamConnection_st
    syntax keyword cudaType     enumcudaError
    syntax keyword cudaType     CUevent_st
    syntax keyword cudaType     cudaGraphicsResource
    syntax keyword cudaType     cudaMipmappedArray
    syntax keyword cudaType     enumcudaOutputMode
    syntax keyword cudaType     CUstream_st
    syntax keyword cudaType     CUuuid_st

    syntax keyword cudaType     cudaCGScope
    syntax keyword cudaConstant cudaCGScopeInvalid cudaCGScopeGrid cudaCGScopeMultiGrid
    syntax keyword cudaType     cudaChannelFormatKind
    syntax keyword cudaConstant cudaChannelFormatKindSigned cudaChannelFormatKindUnsigned cudaChannelFormatKindFloat cudaChannelFormatKindNone
    syntax keyword cudaType     cudaComputeMode
    syntax keyword cudaConstant cudaComputeModeDefault cudaComputeModeExclusive cudaComputeModeProhibited cudaComputeModeExclusiveProcess
    syntax keyword cudaType     cudaDeviceAttr
    syntax keyword cudaConstant cudaDevAttrMaxThreadsPerBlock cudaDevAttrMaxBlockDimX
    syntax keyword cudaConstant cudaDevAttrMaxBlockDimY cudaDevAttrMaxBlockDimZ
    syntax keyword cudaConstant cudaDevAttrMaxGridDimX cudaDevAttrMaxGridDimY
    syntax keyword cudaConstant cudaDevAttrMaxGridDimZ cudaDevAttrMaxSharedMemoryPerBlock
    syntax keyword cudaConstant cudaDevAttrTotalConstantMemory cudaDevAttrWarpSize
    syntax keyword cudaConstant cudaDevAttrMaxPitch cudaDevAttrMaxRegistersPerBlock
    syntax keyword cudaConstant cudaDevAttrClockRate cudaDevAttrTextureAlignment
    syntax keyword cudaConstant cudaDevAttrGpuOverlap cudaDevAttrMultiProcessorCount
    syntax keyword cudaConstant cudaDevAttrKernelExecTimeout cudaDevAttrIntegrated
    syntax keyword cudaConstant cudaDevAttrCanMapHostMemory cudaDevAttrComputeMode
    syntax keyword cudaConstant cudaDevAttrMaxTexture1DWidth cudaDevAttrMaxTexture2DWidth
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DHeight cudaDevAttrMaxTexture3DWidth
    syntax keyword cudaConstant cudaDevAttrMaxTexture3DHeight cudaDevAttrMaxTexture3DDepth
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DLayeredWidth cudaDevAttrMaxTexture2DLayeredHeight
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DLayeredLayers cudaDevAttrSurfaceAlignment
    syntax keyword cudaConstant cudaDevAttrConcurrentKernels cudaDevAttrEccEnabled
    syntax keyword cudaConstant cudaDevAttrPciBusId cudaDevAttrPciDeviceId
    syntax keyword cudaConstant cudaDevAttrTccDriver cudaDevAttrMemoryClockRate
    syntax keyword cudaConstant cudaDevAttrGlobalMemoryBusWidth cudaDevAttrL2CacheSize
    syntax keyword cudaConstant cudaDevAttrMaxThreadsPerMultiProcessor cudaDevAttrAsyncEngineCount
    syntax keyword cudaConstant cudaDevAttrUnifiedAddressing cudaDevAttrMaxTexture1DLayeredWidth
    syntax keyword cudaConstant cudaDevAttrMaxTexture1DLayeredLayers cudaDevAttrMaxTexture2DGatherWidth
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DGatherHeight cudaDevAttrMaxTexture3DWidthAlt
    syntax keyword cudaConstant cudaDevAttrMaxTexture3DHeightAlt cudaDevAttrMaxTexture3DDepthAlt
    syntax keyword cudaConstant cudaDevAttrPciDomainId cudaDevAttrTexturePitchAlignment
    syntax keyword cudaConstant cudaDevAttrMaxTextureCubemapWidth cudaDevAttrMaxTextureCubemapLayeredWidth
    syntax keyword cudaConstant cudaDevAttrMaxTextureCubemapLayeredLayers cudaDevAttrMaxSurface1DWidth
    syntax keyword cudaConstant cudaDevAttrMaxSurface2DWidth cudaDevAttrMaxSurface2DHeight
    syntax keyword cudaConstant cudaDevAttrMaxSurface3DWidth cudaDevAttrMaxSurface3DHeight
    syntax keyword cudaConstant cudaDevAttrMaxSurface3DDepth cudaDevAttrMaxSurface1DLayeredWidth
    syntax keyword cudaConstant cudaDevAttrMaxSurface1DLayeredLayers cudaDevAttrMaxSurface2DLayeredWidth
    syntax keyword cudaConstant cudaDevAttrMaxSurface2DLayeredHeight cudaDevAttrMaxSurface2DLayeredLayers
    syntax keyword cudaConstant cudaDevAttrMaxSurfaceCubemapWidth cudaDevAttrMaxSurfaceCubemapLayeredWidth
    syntax keyword cudaConstant cudaDevAttrMaxSurfaceCubemapLayeredLayers cudaDevAttrMaxTexture1DLinearWidth
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DLinearWidth cudaDevAttrMaxTexture2DLinearHeight
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DLinearPitch cudaDevAttrMaxTexture2DMipmappedWidth
    syntax keyword cudaConstant cudaDevAttrMaxTexture2DMipmappedHeight cudaDevAttrComputeCapabilityMajor
    syntax keyword cudaConstant cudaDevAttrComputeCapabilityMinor cudaDevAttrMaxTexture1DMipmappedWidth
    syntax keyword cudaConstant cudaDevAttrStreamPrioritiesSupported cudaDevAttrGlobalL1CacheSupported
    syntax keyword cudaConstant cudaDevAttrLocalL1CacheSupported cudaDevAttrMaxSharedMemoryPerMultiprocessor
    syntax keyword cudaConstant cudaDevAttrMaxRegistersPerMultiprocessor cudaDevAttrManagedMemory
    syntax keyword cudaConstant cudaDevAttrIsMultiGpuBoard cudaDevAttrMultiGpuBoardGroupID
    syntax keyword cudaConstant cudaDevAttrHostNativeAtomicSupported cudaDevAttrSingleToDoublePrecisionPerfRatio
    syntax keyword cudaConstant cudaDevAttrPageableMemoryAccess cudaDevAttrConcurrentManagedAccess
    syntax keyword cudaConstant cudaDevAttrComputePreemptionSupported cudaDevAttrCanUseHostPointerForRegisteredMem
    syntax keyword cudaConstant cudaDevAttrReserved92 cudaDevAttrReserved93
    syntax keyword cudaConstant cudaDevAttrReserved94 cudaDevAttrCooperativeLaunch
    syntax keyword cudaConstant cudaDevAttrCooperativeMultiDeviceLaunch cudaDevAttrMaxSharedMemoryPerBlockOptin

    syntax keyword cudaType     cudaDeviceP2PAttr
    syntax keyword cudaConstant cudaDevP2PAttrPerformanceRank cudaDevP2PAttrAccessSupported cudaDevP2PAttrNativeAtomicSupported
    syntax keyword cudaType     cudaEglColorFormat
    syntax keyword cudaConstant cudaEglColorFormatYUV420Planar cudaEglColorFormatYUV420SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatYUV422Planar cudaEglColorFormatYUV422SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatRGB cudaEglColorFormatBGR
    syntax keyword cudaConstant cudaEglColorFormatARGB cudaEglColorFormatRGBA
    syntax keyword cudaConstant cudaEglColorFormatL cudaEglColorFormatR
    syntax keyword cudaConstant cudaEglColorFormatYUV444Planar cudaEglColorFormatYUV444SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatYUYV422 cudaEglColorFormatUYVY422
    syntax keyword cudaConstant cudaEglColorFormatABGR cudaEglColorFormatBGRA
    syntax keyword cudaConstant cudaEglColorFormatA cudaEglColorFormatRG
    syntax keyword cudaConstant cudaEglColorFormatAYUV cudaEglColorFormatYVU444SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatYVU422SemiPlanar cudaEglColorFormatYVU420SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatY10V10U10_444SemiPlanar cudaEglColorFormatY10V10U10_420SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatY12V12U12_444SemiPlanar cudaEglColorFormatY12V12U12_420SemiPlanar
    syntax keyword cudaConstant cudaEglColorFormatVYUY_ER cudaEglColorFormatUYVY_ER
    syntax keyword cudaConstant cudaEglColorFormatYUYV_ER cudaEglColorFormatYVYU_ER
    syntax keyword cudaConstant cudaEglColorFormatYUV_ER cudaEglColorFormatYUVA_ER
    syntax keyword cudaConstant cudaEglColorFormatAYUV_ER cudaEglColorFormatYUV444Planar_ER
    syntax keyword cudaConstant cudaEglColorFormatYUV422Planar_ER cudaEglColorFormatYUV420Planar_ER
    syntax keyword cudaConstant cudaEglColorFormatYUV444SemiPlanar_ER cudaEglColorFormatYUV422SemiPlanar_ER
    syntax keyword cudaConstant cudaEglColorFormatYUV420SemiPlanar_ER cudaEglColorFormatYVU444Planar_ER
    syntax keyword cudaConstant cudaEglColorFormatYVU422Planar_ER cudaEglColorFormatYVU420Planar_ER
    syntax keyword cudaConstant cudaEglColorFormatYVU444SemiPlanar_ER cudaEglColorFormatYVU422SemiPlanar_ER
    syntax keyword cudaConstant cudaEglColorFormatYVU420SemiPlanar_ER cudaEglColorFormatBayerRGGB
    syntax keyword cudaConstant cudaEglColorFormatBayerBGGR cudaEglColorFormatBayerGRBG
    syntax keyword cudaConstant cudaEglColorFormatBayerGBRG cudaEglColorFormatBayer10RGGB
    syntax keyword cudaConstant cudaEglColorFormatBayer10BGGR cudaEglColorFormatBayer10GRBG
    syntax keyword cudaConstant cudaEglColorFormatBayer10GBRG cudaEglColorFormatBayer12RGGB
    syntax keyword cudaConstant cudaEglColorFormatBayer12BGGR cudaEglColorFormatBayer12GRBG
    syntax keyword cudaConstant cudaEglColorFormatBayer12GBRG cudaEglColorFormatBayer14RGGB
    syntax keyword cudaConstant cudaEglColorFormatBayer14BGGR cudaEglColorFormatBayer14GRBG
    syntax keyword cudaConstant cudaEglColorFormatBayer14GBRG cudaEglColorFormatBayer20RGGB
    syntax keyword cudaConstant cudaEglColorFormatBayer20BGGR cudaEglColorFormatBayer20GRBG
    syntax keyword cudaConstant cudaEglColorFormatBayer20GBRG cudaEglColorFormatYVU444Planar
    syntax keyword cudaConstant cudaEglColorFormatYVU422Planar cudaEglColorFormatYVU420Planar

    syntax keyword cudaType     cudaEglFrameType
    syntax keyword cudaConstant cudaEglFrameTypeArray cudaEglFrameTypePitch
    syntax keyword cudaType     cudaEglResourceLocationFlags
    syntax keyword cudaConstant cudaEglResourceLocationSysmem cudaEglResourceLocationVidmem
    syntax keyword cudaType     cudaError
    syntax keyword cudaConstant cudaSuccess
    syntax keyword cudaConstant cudaErrorDevicesUnavailable
    syntax keyword cudaConstant cudaErrorDuplicateSurfaceName
    syntax keyword cudaConstant cudaErrorDuplicateTextureName
    syntax keyword cudaConstant cudaErrorDuplicateVariableName
    syntax keyword cudaConstant cudaErrorECCUncorrectable
    syntax keyword cudaConstant cudaErrorIncompatibleDriverContext
    syntax keyword cudaConstant cudaErrorInitializationError
    syntax keyword cudaConstant cudaErrorInsufficientDriver
    syntax keyword cudaConstant cudaErrorInvalidChannelDescriptor
    syntax keyword cudaConstant cudaErrorInvalidConfiguration
    syntax keyword cudaConstant cudaErrorInvalidDevice
    syntax keyword cudaConstant cudaErrorInvalidDeviceFunction
    syntax keyword cudaConstant cudaErrorInvalidDevicePointer
    syntax keyword cudaConstant cudaErrorInvalidFilterSetting
    syntax keyword cudaConstant cudaErrorInvalidHostPointer
    syntax keyword cudaConstant cudaErrorInvalidKernelImage
    syntax keyword cudaConstant cudaErrorInvalidMemcpyDirection
    syntax keyword cudaConstant cudaErrorInvalidNormSetting
    syntax keyword cudaConstant cudaErrorInvalidPitchValue
    syntax keyword cudaConstant cudaErrorInvalidResourceHandle
    syntax keyword cudaConstant cudaErrorInvalidSurface
    syntax keyword cudaConstant cudaErrorInvalidSymbol
    syntax keyword cudaConstant cudaErrorInvalidTexture
    syntax keyword cudaConstant cudaErrorInvalidTextureBinding
    syntax keyword cudaConstant cudaErrorInvalidValue
    syntax keyword cudaConstant cudaErrorLaunchFailure
    syntax keyword cudaConstant cudaErrorLaunchOutOfResources
    syntax keyword cudaConstant cudaErrorLaunchTimeout
    syntax keyword cudaConstant cudaErrorMapBufferObjectFailed
    syntax keyword cudaConstant cudaErrorMemoryAllocation
    syntax keyword cudaConstant cudaErrorMissingConfiguration
    syntax keyword cudaConstant cudaErrorNoDevice
    syntax keyword cudaConstant cudaErrorNoKernelImageForDevice
    syntax keyword cudaConstant cudaErrorNotReady
    syntax keyword cudaConstant cudaErrorSetOnActiveProcess
    syntax keyword cudaConstant cudaErrorSharedObjectInitFailed
    syntax keyword cudaConstant cudaErrorSharedObjectSymbolNotFound
    syntax keyword cudaConstant cudaErrorStartupFailure
    syntax keyword cudaConstant cudaErrorUnknown
    syntax keyword cudaConstant cudaErrorUnmapBufferObjectFailed
    syntax keyword cudaConstant cudaErrorUnsupportedLimit

    syntax keyword cudaType     cudaFuncAttribute
    syntax keyword cudaConstant cudaFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributePreferredSharedMemoryCarveout cudaFuncAttributeMax
    syntax keyword cudaType     cudaFuncCache
    syntax keyword cudaConstant cudaFuncCachePreferNone cudaFuncCachePreferShared cudaFuncCachePreferL1 cudaFuncCachePreferEqual
    syntax keyword cudaType     cudaGraphicsCubeFace
    syntax keyword cudaConstant cudaGraphicsCubeFacePositiveX cudaGraphicsCubeFaceNegativeX cudaGraphicsCubeFacePositiveY cudaGraphicsCubeFaceNegativeY cudaGraphicsCubeFacePositiveZ cudaGraphicsCubeFaceNegativeZ
    syntax keyword cudaType     cudaGraphicsMapFlags
    syntax keyword cudaConstant cudaGraphicsMapFlagsNone cudaGraphicsMapFlagsReadOnly cudaGraphicsMapFlagsWriteDiscard
    syntax keyword cudaType     cudaGraphicsRegisterFlags
    syntax keyword cudaConstant cudaGraphicsRegisterFlagsNone cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsSurfaceLoadStore cudaGraphicsRegisterFlagsTextureGather
    syntax keyword cudaType     cudaLimit
    syntax keyword cudaConstant cudaLimitStackSize cudaLimitPrintfFifoSize cudaLimitMallocHeapSize cudaLimitDevRuntimeSyncDepth cudaLimitDevRuntimePendingLaunchCount
    syntax keyword cudaType     cudaMemRangeAttribute
    syntax keyword cudaConstant cudaMemRangeAttributeReadMostly cudaMemRangeAttributePreferredLocation cudaMemRangeAttributeAccessedBy cudaMemRangeAttributeLastPrefetchLocation
    syntax keyword cudaType     cudaMemcpyKind
    syntax keyword cudaConstant cudaMemcpyHostToHost cudaMemcpyHostToDevice cudaMemcpyDeviceToHost cudaMemcpyDeviceToDevice cudaMemcpyDefault
    syntax keyword cudaType     cudaMemoryAdvise
    syntax keyword cudaConstant cudaMemAdviseSetReadMostly cudaMemAdviseUnsetReadMostly cudaMemAdviseSetPreferredLocation cudaMemAdviseUnsetPreferredLocation cudaMemAdviseSetAccessedBy cudaMemAdviseUnsetAccessedBy
    syntax keyword cudaType     cudaMemoryType
    syntax keyword cudaConstant cudaMemoryTypeHost cudaMemoryTypeDevice
    syntax keyword cudaType     cudaOutputMode
    syntax keyword cudaConstant cudaKeyValuePair cudaCSV
    syntax keyword cudaType     cudaResourceType
    syntax keyword cudaConstant cudaResourceTypeArray cudaResourceTypeMipmappedArray cudaResourceTypeLinear cudaResourceTypePitch2D
    syntax keyword cudaType     cudaResourceViewFormat
    syntax keyword cudaConstant cudaResViewFormatNone cudaResViewFormatUnsignedChar1
    syntax keyword cudaConstant cudaResViewFormatUnsignedChar2 cudaResViewFormatUnsignedChar4
    syntax keyword cudaConstant cudaResViewFormatSignedChar1 cudaResViewFormatSignedChar2
    syntax keyword cudaConstant cudaResViewFormatSignedChar4 cudaResViewFormatUnsignedShort1
    syntax keyword cudaConstant cudaResViewFormatUnsignedShort2 cudaResViewFormatUnsignedShort4
    syntax keyword cudaConstant cudaResViewFormatSignedShort1 cudaResViewFormatSignedShort2
    syntax keyword cudaConstant cudaResViewFormatSignedShort4 cudaResViewFormatUnsignedInt1
    syntax keyword cudaConstant cudaResViewFormatUnsignedInt2 cudaResViewFormatUnsignedInt4
    syntax keyword cudaConstant cudaResViewFormatSignedInt1 cudaResViewFormatSignedInt2
    syntax keyword cudaConstant cudaResViewFormatSignedInt4 cudaResViewFormatHalf1
    syntax keyword cudaConstant cudaResViewFormatHalf2 cudaResViewFormatHalf4
    syntax keyword cudaConstant cudaResViewFormatFloat1 cudaResViewFormatFloat2
    syntax keyword cudaConstant cudaResViewFormatFloat4 cudaResViewFormatUnsignedBlockCompressed1
    syntax keyword cudaConstant cudaResViewFormatUnsignedBlockCompressed2 cudaResViewFormatUnsignedBlockCompressed3
    syntax keyword cudaConstant cudaResViewFormatUnsignedBlockCompressed4 cudaResViewFormatSignedBlockCompressed4
    syntax keyword cudaConstant cudaResViewFormatUnsignedBlockCompressed5 cudaResViewFormatSignedBlockCompressed5
    syntax keyword cudaConstant cudaResViewFormatUnsignedBlockCompressed6H cudaResViewFormatSignedBlockCompressed6H
    syntax keyword cudaConstant cudaResViewFormatUnsignedBlockCompressed7
    syntax keyword cudaType     cudaSharedCarveout
    syntax keyword cudaConstant cudaSharedmemCarveoutDefault cudaSharedmemCarveoutMaxShared cudaSharedmemCarveoutMaxL1
    syntax keyword cudaType     cudaSharedMemConfig
    syntax keyword cudaConstant cudaSharedMemBankSizeDefault cudaSharedMemBankSizeFourByte cudaSharedMemBankSizeEightByte
    syntax keyword cudaType     cudaSurfaceBoundaryMode
    syntax keyword cudaConstant cudaBoundaryModeZero cudaBoundaryModeClamp cudaBoundaryModeTrap
    syntax keyword cudaType     cudaSurfaceFormatMode
    syntax keyword cudaConstant cudaFormatModeForced cudaFormatModeAuto
    syntax keyword cudaType     cudaTextureAddressMode
    syntax keyword cudaConstant cudaAddressModeWrap cudaAddressModeClamp cudaAddressModeMirror cudaAddressModeBorder
    syntax keyword cudaType     cudaTextureFilterMode
    syntax keyword cudaConstant cudaFilterModePoint cudaFilterModeLinear
    syntax keyword cudaType     cudaTextureReadMode
    syntax keyword cudaConstant cudaReadModeElementType cudaReadModeNormalizedFloat

    " 6. Data Fields
    " Currently those are not used because of potential keyword collisions,
    " for example, the data field 'array' and std::array
    " TODO: find a way to highlight these keywords only when they appear after
    " a dot or arrow, like x.depth or x->depth, but not on their own
    " syntax keyword cudaMember ECCEnabled contained
    " syntax keyword cudaMember asyncEngineCount contained
    " syntax keyword cudaMember canMapHostMemory contained
    " syntax keyword cudaMember canUseHostPointerForRegisteredMem contained
    " syntax keyword cudaMember clockRate contained
    " syntax keyword cudaMember computeMode contained
    " syntax keyword cudaMember computePreemptionSupported contained
    " syntax keyword cudaMember concurrentKernels contained
    " syntax keyword cudaMember concurrentManagedAccess contained
    " syntax keyword cudaMember cooperativeLaunch contained
    " syntax keyword cudaMember cooperativeMultiDeviceLaunch contained
    " syntax keyword cudaMember deviceOverlap contained
    " syntax keyword cudaMember globalL1CacheSupported contained
    " syntax keyword cudaMember hostNativeAtomicSupported contained
    " syntax keyword cudaMember integrated contained
    " syntax keyword cudaMember isMultiGpuBoard contained
    " syntax keyword cudaMember kernelExecTimeoutEnabled contained
    " syntax keyword cudaMember l2CacheSize contained
    " syntax keyword cudaMember localL1CacheSupported contained
    " syntax keyword cudaMember managedMemory contained
    " syntax keyword cudaMember maxGridSize contained
    " syntax keyword cudaMember maxSurface1D contained
    " syntax keyword cudaMember maxSurface1DLayered contained
    " syntax keyword cudaMember maxSurface2D contained
    " syntax keyword cudaMember maxSurface2DLayered contained
    " syntax keyword cudaMember maxSurface3D contained
    " syntax keyword cudaMember maxSurfaceCubemap contained
    " syntax keyword cudaMember maxSurfaceCubemapLayered contained
    " syntax keyword cudaMember maxTexture1D contained
    " syntax keyword cudaMember maxTexture1DLayered contained
    " syntax keyword cudaMember maxTexture1DLinear contained
    " syntax keyword cudaMember maxTexture1DMipmap contained
    " syntax keyword cudaMember maxTexture2D contained
    " syntax keyword cudaMember maxTexture2DGather contained
    " syntax keyword cudaMember maxTexture2DLayered contained
    " syntax keyword cudaMember maxTexture2DLinear contained
    " syntax keyword cudaMember maxTexture2DMipmap contained
    " syntax keyword cudaMember maxTexture3D contained
    " syntax keyword cudaMember maxTexture3DAlt contained
    " syntax keyword cudaMember maxTextureCubemap contained
    " syntax keyword cudaMember maxTextureCubemapLayered contained
    " syntax keyword cudaMember maxThreadsDim contained
    " syntax keyword cudaMember maxThreadsPerBlock contained
    " syntax keyword cudaMember maxThreadsPerMultiProcessor contained
    " syntax keyword cudaMember memPitch contained
    " syntax keyword cudaMember memoryBusWidth contained
    " syntax keyword cudaMember memoryClockRate contained
    " syntax keyword cudaMember multiGpuBoardGroupID contained
    " syntax keyword cudaMember multiProcessorCount contained
    " syntax keyword cudaMember pageableMemoryAccess contained
    " syntax keyword cudaMember pciBusID contained
    " syntax keyword cudaMember pciDeviceID contained
    " syntax keyword cudaMember pciDomainID contained
    " syntax keyword cudaMember regsPerBlock contained
    " syntax keyword cudaMember regsPerMultiprocessor contained
    " syntax keyword cudaMember sharedMemPerBlock contained
    " syntax keyword cudaMember sharedMemPerBlockOptin contained
    " syntax keyword cudaMember sharedMemPerMultiprocessor contained
    " syntax keyword cudaMember singleToDoublePrecisionPerfRatio contained
    " syntax keyword cudaMember streamPrioritiesSupported contained
    " syntax keyword cudaMember surfaceAlignment contained
    " syntax keyword cudaMember tccDriver contained
    " syntax keyword cudaMember textureAlignment contained
    " syntax keyword cudaMember texturePitchAlignment contained
    " syntax keyword cudaMember totalConstMem contained
    " syntax keyword cudaMember totalGlobalMem contained
    " syntax keyword cudaMember unifiedAddressing contained
    " syntax keyword cudaMember major minor name contained
    " syntax keyword cudaMember eglColorFormat contained
    " syntax keyword cudaMember frameType contained
    " syntax keyword cudaMember pArray contained
    " syntax keyword cudaMember pPitch contained
    " syntax keyword cudaMember planeCount contained
    " syntax keyword cudaMember planeDesc contained
    " syntax keyword cudaMember pitch numChannels channelDesc reserved contained
    " syntax keyword cudaMember binaryVersion contained
    " syntax keyword cudaMember cacheModeCA contained
    " syntax keyword cudaMember constSizeBytes contained
    " syntax keyword cudaMember localSizeBytes contained
    " syntax keyword cudaMember maxDynamicSharedSizeBytes contained
    " syntax keyword cudaMember maxThreadsPerBlock contained
    " syntax keyword cudaMember numRegs contained
    " syntax keyword cudaMember preferredShmemCarveout contained
    " syntax keyword cudaMember ptxVersion contained
    " syntax keyword cudaMember sharedSizeBytes contained
    " syntax keyword cudaMember sharedMem contained
    " syntax keyword cudaMember stream contained
    " syntax keyword cudaMember dstArray contained
    " syntax keyword cudaMember dstPos contained
    " syntax keyword cudaMember dstPtr contained
    " syntax keyword cudaMember srcArray contained
    " syntax keyword cudaMember srcPos contained
    " syntax keyword cudaMember srcPtr contained
    " syntax keyword cudaMember extent contained
    " syntax keyword cudaMember kind contained
    " syntax keyword cudaMember dstDevice contained
    " syntax keyword cudaMember srcDevice contained
    " syntax keyword cudaMember devicePointer contained
    " syntax keyword cudaMember hostPointer contained
    " syntax keyword cudaMember isManaged contained
    " syntax keyword cudaMember memoryType contained
    " syntax keyword cudaMember device contained
    " syntax keyword cudaMember devPtr contained
    " syntax keyword cudaMember mipmap contained
    " syntax keyword cudaMember pitchInBytes contained
    " syntax keyword cudaMember resType contained
    " syntax keyword cudaMember sizeInBytes contained
    " syntax keyword cudaMember desc contained
    " syntax keyword cudaMember array contained
    " syntax keyword cudaMember firstLayer contained
    " syntax keyword cudaMember firstMipmapLevel contained
    " syntax keyword cudaMember lastLayer contained
    " syntax keyword cudaMember lastMipmapLevel contained
    " syntax keyword cudaMember format contained
    " syntax keyword cudaMember addressMode contained
    " syntax keyword cudaMember borderColor contained
    " syntax keyword cudaMember filterMode contained
    " syntax keyword cudaMember maxAnisotropy contained
    " syntax keyword cudaMember maxMipmapLevelClamp contained
    " syntax keyword cudaMember minMipmapLevelClamp contained
    " syntax keyword cudaMember mipmapFilterMode contained
    " syntax keyword cudaMember mipmapLevelBias contained
    " syntax keyword cudaMember normalizedCoords contained
    " syntax keyword cudaMember readMode contained
    " syntax keyword cudaMember sRGB contained
    " syntax keyword cudaMember xsize ysize contained
    " syntax keyword cudaMember width height depth contained
    " syntax keyword cudaMember func contained
    " syntax keyword cudaMember ptr contained
    " syntax keyword cudaMember x y z contained
    " Already added above as global variable
    " syntax keyword cudaMember warpSize blockDim gridDim
endif " g:cuda_runtime_api_highlight


" CUDA Driver API {{{1
" Based on: http://docs.nvidia.com/cuda/cuda-driver-api (v9.1.85, Jan 24, 2018)

if exists('g:cuda_driver_api_highlight') && g:cuda_driver_api_highlight
    " 4.1 Data types
    syntax keyword cudaType     CUDA_ARRAY3D_DESCRIPTOR
    syntax keyword cudaType     CUDA_ARRAY_DESCRIPTOR
    syntax keyword cudaType     CUDA_LAUNCH_PARAMS
    syntax keyword cudaType     CUDA_MEMCPY2D
    syntax keyword cudaType     CUDA_MEMCPY3D
    syntax keyword cudaType     CUDA_MEMCPY3D_PEER
    syntax keyword cudaType     CUDA_POINTER_ATTRIBUTE_P2P_TOKENS
    syntax keyword cudaType     CUDA_RESOURCE_DESC
    syntax keyword cudaType     CUDA_RESOURCE_VIEW_DESC
    syntax keyword cudaType     CUDA_TEXTURE_DESC
    syntax keyword cudaType     CUdevprop
    syntax keyword cudaType     CUeglFrame
    syntax keyword cudaType     CUipcEventHandle
    syntax keyword cudaType     CUipcMemHandle
    syntax keyword cudaType     CUstreamBatchMemOpParams
    syntax keyword cudaConstant CUDA_ARRAY3D_2DARRAY
    syntax keyword cudaConstant CUDA_ARRAY3D_CUBEMAP
    syntax keyword cudaConstant CUDA_ARRAY3D_DEPTH_TEXTURE
    syntax keyword cudaConstant CUDA_ARRAY3D_LAYERED
    syntax keyword cudaConstant CUDA_ARRAY3D_SURFACE_LDST
    syntax keyword cudaConstant CUDA_ARRAY3D_TEXTURE_GATHER
    syntax keyword cudaConstant CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC
    syntax keyword cudaConstant CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC
    syntax keyword cudaConstant CUDA_VERSION
    syntax keyword cudaConstant CU_DEVICE_CPU
    syntax keyword cudaConstant CU_DEVICE_INVALID
    syntax keyword cudaConstant CU_IPC_HANDLE_SIZE
    syntax keyword cudaConstant CU_LAUNCH_PARAM_BUFFER_POINTER
    syntax keyword cudaConstant CU_LAUNCH_PARAM_BUFFER_SIZE
    syntax keyword cudaConstant CU_LAUNCH_PARAM_END
    syntax keyword cudaConstant CU_MEMHOSTALLOC_DEVICEMAP
    syntax keyword cudaConstant CU_MEMHOSTALLOC_PORTABLE
    syntax keyword cudaConstant CU_MEMHOSTALLOC_WRITECOMBINED
    syntax keyword cudaConstant CU_MEMHOSTREGISTER_DEVICEMAP
    syntax keyword cudaConstant CU_MEMHOSTREGISTER_IOMEMORY
    syntax keyword cudaConstant CU_MEMHOSTREGISTER_PORTABLE
    syntax keyword cudaConstant CU_PARAM_TR_DEFAULT
    syntax keyword cudaConstant CU_STREAM_LEGACY
    syntax keyword cudaConstant CU_STREAM_PER_THREAD
    syntax keyword cudaConstant CU_TRSA_OVERRIDE_FORMAT
    syntax keyword cudaConstant CU_TRSF_NORMALIZED_COORDINATES
    syntax keyword cudaConstant CU_TRSF_READ_AS_INTEGER
    syntax keyword cudaConstant CU_TRSF_SRGB
    syntax keyword cudaConstant MAX_PLANES
    syntax keyword cudaType     CUarray
    syntax keyword cudaType     CUcontext
    syntax keyword cudaType     CUdevice
    syntax keyword cudaType     CUdeviceptr
    syntax keyword cudaType     CUeglStreamConnection
    syntax keyword cudaType     CUevent
    syntax keyword cudaType     CUfunction
    syntax keyword cudaType     CUgraphicsResource
    syntax keyword cudaType     CUmipmappedArray
    syntax keyword cudaType     CUmodule
    syntax keyword cudaType     CUoccupancyB2DSize
    syntax keyword cudaType     CUstream
    syntax keyword cudaType     CUstreamCallback
    syntax keyword cudaType     CUsurfObject
    syntax keyword cudaType     CUsurfref
    syntax keyword cudaType     CUtexObject
    syntax keyword cudaType     CUtexref

    syntax keyword cudaType     CUarray_st
    syntax keyword cudaType     CUctx_st
    syntax keyword cudaType     CUfunc_st
    syntax keyword cudaType     CUgraphicsResource_st
    syntax keyword cudaType     CUmipmappedArray_st
    syntax keyword cudaType     CUmod_st
    syntax keyword cudaType     CUsurfref_st
    syntax keyword cudaType     CUtexref_st

    syntax keyword cudaType     CUaddress_mode
    syntax keyword cudaConstant CU_TR_ADDRESS_MODE_WRAP CU_TR_ADDRESS_MODE_CLAMP CU_TR_ADDRESS_MODE_MIRROR CU_TR_ADDRESS_MODE_BORDER
    syntax keyword cudaType     CUarray_cubemap_face
    syntax keyword cudaConstant CU_CUBEMAP_FACE_POSITIVE_X CU_CUBEMAP_FACE_NEGATIVE_X CU_CUBEMAP_FACE_POSITIVE_Y CU_CUBEMAP_FACE_NEGATIVE_Y CU_CUBEMAP_FACE_POSITIVE_Z CU_CUBEMAP_FACE_NEGATIVE_Z
    syntax keyword cudaType     CUarray_format
    syntax keyword cudaConstant CU_AD_FORMAT_UNSIGNED_INT8 CU_AD_FORMAT_UNSIGNED_INT16 CU_AD_FORMAT_UNSIGNED_INT32 CU_AD_FORMAT_SIGNED_INT8 CU_AD_FORMAT_SIGNED_INT16 CU_AD_FORMAT_SIGNED_INT32 CU_AD_FORMAT_HALF CU_AD_FORMAT_FLOAT
    syntax keyword cudaType     CUcomputemode
    syntax keyword cudaConstant CU_COMPUTEMODE_DEFAULT CU_COMPUTEMODE_PROHIBITED CU_COMPUTEMODE_EXCLUSIVE_PROCESS
    syntax keyword cudaType     CUctx_flags
    syntax keyword cudaConstant CU_CTX_SCHED_AUTO CU_CTX_SCHED_SPIN CU_CTX_SCHED_YIELD CU_CTX_SCHED_BLOCKING_SYNC CU_CTX_SCHED_MASK CU_CTX_MAP_HOST CU_CTX_LMEM_RESIZE_TO_MAX CU_CTX_FLAGS_MASK
    syntax keyword cudaType     CUdevice_P2PAttribute
    syntax keyword cudaConstant CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED

    syntax keyword cudaType     CUdevice_attribute
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_WARP_SIZE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_PITCH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CLOCK_RATE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_INTEGRATED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_ECC_ENABLED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_TCC_DRIVER
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    syntax keyword cudaConstant CU_DEVICE_ATTRIBUTE_MAX

    syntax keyword cudaType     CUeglColorFormat
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV420_PLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV422_PLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_RGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_ARGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_RGBA
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_L
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_R
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV444_PLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUYV_422
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_UYVY_422
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_ABGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BGRA
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_A
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_RG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_AYUV
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_VYUY_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_UYVY_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUYV_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVYU_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUVA_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_AYUV_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER_RGGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER_BGGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER_GRBG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER_GBRG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER10_RGGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER10_BGGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER10_GRBG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER10_GBRG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER12_RGGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER12_BGGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER12_GRBG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER12_GBRG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER14_RGGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER14_BGGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER14_GRBG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER14_GBRG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER20_RGGB
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER20_BGGR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER20_GRBG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_BAYER20_GBRG
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU444_PLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU422_PLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_YVU420_PLANAR
    syntax keyword cudaConstant CU_EGL_COLOR_FORMAT_MAX

    syntax keyword cudaType     CUeglFrameType
    syntax keyword cudaConstant CU_EGL_FRAME_TYPE_ARRAY CU_EGL_FRAME_TYPE_PITCH
    syntax keyword cudaType     CUeglResourceLocationFlags
    syntax keyword cudaConstant CU_EGL_RESOURCE_LOCATION_SYSMEM CU_EGL_RESOURCE_LOCATION_VIDMEM
    syntax keyword cudaType     CUevent_flags
    syntax keyword cudaConstant CU_EVENT_DEFAULT CU_EVENT_BLOCKING_SYNC CU_EVENT_DISABLE_TIMING CU_EVENT_INTERPROCESS
    syntax keyword cudaType     CUfilter_mode
    syntax keyword cudaConstant CU_TR_FILTER_MODE_POINT CU_TR_FILTER_MODE_LINEAR
    syntax keyword cudaType     CUfunc_cache
    syntax keyword cudaConstant CU_FUNC_CACHE_PREFER_NONE CU_FUNC_CACHE_PREFER_SHARED CU_FUNC_CACHE_PREFER_L1 CU_FUNC_CACHE_PREFER_EQUAL
    syntax keyword cudaType     CUfunction_attribute
    syntax keyword cudaConstant CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES CU_FUNC_ATTRIBUTE_NUM_REGS CU_FUNC_ATTRIBUTE_PTX_VERSION CU_FUNC_ATTRIBUTE_BINARY_VERSION CU_FUNC_ATTRIBUTE_CACHE_MODE_CA CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT CU_FUNC_ATTRIBUTE_MAX
    syntax keyword cudaType     CUgraphicsMapResourceFlags
    syntax keyword cudaConstant CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD
    syntax keyword cudaType     CUgraphicsRegisterFlags
    syntax keyword cudaConstant CU_GRAPHICS_REGISTER_FLAGS_NONE CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER
    syntax keyword cudaType     CUipcMem_flags
    syntax keyword cudaConstant CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
    syntax keyword cudaType     CUjitInputType
    syntax keyword cudaConstant CU_JIT_INPUT_CUBIN CU_JIT_INPUT_PTX CU_JIT_INPUT_FATBINARY CU_JIT_INPUT_OBJECT CU_JIT_INPUT_LIBRARY CU_JIT_NUM_INPUT_TYPES
    syntax keyword cudaType     CUjit_cacheMode
    syntax keyword cudaConstant CU_JIT_CACHE_OPTION_NONE CU_JIT_CACHE_OPTION_CG CU_JIT_CACHE_OPTION_CA
    syntax keyword cudaType     CUjit_fallback
    syntax keyword cudaConstant CU_PREFER_PTX CU_PREFER_BINARY
    syntax keyword cudaType     CUjit_option
    syntax keyword cudaConstant CU_JIT_MAX_REGISTERS CU_JIT_THREADS_PER_BLOCK CU_JIT_WALL_TIME CU_JIT_INFO_LOG_BUFFER CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES CU_JIT_ERROR_LOG_BUFFER CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES CU_JIT_OPTIMIZATION_LEVEL CU_JIT_TARGET_FROM_CUCONTEXT CU_JIT_TARGET CU_JIT_FALLBACK_STRATEGY CU_JIT_GENERATE_DEBUG_INFO CU_JIT_LOG_VERBOSE CU_JIT_GENERATE_LINE_INFO CU_JIT_CACHE_MODE CU_JIT_NEW_SM3X_OPT CU_JIT_FAST_COMPILE CU_JIT_NUM_OPTIONS
    syntax keyword cudaType     CUjit_target
    syntax keyword cudaConstant CU_TARGET_COMPUTE_20 CU_TARGET_COMPUTE_21 CU_TARGET_COMPUTE_30 CU_TARGET_COMPUTE_32 CU_TARGET_COMPUTE_35 CU_TARGET_COMPUTE_37 CU_TARGET_COMPUTE_50 CU_TARGET_COMPUTE_52 CU_TARGET_COMPUTE_53 CU_TARGET_COMPUTE_60 CU_TARGET_COMPUTE_61 CU_TARGET_COMPUTE_62 CU_TARGET_COMPUTE_70 CU_TARGET_COMPUTE_73 CU_TARGET_COMPUTE_75
    syntax keyword cudaType     CUlimit
    syntax keyword cudaConstant CU_LIMIT_STACK_SIZE CU_LIMIT_PRINTF_FIFO_SIZE CU_LIMIT_MALLOC_HEAP_SIZE CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT CU_LIMIT_MAX
    syntax keyword cudaType     CUmemAttach_flags
    syntax keyword cudaConstant CU_MEM_ATTACH_GLOBAL CU_MEM_ATTACH_HOST CU_MEM_ATTACH_SINGLE
    syntax keyword cudaType     CUmem_advise
    syntax keyword cudaConstant CU_MEM_ADVISE_SET_READ_MOSTLY CU_MEM_ADVISE_UNSET_READ_MOSTLY CU_MEM_ADVISE_SET_PREFERRED_LOCATION CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION CU_MEM_ADVISE_SET_ACCESSED_BY CU_MEM_ADVISE_UNSET_ACCESSED_BY
    syntax keyword cudaType     CUmemorytype
    syntax keyword cudaConstant CU_MEMORYTYPE_HOST CU_MEMORYTYPE_DEVICE CU_MEMORYTYPE_ARRAY CU_MEMORYTYPE_UNIFIED
    syntax keyword cudaType     CUoccupancy_flags
    syntax keyword cudaConstant CU_OCCUPANCY_DEFAULT CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
    syntax keyword cudaType     CUpointer_attribute
    syntax keyword cudaConstant CU_POINTER_ATTRIBUTE_CONTEXT CU_POINTER_ATTRIBUTE_MEMORY_TYPE CU_POINTER_ATTRIBUTE_DEVICE_POINTER CU_POINTER_ATTRIBUTE_HOST_POINTER CU_POINTER_ATTRIBUTE_P2P_TOKENS CU_POINTER_ATTRIBUTE_SYNC_MEMOPS CU_POINTER_ATTRIBUTE_BUFFER_ID CU_POINTER_ATTRIBUTE_IS_MANAGED

    syntax keyword cudaType     CUresourceViewFormat
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_NONE
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_1X8
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_2X8
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_4X8
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_1X8
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_2X8
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_4X8
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_1X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_2X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_4X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_1X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_2X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_4X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_1X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_2X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UINT_4X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_1X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_2X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SINT_4X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_FLOAT_1X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_FLOAT_2X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_FLOAT_4X16
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_FLOAT_1X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_FLOAT_2X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_FLOAT_4X32
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC1
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC2
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC3
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC4
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SIGNED_BC4
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC5
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SIGNED_BC5
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC6H
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_SIGNED_BC6H
    syntax keyword cudaConstant CU_RES_VIEW_FORMAT_UNSIGNED_BC7

    syntax keyword cudaType     CUresourcetype
    syntax keyword cudaConstant CU_RESOURCE_TYPE_ARRAY CU_RESOURCE_TYPE_MIPMAPPED_ARRAY CU_RESOURCE_TYPE_LINEAR CU_RESOURCE_TYPE_PITCH2D

    syntax keyword cudaType     CUresult
    syntax keyword cudaConstant CUDA_SUCCESS
    syntax keyword cudaConstant CUDA_ERROR_INVALID_VALUE
    syntax keyword cudaConstant CUDA_ERROR_OUT_OF_MEMORY
    syntax keyword cudaConstant CUDA_ERROR_NOT_INITIALIZED
    syntax keyword cudaConstant CUDA_ERROR_DEINITIALIZED
    syntax keyword cudaConstant CUDA_ERROR_PROFILER_DISABLED
    syntax keyword cudaConstant CUDA_ERROR_NO_DEVICE
    syntax keyword cudaConstant CUDA_ERROR_INVALID_DEVICE
    syntax keyword cudaConstant CUDA_ERROR_INVALID_IMAGE
    syntax keyword cudaConstant CUDA_ERROR_INVALID_CONTEXT
    syntax keyword cudaConstant CUDA_ERROR_CONTEXT_ALREADY_CURRENT
    syntax keyword cudaConstant CUDA_ERROR_MAP_FAILED
    syntax keyword cudaConstant CUDA_ERROR_UNMAP_FAILED
    syntax keyword cudaConstant CUDA_ERROR_ARRAY_IS_MAPPED
    syntax keyword cudaConstant CUDA_ERROR_ALREADY_MAPPED
    syntax keyword cudaConstant CUDA_ERROR_NO_BINARY_FOR_GPU
    syntax keyword cudaConstant CUDA_ERROR_ALREADY_ACQUIRED
    syntax keyword cudaConstant CUDA_ERROR_NOT_MAPPED
    syntax keyword cudaConstant CUDA_ERROR_NOT_MAPPED_AS_ARRAY
    syntax keyword cudaConstant CUDA_ERROR_NOT_MAPPED_AS_POINTER
    syntax keyword cudaConstant CUDA_ERROR_ECC_UNCORRECTABLE
    syntax keyword cudaConstant CUDA_ERROR_UNSUPPORTED_LIMIT
    syntax keyword cudaConstant CUDA_ERROR_CONTEXT_ALREADY_IN_USE
    syntax keyword cudaConstant CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
    syntax keyword cudaConstant CUDA_ERROR_INVALID_PTX
    syntax keyword cudaConstant CUDA_ERROR_INVALID_GRAPHICS_CONTEXT
    syntax keyword cudaConstant CUDA_ERROR_NVLINK_UNCORRECTABLE
    syntax keyword cudaConstant CUDA_ERROR_JIT_COMPILER_NOT_FOUND
    syntax keyword cudaConstant CUDA_ERROR_INVALID_SOURCE
    syntax keyword cudaConstant CUDA_ERROR_FILE_NOT_FOUND
    syntax keyword cudaConstant CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
    syntax keyword cudaConstant CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
    syntax keyword cudaConstant CUDA_ERROR_OPERATING_SYSTEM
    syntax keyword cudaConstant CUDA_ERROR_INVALID_HANDLE
    syntax keyword cudaConstant CUDA_ERROR_NOT_FOUND
    syntax keyword cudaConstant CUDA_ERROR_NOT_READY
    syntax keyword cudaConstant CUDA_ERROR_ILLEGAL_ADDRESS
    syntax keyword cudaConstant CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
    syntax keyword cudaConstant CUDA_ERROR_LAUNCH_TIMEOUT
    syntax keyword cudaConstant CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING
    syntax keyword cudaConstant CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
    syntax keyword cudaConstant CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
    syntax keyword cudaConstant CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
    syntax keyword cudaConstant CUDA_ERROR_CONTEXT_IS_DESTROYED
    syntax keyword cudaConstant CUDA_ERROR_ASSERT
    syntax keyword cudaConstant CUDA_ERROR_TOO_MANY_PEERS
    syntax keyword cudaConstant CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
    syntax keyword cudaConstant CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
    syntax keyword cudaConstant CUDA_ERROR_HARDWARE_STACK_ERROR
    syntax keyword cudaConstant CUDA_ERROR_ILLEGAL_INSTRUCTION
    syntax keyword cudaConstant CUDA_ERROR_MISALIGNED_ADDRESS
    syntax keyword cudaConstant CUDA_ERROR_INVALID_ADDRESS_SPACE
    syntax keyword cudaConstant CUDA_ERROR_INVALID_PC
    syntax keyword cudaConstant CUDA_ERROR_LAUNCH_FAILED
    syntax keyword cudaConstant CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
    syntax keyword cudaConstant CUDA_ERROR_NOT_PERMITTED
    syntax keyword cudaConstant CUDA_ERROR_NOT_SUPPORTED
    syntax keyword cudaConstant CUDA_ERROR_UNKNOWN

    syntax keyword cudaType     CUshared_carveout
    syntax keyword cudaConstant CU_SHAREDMEM_CARVEOUT_DEFAULT CU_SHAREDMEM_CARVEOUT_MAX_SHARED CU_SHAREDMEM_CARVEOUT_MAX_L1
    syntax keyword cudaType     CUsharedconfig
    syntax keyword cudaConstant CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
    syntax keyword cudaType     CUstreamBatchMemOpType
    syntax keyword cudaConstant CU_STREAM_MEM_OP_WAIT_VALUE_32 CU_STREAM_MEM_OP_WRITE_VALUE_32 CU_STREAM_MEM_OP_WAIT_VALUE_64 CU_STREAM_MEM_OP_WRITE_VALUE_64 CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES
    syntax keyword cudaType     CUstreamWaitValue_flags
    syntax keyword cudaConstant CU_STREAM_WAIT_VALUE_GEQ CU_STREAM_WAIT_VALUE_EQ CU_STREAM_WAIT_VALUE_AND CU_STREAM_WAIT_VALUE_NOR CU_STREAM_WAIT_VALUE_FLUSH
    syntax keyword cudaType     CUstreamWriteValue_flags
    syntax keyword cudaConstant CU_STREAM_WRITE_VALUE_DEFAULT CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
    syntax keyword cudaType     CUstream_flags
    syntax keyword cudaConstant CU_STREAM_DEFAULT CU_STREAM_NON_BLOCKING

    " 4.2, 4.3, 4.4, 4.5, 4.7, 4.8, 4.10
    syntax keyword cudaFunction cuGetErrorName cuGetErrorString
    syntax keyword cudaFunction cuInit
    syntax keyword cudaFunction cuDriverGetVersion
    syntax keyword cudaFunction cuDeviceGet cuDeviceGetAttribute cuDeviceGetCount cuDeviceGetName cuDeviceTotalMem
    syntax keyword cudaFunction cuDevicePrimaryCtxGetState cuDevicePrimaryCtxRelease cuDevicePrimaryCtxReset cuDevicePrimaryCtxRetain cuDevicePrimaryCtxSetFlags
    syntax keyword cudaFunction cuCtxCreate cuCtxDestroy cuCtxGetApiVersion cuCtxGetCacheConfig cuCtxGetCurrent cuCtxGetDevice cuCtxGetFlags cuCtxGetLimit cuCtxGetSharedMemConfig cuCtxGetStreamPriorityRange cuCtxPopCurrent cuCtxPushCurrent cuCtxSetCacheConfig cuCtxSetCurrent cuCtxSetLimit cuCtxSetSharedMemConfig cuCtxSynchronize
    syntax keyword cudaFunction cuLinkAddData cuLinkAddFile cuLinkComplete cuLinkCreate cuLinkDestroy cuModuleGetFunction cuModuleGetGlobal cuModuleGetSurfRef cuModuleGetTexRef cuModuleLoad cuModuleLoadData cuModuleLoadDataEx cuModuleLoadFatBinary cuModuleUnload

    " 4.11
    syntax keyword cudaFunction cuArray3DCreate
    syntax keyword cudaFunction cuArray3DGetDescriptor
    syntax keyword cudaFunction cuArrayCreate
    syntax keyword cudaFunction cuArrayDestroy
    syntax keyword cudaFunction cuArrayGetDescriptor
    syntax keyword cudaFunction cuDeviceGetByPCIBusId
    syntax keyword cudaFunction cuDeviceGetPCIBusId
    syntax keyword cudaFunction cuIpcCloseMemHandle
    syntax keyword cudaFunction cuIpcGetEventHandle
    syntax keyword cudaFunction cuIpcGetMemHandle
    syntax keyword cudaFunction cuIpcOpenEventHandle
    syntax keyword cudaFunction cuIpcOpenMemHandle
    syntax keyword cudaFunction cuMemAlloc
    syntax keyword cudaFunction cuMemAllocHost
    syntax keyword cudaFunction cuMemAllocManaged
    syntax keyword cudaFunction cuMemAllocPitch
    syntax keyword cudaFunction cuMemFree
    syntax keyword cudaFunction cuMemFreeHost
    syntax keyword cudaFunction cuMemGetAddressRange
    syntax keyword cudaFunction cuMemGetInfo
    syntax keyword cudaFunction cuMemHostAlloc
    syntax keyword cudaFunction cuMemHostGetDevicePointer
    syntax keyword cudaFunction cuMemHostGetFlags
    syntax keyword cudaFunction cuMemHostRegister
    syntax keyword cudaFunction cuMemHostUnregister
    syntax keyword cudaFunction cuMemcpy
    syntax keyword cudaFunction cuMemcpy2D
    syntax keyword cudaFunction cuMemcpy2DAsync
    syntax keyword cudaFunction cuMemcpy2DUnaligned
    syntax keyword cudaFunction cuMemcpy3D
    syntax keyword cudaFunction cuMemcpy3DAsync
    syntax keyword cudaFunction cuMemcpy3DPeer
    syntax keyword cudaFunction cuMemcpy3DPeerAsync
    syntax keyword cudaFunction cuMemcpyAsync
    syntax keyword cudaFunction cuMemcpyAtoA
    syntax keyword cudaFunction cuMemcpyAtoD
    syntax keyword cudaFunction cuMemcpyAtoH
    syntax keyword cudaFunction cuMemcpyAtoHAsync
    syntax keyword cudaFunction cuMemcpyDtoA
    syntax keyword cudaFunction cuMemcpyDtoD
    syntax keyword cudaFunction cuMemcpyDtoDAsync
    syntax keyword cudaFunction cuMemcpyDtoH
    syntax keyword cudaFunction cuMemcpyDtoHAsync
    syntax keyword cudaFunction cuMemcpyHtoA
    syntax keyword cudaFunction cuMemcpyHtoAAsync
    syntax keyword cudaFunction cuMemcpyHtoD
    syntax keyword cudaFunction cuMemcpyHtoDAsync
    syntax keyword cudaFunction cuMemcpyPeer
    syntax keyword cudaFunction cuMemcpyPeerAsync
    syntax keyword cudaFunction cuMemsetD16
    syntax keyword cudaFunction cuMemsetD16Async
    syntax keyword cudaFunction cuMemsetD2D16
    syntax keyword cudaFunction cuMemsetD2D16Async
    syntax keyword cudaFunction cuMemsetD2D32
    syntax keyword cudaFunction cuMemsetD2D32Async
    syntax keyword cudaFunction cuMemsetD2D8
    syntax keyword cudaFunction cuMemsetD2D8Async
    syntax keyword cudaFunction cuMemsetD32
    syntax keyword cudaFunction cuMemsetD32Async
    syntax keyword cudaFunction cuMemsetD8
    syntax keyword cudaFunction cuMemsetD8Async
    syntax keyword cudaFunction cuMipmappedArrayCreate
    syntax keyword cudaFunction cuMipmappedArrayDestroy
    syntax keyword cudaFunction cuMipmappedArrayGetLevel

    " 4.12, 4.13, 4.14, 4.15, 4.16, 4.18
    syntax keyword cudaFunction cuMemAdvise cuMemPrefetchAsync cuMemRangeGetAttribute cuMemRangeGetAttributes cuPointerGetAttribute cuPointerGetAttributes cuPointerSetAttribute
    syntax keyword cudaFunction cuStreamAddCallback cuStreamAttachMemAsync cuStreamCreate cuStreamCreateWithPriority cuStreamDestroy cuStreamGetFlags cuStreamGetPriority cuStreamQuery cuStreamSynchronize cuStreamWaitEvent
    syntax keyword cudaFunction cuEventCreate cuEventDestroy cuEventElapsedTime cuEventQuery cuEventRecord cuEventSynchronize
    syntax keyword cudaFunction cuStreamBatchMemOp cuStreamWaitValue32 cuStreamWaitValue64 cuStreamWriteValue32 cuStreamWriteValue64
    syntax keyword cudaFunction cuFuncGetAttribute cuFuncSetAttribute cuFuncSetCacheConfig cuFuncSetSharedMemConfig cuLaunchCooperativeKernel cuLaunchCooperativeKernelMultiDevice cuLaunchKernel
    syntax keyword cudaFunction cuOccupancyMaxActiveBlocksPerMultiprocessor cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags cuOccupancyMaxPotentialBlockSize cuOccupancyMaxPotentialBlockSizeWithFlags

    " 4.19
    syntax keyword cudaFunction cuTexRefGetAddress
    syntax keyword cudaFunction cuTexRefGetAddressMode
    syntax keyword cudaFunction cuTexRefGetArray
    syntax keyword cudaFunction cuTexRefGetBorderColor
    syntax keyword cudaFunction cuTexRefGetFilterMode
    syntax keyword cudaFunction cuTexRefGetFlags
    syntax keyword cudaFunction cuTexRefGetFormat
    syntax keyword cudaFunction cuTexRefGetMaxAnisotropy
    syntax keyword cudaFunction cuTexRefGetMipmapFilterMode
    syntax keyword cudaFunction cuTexRefGetMipmapLevelBias
    syntax keyword cudaFunction cuTexRefGetMipmapLevelClamp
    syntax keyword cudaFunction cuTexRefGetMipmappedArray
    syntax keyword cudaFunction cuTexRefSetAddress
    syntax keyword cudaFunction cuTexRefSetAddress2D
    syntax keyword cudaFunction cuTexRefSetAddressMode
    syntax keyword cudaFunction cuTexRefSetArray
    syntax keyword cudaFunction cuTexRefSetBorderColor
    syntax keyword cudaFunction cuTexRefSetFilterMode
    syntax keyword cudaFunction cuTexRefSetFlags
    syntax keyword cudaFunction cuTexRefSetFormat
    syntax keyword cudaFunction cuTexRefSetMaxAnisotropy
    syntax keyword cudaFunction cuTexRefSetMipmapFilterMode
    syntax keyword cudaFunction cuTexRefSetMipmapLevelBias
    syntax keyword cudaFunction cuTexRefSetMipmapLevelClamp
    syntax keyword cudaFunction cuTexRefSetMipmappedArray

    " 4.21, 4.22, 4.23, 4.24, 4.25, 4.26
    syntax keyword cudaFunction cuSurfRefGetArray cuSurfRefSetArray
    syntax keyword cudaFunction cuTexObjectCreate cuTexObjectDestroy cuTexObjectGetResourceDesc cuTexObjectGetResourceViewDesc cuTexObjectGetTextureDesc
    syntax keyword cudaFunction cuSurfObjectCreate cuSurfObjectDestroy cuSurfObjectGetResourceDesc
    syntax keyword cudaFunction cuCtxDisablePeerAccess cuCtxEnablePeerAccess cuDeviceCanAccessPeer cuDeviceGetP2PAttribute
    syntax keyword cudaFunction cuGraphicsMapResources cuGraphicsResourceGetMappedMipmappedArray cuGraphicsResourceGetMappedPointer cuGraphicsResourceSetMapFlags cuGraphicsSubResourceGetMappedArray cuGraphicsUnmapResources cuGraphicsUnregisterResource
    syntax keyword cudaFunction cuProfilerInitialize cuProfilerStart cuProfilerStop

    " 4.27, 4.28, 4.29, 4.30, 4.31, 4.32
    syntax keyword cudaFunction cuGLGetDevices cuGraphicsGLRegisterBuffer cuGraphicsGLRegisterImage cuWGLGetDevice
    syntax keyword cudaFunction cuD3D9CtxCreate cuD3D9CtxCreateOnDevice cuD3D9GetDevice cuD3D9GetDevices cuD3D9GetDirect3DDevice cuGraphicsD3D9RegisterResource
    syntax keyword cudaFunction cuD3D10GetDevice cuD3D10GetDevices cuGraphicsD3D10RegisterResource
    syntax keyword cudaFunction cuD3D11GetDevice cuD3D11GetDevices cuGraphicsD3D11RegisterResource
    syntax keyword cudaFunction cuGraphicsVDPAURegisterOutputSurface cuGraphicsVDPAURegisterVideoSurface cuVDPAUCtxCreate cuVDPAUGetDevice
    syntax keyword cudaFunction cuEGLStreamConsumerAcquireFrame cuEGLStreamConsumerConnect cuEGLStreamConsumerConnectWithFlags cuEGLStreamConsumerDisconnect cuEGLStreamConsumerReleaseFrame cuEGLStreamProducerConnect cuEGLStreamProducerDisconnect cuEGLStreamProducerPresentFrame cuEGLStreamProducerReturnFrame cuEventCreateFromEGLSync cuGraphicsEGLRegisterImage cuGraphicsResourceGetMappedEglFrame
    syntax keyword cudaType     CUGLDeviceList
    syntax keyword cudaType     CUd3d9DeviceList
    syntax keyword cudaType     CUd3d10DeviceList
    syntax keyword cudaType     CUd3d11DeviceList
    syntax keyword cudaConstant CU_GL_DEVICE_LIST_ALL CU_GL_DEVICE_LIST_CURRENT_FRAME CU_GL_DEVICE_LIST_NEXT_FRAME
    syntax keyword cudaConstant CU_D3D9_DEVICE_LIST_ALL CU_D3D9_DEVICE_LIST_CURRENT_FRAME CU_D3D9_DEVICE_LIST_NEXT_FRAME
    syntax keyword cudaConstant CU_D3D10_DEVICE_LIST_ALL CU_D3D10_DEVICE_LIST_CURRENT_FRAME CU_D3D10_DEVICE_LIST_NEXT_FRAME
    syntax keyword cudaConstant CU_D3D11_DEVICE_LIST_ALL CU_D3D11_DEVICE_LIST_CURRENT_FRAME CU_D3D11_DEVICE_LIST_NEXT_FRAME

    " 6. Data Fields
    " Currently not used because of potential keyword collisions
    " For example, 'function' would collide with std::function
    " TODO: find a way to highlight these keywords only when they appear after
    " a dot or arrow, like x.depth or x->depth, but not on their own
    " syntax keyword cudaMember addressMode contained
    " syntax keyword cudaMember blockDimX blockDimY blockDimZ contained
    " syntax keyword cudaMember borderColor contained
    " syntax keyword cudaMember clockRate contained
    " syntax keyword cudaMember cuFormat contained
    " syntax keyword cudaMember depth contained
    " syntax keyword cudaMember Depth contained
    " syntax keyword cudaMember devPtr contained
    " syntax keyword cudaMember dstArray contained
    " syntax keyword cudaMember dstContext contained
    " syntax keyword cudaMember dstDevice contained
    " syntax keyword cudaMember dstHeight contained
    " syntax keyword cudaMember dstHost contained
    " syntax keyword cudaMember dstLOD contained
    " syntax keyword cudaMember dstMemoryType contained
    " syntax keyword cudaMember dstPitch contained
    " syntax keyword cudaMember dstXInBytes contained
    " syntax keyword cudaMember dstY dstZ contained
    " syntax keyword cudaMember eglColorFormat contained
    " syntax keyword cudaMember filterMode contained
    " syntax keyword cudaMember firstLayer contained
    " syntax keyword cudaMember firstMipmapLevel contained
    " syntax keyword cudaMember flags contained
    " syntax keyword cudaMember Flags contained
    " syntax keyword cudaMember Format contained
    " syntax keyword cudaMember format contained
    " syntax keyword cudaMember frameType contained
    " syntax keyword cudaMember function contained
    " syntax keyword cudaMember gridDimX gridDimY gridDimZ contained
    " syntax keyword cudaMember hArray contained
    " syntax keyword cudaMember Height contained
    " syntax keyword cudaMember height contained
    " syntax keyword cudaMember hMipmappedArray contained
    " syntax keyword cudaMember hStream contained
    " syntax keyword cudaMember kernelParams contained
    " syntax keyword cudaMember lastLayer contained
    " syntax keyword cudaMember lastMipmapLevel contained
    " syntax keyword cudaMember maxAnisotropy contained
    " syntax keyword cudaMember maxGridSize contained
    " syntax keyword cudaMember maxMipmapLevelClamp contained
    " syntax keyword cudaMember maxThreadsDim contained
    " syntax keyword cudaMember maxThreadsPerBlock contained
    " syntax keyword cudaMember memPitch contained
    " syntax keyword cudaMember minMipmapLevelClamp contained
    " syntax keyword cudaMember mipmapFilterMode contained
    " syntax keyword cudaMember mipmapLevelBias contained
    " syntax keyword cudaMember numChannels contained
    " syntax keyword cudaMember NumChannels contained
    " syntax keyword cudaMember pArray contained
    " syntax keyword cudaMember pitch contained
    " syntax keyword cudaMember pitchInBytes contained
    " syntax keyword cudaMember planeCount contained
    " syntax keyword cudaMember pPitch contained
    " syntax keyword cudaMember regsPerBlock contained
    " syntax keyword cudaMember reserved0 reserved1 contained
    " syntax keyword cudaMember resType contained
    " syntax keyword cudaMember sharedMemBytes contained
    " syntax keyword cudaMember sharedMemPerBlock contained
    " syntax keyword cudaMember SIMDWidth contained
    " syntax keyword cudaMember sizeInBytes contained
    " syntax keyword cudaMember srcArray contained
    " syntax keyword cudaMember srcContext contained
    " syntax keyword cudaMember srcDevice contained
    " syntax keyword cudaMember srcHeight contained
    " syntax keyword cudaMember srcHost contained
    " syntax keyword cudaMember srcLOD contained
    " syntax keyword cudaMember srcMemoryType contained
    " syntax keyword cudaMember srcPitch contained
    " syntax keyword cudaMember srcXInBytes contained
    " syntax keyword cudaMember srcY srcZ contained
    " syntax keyword cudaMember textureAlign contained
    " syntax keyword cudaMember totalConstantMemory contained
    " syntax keyword cudaMember Width contained
    " syntax keyword cudaMember width contained
    " syntax keyword cudaMember WidthInBytes contained
endif " g:cuda_driver_api_highlight


" CUDA Thrust library {{{1
" Based on: http://docs.nvidia.com/cuda/thrust/index.html (v9.1.85, Jan 24, 2018)

if exists('g:cuda_thrust_highlight') && g:cuda_thrust_highlight
    syntax keyword cudaThrustNamespace  thrust
    syntax keyword cudaThrustType       device_vector host_vector
    syntax keyword cudaThrustType       constant_iterator counting_iterator transform_iterator permutation_iterator zip_iterator
    syntax keyword cudaThrustType       execution_policy host_execution_policy device_execution_policy
    syntax keyword cudaThrustType       device_allocator device_reference device_ptr device_malloc_reference
    hi default link cudaThrustNamespace Constant
    hi default link cudaThrustType      Type
endif


" CUDA Math API {{{1
" Based on: http://docs.nvidia.com/cuda/cuda-math-api (v9.1.85, Jan 24, 2018)

" syntax keyword cudaFunction __h2div __hadd __hadd_sat __hdiv __hfma __hfma_sat __hmul __hmul_sat __hneg __hsub __hsub_sat
" syntax keyword cudaFunction __hadd2 __hadd2_sat __hfma2 __hfma2_sat __hmul2 __hmul2_sat __hneg2 __hsub2 __hsub2_sat
" syntax keyword cudaFunction __heq __hequ __hge __hgeu __hgt __hgtu __hisinf __hisnan __hle __hleu __hlt __hltu __hne __hneu
" syntax keyword cudaFunction __hbeq2 __hbequ2 __hbge2 __hbgeu2 __hbgt2 __hbgtu2 __hble2 __hbleu2 __hblt2 __hbltu2 __hbne2 __hbneu2 __heq2 __hequ2 __hge2 __hgeu2 __hgt2 __hgtu2 __hisnan2 __hle2 __hleu2 __hlt2 __hltu2 __hne2 __hneu2
" syntax keyword cudaFunction __float22half2_rn __float2half __float2half2_rn __float2half_rd __float2half_rn __float2half_ru __float2half_rz __floats2half2_rn __half22float2 __half2float __half2half2 __half2int_rd __half2int_rn __half2int_ru __half2int_rz __half2ll_rd __half2ll_rn __half2ll_ru __half2ll_rz __half2short_rd __half2short_rn __half2short_ru __half2short_rz __half2uint_rd __half2uint_rn
" syntax keyword cudaFunction __half2uint_ru __half2uint_rz __half2ull_rd __half2ull_rn __half2ull_ru __half2ull_rz __half2ushort_rd
" syntax keyword cudaFunction __half2ushort_rn __half2ushort_ru __half2ushort_rz __half_as_short __half_as_ushort __halves2half2 __high2float __high2half __high2half2 __highs2half2 __int2half_rd __int2half_rn __int2half_ru __int2half_rz __ll2half_rd __ll2half_rn __ll2half_ru __ll2half_rz __low2float __low2half __low2half2 __lowhigh2highlow __lows2half2 __short2half_rd __short2half_rn __short2half_ru __short2half_rz __short_as_half __uint2half_rd
" syntax keyword cudaFunction __uint2half_rn __uint2half_ru __uint2half_rz __ull2half_rd __ull2half_rn __ull2half_ru __ull2half_rz __ushort2half_rd __ushort2half_rn __ushort2half_ru __ushort2half_rz __ushort_as_half
" syntax keyword cudaFunction hceil hcos hexp hexp10 hexp2 hfloor hlog hlog10 hlog2 hrcp hrint hrsqrt hsin hsqrt htrunc
" syntax keyword cudaFunction h2ceil h2cos h2exp h2exp10 h2exp2 h2floor h2log h2log10 h2log2 h2rcp h2rint h2rsqrt h2sin h2sqrt h2trunc

" syntax keyword cudaFunction acosf acoshf asinf asinhf atan2f atanf atanhf cbrtf ceilf copysignf cosf coshf cospif
" syntax keyword cudaFunction cyl_bessel_i0f cyl_bessel_i1f erfcf erfcinvf erfcxf erff erfinvf exp10f exp2f expf expm1f
" syntax keyword cudaFunction fabsf fdimf fdividef floorf fmaf fmaxf fminf fmodf frexpf hypotf ilogbf isfinite isinf isnan
" syntax keyword cudaFunction j0f j1f jnf ldexpf lgammaf llrintf llroundf log10f log1pf log2f logbf logf lrintf lroundf
" syntax keyword cudaFunction modff nanf nearbyintf nextafterf norm3df norm4df normcdff normcdfinvf normf powf
" syntax keyword cudaFunction rcbrtf remainderf remquof rhypotf rintf rnorm3df rnorm4df rnormf roundf rsqrtf
" syntax keyword cudaFunction scalblnf scalbnf signbit sincosf sincospif sinf sinhf sinpif sqrtf tanf tanhf tgammaf truncf y0f y1f ynf
" syntax keyword cudaFunction acos acosh asin asinh atan atan2 atanh cbrt ceil copysign cos cosh cospi
" syntax keyword cudaFunction cyl_bessel_i0 cyl_bessel_i1 erf erfc erfcinv erfcx erfinv exp exp10 exp2 expm1
" syntax keyword cudaFunction fabs fdim floor fma fmax fmin fmod frexp hypot ilogb isfinite isinf isnan
" syntax keyword cudaFunction j0 j1 jn ldexp lgamma llrint llround log log10 log1p log2 logb lrint lround
" syntax keyword cudaFunction modf nan nearbyint nextafter norm norm3d norm4d normcdf normcdfinv pow
" syntax keyword cudaFunction rcbrt remainder remquo rhypot rint rnorm rnorm3d rnorm4d round rsqrt
" syntax keyword cudaFunction scalbln scalbn signbit sin sincos sincospi sinh sinpi sqrt tan tanh tgamma trunc y0 y1 yn

" syntax keyword cudaFunction __cosf __exp10f __expf __fadd_rd __fadd_rn __fadd_ru __fadd_rz __fdiv_rd __fdiv_rn __fdiv_ru __fdiv_rz __fdividef __fmaf_rd __fmaf_rn __fmaf_ru __fmaf_rz __fmul_rd __fmul_rn __fmul_ru __fmul_rz __frcp_rd __frcp_rn __frcp_ru __frcp_rz __frsqrt_rn __fsqrt_rd __fsqrt_rn __fsqrt_ru __fsqrt_rz __fsub_rd __fsub_rn __fsub_ru __fsub_rz __log10f __log2f __logf __powf __saturatef __sincosf __sinf __tanf
" syntax keyword cudaFunction __dadd_rd __dadd_rn __dadd_ru __dadd_rz __ddiv_rd __ddiv_rn __ddiv_ru __ddiv_rz __dmul_rd __dmul_rn __dmul_ru __dmul_rz __drcp_rd __drcp_rn __drcp_ru __drcp_rz __dsqrt_rd __dsqrt_rn __dsqrt_ru __dsqrt_rz __dsub_rd __dsub_rn __dsub_ru __dsub_rz __fma_rd __fma_rn __fma_ru __fma_rz
" syntax keyword cudaFunction __brev __brevll __byte_perm __clz __clzll __ffs __ffsll __hadd __mul24 __mul64hi __mulhi __popc __popcll __rhadd __sad __uhadd __umul24 __umul64hi __umulhi __urhadd __usad
" syntax keyword cudaFunction __double2float_rd __double2float_rn __double2float_ru __double2float_rz __double2hiint __double2int_rd __double2int_rn __double2int_ru __double2int_rz __double2ll_rd __double2ll_rn __double2ll_ru __double2ll_rz __double2loint __double2uint_rd __double2uint_rn __double2uint_ru __double2uint_rz __double2ull_rd __double2ull_rn __double2ull_ru __double2ull_rz __double_as_longlong
" syntax keyword cudaFunction __float2int_rd __float2int_rn __float2int_ru __float2int_rz __float2ll_rd __float2ll_rn __float2ll_ru __float2ll_rz __float2uint_rd __float2uint_rn __float2uint_ru __float2uint_rz __float2ull_rd __float2ull_rn __float2ull_ru __float2ull_rz __float_as_int __float_as_uint __hiloint2double __int2double_rn __int2float_rd __int2float_rn __int2float_ru __int2float_rz __int_as_float
" syntax keyword cudaFunction __ll2double_rd __ll2double_rn __ll2double_ru __ll2double_rz __ll2float_rd __ll2float_rn __ll2float_ru __ll2float_rz __longlong_as_double __uint2double_rn __uint2float_rd __uint2float_rn __uint2float_ru __uint2float_rz __uint_as_float __ull2double_rd __ull2double_rn __ull2double_ru __ull2double_rz __ull2float_rd __ull2float_rn __ull2float_ru __ull2float_rz
" syntax keyword cudaFunction __vabs2 __vabs4 __vabsdiffs2 __vabsdiffs4 __vabsdiffu2 __vabsdiffu4 __vabsss2 __vabsss4 __vadd2 __vadd4 __vaddss2 __vaddss4 __vaddus2 __vaddus4 __vavgs2 __vavgs4 __vavgu2 __vavgu4 __vcmpeq2 __vcmpeq4 __vcmpges2 __vcmpges4 __vcmpgeu2 __vcmpgeu4 __vcmpgts2 __vcmpgts4 __vcmpgtu2 __vcmpgtu4 __vcmples2 __vcmples4 __vcmpleu2 __vcmpleu4 __vcmplts2 __vcmplts4 __vcmpltu2 __vcmpltu4 __vcmpne2 __vcmpne4 __vhaddu2
" syntax keyword cudaFunction __vhaddu4 __vmaxs2 __vmaxs4 __vmaxu2 __vmaxu4 __vmins2 __vmins4 __vminu2 __vminu4 __vneg2 __vneg4 __vnegss2 __vnegss4 __vsads2 __vsads4 __vsadu2 __vsadu4 __vseteq2 __vseteq4 __vsetges2 __vsetges4 __vsetgeu2 __vsetgeu4 __vsetgts2 __vsetgts4 __vsetgtu2 __vsetgtu4 __vsetles2 __vsetles4 __vsetleu2 __vsetleu4 __vsetlts2 __vsetlts4 __vsetltu2 __vsetltu4 __vsetne2 __vsetne4 __vsub2 __vsub4 __vsubss2 __vsubss4 __vsubus2 __vsubus4


" Deprecated CUDA items {{{1

" Deprecated as of CUDA 9.0
" syntax keyword cudaFunction __any __all __ballot
" syntax keyword cudaFunction __shfl __shfl_up __shfl_down __shfl_xor

" 4. Modules -- 4.2. Thread Management [DEPRECATED]
" syntax keyword cudaFunction cudaThreadExit
" syntax keyword cudaFunction cudaThreadGetCacheConfig
" syntax keyword cudaFunction cudaThreadGetLimit
" syntax keyword cudaFunction cudaThreadSetCacheConfig
" syntax keyword cudaFunction cudaThreadSetLimit
" syntax keyword cudaFunction cudaThreadSynchronize

" Deprecated as of CUDA 7.5
" syntax keyword cudaFunction cudaSetDoubleForDevice cudaSetDoubleForHost

" 4. Modules -- 4.8. Execution Control [DEPRECATED]
" syntax keyword cudaFunction cudaConfigureCall
" syntax keyword cudaFunction cudaLaunch
" syntax keyword cudaFunction cudaSetupArgument

" 4. Modules -- 4.13. OpenGL Interoperability [DEPRECATED]
" syntax keyword cudaType     cudaGLMapFlags
" syntax keyword cudaConstant cudaGLMapFlagsNone cudaGLMapFlagsReadOnly cudaGLMapFlagsWriteDiscard
" syntax keyword cudaFunction cudaGLMapBufferObject
" syntax keyword cudaFunction cudaGLMapBufferObjectAsync
" syntax keyword cudaFunction cudaGLRegisterBufferObject
" syntax keyword cudaFunction cudaGLSetBufferObjectMapFlags
" syntax keyword cudaFunction cudaGLSetGLDevice
" syntax keyword cudaFunction cudaGLUnmapBufferObject
" syntax keyword cudaFunction cudaGLUnmapBufferObjectAsync
" syntax keyword cudaFunction cudaGLUnregisterBufferObject

" 4. Modules -- 4.15. Direct3D 9 Interoperability [DEPRECATED]
" syntax keyword cudaType     cudaD3D9MapFlags
" syntax keyword cudaConstant cudaD3D9MapFlagsNone cudaD3D9MapFlagsReadOnly cudaD3D9MapFlagsWriteDiscard
" syntax keyword cudaConstant cudaD3D9RegisterFlagsNone cudaD3D9RegisterFlagsArray
" syntax keyword cudaType     cudaD3D9RegisterFlags
" syntax keyword cudaFunction cudaD3D9MapResources
" syntax keyword cudaFunction cudaD3D9RegisterResource
" syntax keyword cudaFunction cudaD3D9ResourceGetMappedArray
" syntax keyword cudaFunction cudaD3D9ResourceGetMappedPitch
" syntax keyword cudaFunction cudaD3D9ResourceGetMappedPointer
" syntax keyword cudaFunction cudaD3D9ResourceGetMappedSize
" syntax keyword cudaFunction cudaD3D9ResourceGetSurfaceDimensions
" syntax keyword cudaFunction cudaD3D9ResourceSetMapFlags
" syntax keyword cudaFunction cudaD3D9UnmapResources
" syntax keyword cudaFunction cudaD3D9UnregisterResource

" 4. Modules -- 4.17. Direct3D 10 Interoperability [DEPRECATED]
" syntax keyword cudaType     cudaD3D10MapFlags
" syntax keyword cudaConstant cudaD3D10MapFlagsNone cudaD3D10MapFlagsReadOnly cudaD3D10MapFlagsWriteDiscard
" syntax keyword cudaType     cudaD3D10RegisterFlags
" syntax keyword cudaType     cudaD3D10RegisterFlagsNone cudaD3D10RegisterFlagsArray
" syntax keyword cudaFunction cudaD3D10GetDirect3DDevice
" syntax keyword cudaFunction cudaD3D10MapResources
" syntax keyword cudaFunction cudaD3D10RegisterResource
" syntax keyword cudaFunction cudaD3D10ResourceGetMappedArray
" syntax keyword cudaFunction cudaD3D10ResourceGetMappedPitch
" syntax keyword cudaFunction cudaD3D10ResourceGetMappedPointer
" syntax keyword cudaFunction cudaD3D10ResourceGetMappedSize
" syntax keyword cudaFunction cudaD3D10ResourceGetSurfaceDimensions
" syntax keyword cudaFunction cudaD3D10ResourceSetMapFlags
" syntax keyword cudaFunction cudaD3D10SetDirect3DDevice
" syntax keyword cudaFunction cudaD3D10UnmapResources
" syntax keyword cudaFunction cudaD3D10UnregisterResource

" 4. Modules -- 4.19. Direct3D 11 Interoperability [DEPRECATED]
" syntax keyword cudaFunction cudaD3D11GetDirect3DDevice
" syntax keyword cudaFunction cudaD3D11SetDirect3DDevice

" Deprecated cudaError flags:
" syntax keyword cudaConstant cudaErrorPriorLaunchFailure cudaErrorAddressOfConstant cudaErrorTextureFetchFailed cudaErrorTextureNotBound cudaErrorSynchronizationError cudaErrorMixedDeviceExecution cudaErrorMemoryValueTooLarge
" syntax keyword cudaConstant cudaErrorCudartUnloading cudaErrorApiFailureBase cudaErrorNotYetImplemented
" syntax keyword cudaConstant cudaErrorProfilerNotInitialized cudaErrorProfilerAlreadyStarted cudaErrorProfilerAlreadyStopped


" Default highlighting {{{1

" Link syntax groups to common highlight groups
hi default link cudaStorageClass StorageClass
hi default link cudaType         Type
hi default link cudaFunction     Function
hi default link cudaConstant     Constant
hi default link cudaVariable     Identifier
hi default link cudaNamespace    Constant

" }}}

let b:current_syntax = 'cuda'
