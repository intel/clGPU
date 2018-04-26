# OpenCL convolution primitives for Intel&reg; GPU platforms

Repository contains optimized 1x1 convolution primitives implemented in OpenCL&trade; for Intel&reg; GPU platforms.

-----------------------------------------------------------------
## OpenCL KERNEL CONFIGURATIONS
-----------------------------------------------------------------

1x1 Convolution is performed as a GEMM ( matrix-multiply ) operation.

Assume a 1x1 convolution layer has parameters _Batch_size, H, W, P, Q, Ci, Ko_

- H,W - Spatial dimensions of input activations
- P,Q - Spatial dimensions of output
- Ci  - Number of input channels
- Ko  - Number of output channels

Activation, Weights and Output are packed into matrix format as given below.

Activation Matrix (MxK), Weight Matrix (KxN), Output Matrix (MxN)

- M = Batch_size * H * W
- K = Ci
- N = Ko

The OpenCL kernels have optimizations specific to 1x1 convolution dimensions of popular CNNs like resnet50.
### Optimizations have been summarized below

- Different tile-sizes: In parallel programming languages such as OpenCL&trade;/CUDA&reg;, gemm operation is
     performed in a tiled fashion where each work-item/thread computes a specific output-tile.
     The tile-size determines work distribution, caching etc.
     In this repository tile-sizes specific to 1x1 convolution dimension of resnet-50 are chosen
     so that it fits the dimensions well and optimizes machine occupancy,bandwidth. Some of tile-sizes used are 16x2,7x2,14x2 etc.

- K-Slicing: Tiling methodology helps to split work among M,N dimensions. Each work-item/thread has to process
     data along K-dimension. When K-dimension is huge compared to M,N , this tiling creates few threads which have to
     sequentially process data along K-dimension. "K-Slicing" method is used to launch multiple threads along K-dimension
     to process in parallel and partial sum is reduced through Shared Local Memory.

-----------------------------------------------------------------
## OpenCL KERNELS
-----------------------------------------------------------------
 Kernels for FP16 and FP32 precision have been provided.

- FP32 kernel names start with buf_
- FP16 kernel names start with h_ ( to denote half-float ).
- gemm_common.cl - header containing common defines

 Host-side configuration for kernels - build parameters, OpenCL&trade; grid size, kernel arguments have been provided in comment section ( top ) of each kernel file.

 This release contains kernels for resnet50 1x1 convolution layers. Kernels for other networks will be released in future.

-----------------------------------------------------------------
## INSTALLATION
-----------------------------------------------------------------

- HW Platforms: Intel&reg; Skylake, Kabylake
- OS          : Windows&reg;, Linux

Software Required

- Latest Intel&reg; OpenCL&trade; SDK : https://software.intel.com/en-us/intel-opencl/download

Host-code is required to select OpenCL&trade; platform for execution (Intel&reg; GPU), build the kernel with required build parameters and
invoke the kernel with appropriate grid dimensions and kernel arguments.

Host-code can be written in C++ or Python ( using pyOpenCL ).
C++ host-code example is given in SGEMM for Intel Graphics Blog:  https://software.intel.com/en-us/articles/sgemm-for-intel-processor-graphics

Contributors: Andrew Lavin, Sabareesh Ganapathy, Pradeep Ramani, Varghese George.
