# Intel&reg; clGPU Changelog

## Drop 0

- Version 0.1.1 Alpha

## Drop 1.0

- Version 0.1.2 Alpha
- clGPU
  - Added support for _#include_ statements in OpenCL kernels.
- clBLAS
  - Fixed/optimized functions:
    - Complex:
      - Cgerc, Cgeru, Cher, Chpr, Chpr2, Crot, Icamax, Icamin
    - Float:
      - Isamax, Isamin, Sgbmv, Sgemm, Sgemv, Sger, Srotm, Sscal, Ssymv, Ssyr, Ssyr2, Strsv
  - API: Define clBLAS complex type as ```std::complex<float>``` for C++.
- Improved CMAKE build scripts.


Copyright &copy; 2018, Intel&reg; Corporation
