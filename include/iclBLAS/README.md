# Intel&reg; Compute Libraries for GPU BLAS library (iclBLAS)

## Introduction
*Intel&reg; Compute Libraries BLAS* (*iclBLAS*) is an open source implementation of Basic Linear Algebra Subprograms (BLAS) functions.

*iclBLAS* is intended to accelerate mathematical operations using Intel&reg; Processor Graphics - including HD Graphics and Iris&reg; Graphics.

It includes optimized kernels for mathematical operations based on BLAS Library written with C and C++ interfaces.

## Limitations
At this time *iclBLAS* library supports only positive values for vector strides (i.e. **incx**, **incy**, ..).

*iclBLAS* supports Intel&reg; HD Graphics and Intel&reg; Iris&reg; Graphics and is optimized for:
- Codename *Skylake*:
    * Intel&reg; HD Graphics 510 (GT1, *client* market)
    * Intel&reg; HD Graphics 515 (GT2, *client* market)
    * Intel&reg; HD Graphics 520 (GT2, *client* market)
    * Intel&reg; HD Graphics 530 (GT2, *client* market)
    * Intel&reg; Iris&reg; Graphics 540 (GT3e, *client* market)
    * Intel&reg; Iris&reg; Graphics 550 (GT3e, *client* market)
    * Intel&reg; Iris&reg; Pro Graphics 580 (GT4e, *client* market)
    * Intel&reg; HD Graphics P530 (GT2, *server* market)
    * Intel&reg; Iris&reg; Pro Graphics P555 (GT3e, *server* market)
    * Intel&reg; Iris&reg; Pro Graphics P580 (GT4e, *server* market)
- Codename *Apollolake*:
    * Intel&reg; HD Graphics 500
    * Intel&reg; HD Graphics 505
- Codename *Kabylake*:
    * Intel&reg; HD Graphics 610 (GT1, *client* market)
    * Intel&reg; HD Graphics 615 (GT2, *client* market)
    * Intel&reg; HD Graphics 620 (GT2, *client* market)
    * Intel&reg; HD Graphics 630 (GT2, *client* market)
    * Intel&reg; Iris&reg; Graphics 640 (GT3e, *client* market)
    * Intel&reg; Iris&reg; Graphics 650 (GT3e, *client* market)
    * Intel&reg; HD Graphics P630 (GT2, *server* market)
    * Intel&reg; Iris&reg; Pro Graphics 630 (GT2, *server* market)

clGPU currently uses OpenCL&trade; with multiple Intel&reg; OpenCL&trade; extensions and requires Intel&reg; Graphics Driver to run.

The definition of *iclBLAS* may differ from BLAS interface in some cases, so be informed to check our implementation definition first.

## Usage
To use the *iclBLAS* Library, user has to allocate required by functions parameters in host memory space (The Library will automatically copy data to GPU memory space if needed).
Next step is to fill allocated buffers with data and call the function from API. When computation ends result will be automatically copied back to host memory and stored as function definition describes.

---

Copyright &copy; 2018, Intel&reg; Corporation
