DISCONTINUATION OF PROJECT.

This project will no longer be maintained by Intel.

Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project. 

Intel no longer accepts patches to this project.

If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project. 
# Intel&reg; Compute Libraries for GPU

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![v0.1.1](https://img.shields.io/badge/0.1.1-alpha-green.svg)

## Introduction

*Intel&reg; Compute Libraries for GPU* (**Intel&reg; clGPU**) is a framework and repository for implementation of compute libraries (e.g. BLAS) accelerated using Intel&reg; Processor Graphics.
It provides ability to develop efficient and optimized kernels for Intel GPU and mechanism of their execution.

This technical preview includes [**Intel&reg; clBLAS**](include/iclBLAS) - Basic Linear Algebra Subprograms (BLAS) implementation optimized for Intel&reg; GPU.

As with any technical preview, APIs may change in future updates.

## License

Intel&reg; clGPU is licensed under
[Apache License Version 2.0](LICENSE).

### Attached licenses

Intel&reg; clGPU uses 3rd-party components licensed under following licenses:

- *googletest* under [Google\* License](https://github.com/google/googletest/blob/master/googletest/LICENSE)
- *OpenCL&trade; ICD and C++ Wrapper* under [Khronos&trade; License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt)

## Project structure

- [include/iclgpu](include/iclgpu) - core library API
- [include/iclBLAS](include/iclBLAS) - Intel&reg; clBLAS API
- [core](core) - the core library implementation
- [libs/blas](libs/blas) - Intel&reg; clBLAS API implementation

## Documentation

The online Intel&reg; clGPU documentation is at [GitHub pages](https://intel.github.io/clGPU)

Module specific documentation:

- [Intel&reg; clGPU Core Library](https://intel.github.io/clGPU/docs/iclgpu/html)
- [Intel&reg; clBLAS API](https://intel.github.io/clGPU/docs/iclBLAS/html)

> There is also inline documentation available that can be [generated with Doxygen](#generating-documentation).

## Support

Please report issues and suggestions via
[GitHub issues](../../issues).

## How to Contribute

We welcome community contributions to Intel&reg; clGPU. If you have an idea how to improve the library:

- Share your proposal via
 [GitHub issues](../../issues)
- Ensure you can build the product and run all the examples with your patch
- In the case of a larger feature, create a test
- Submit a [pull request](../../pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged into our internal and GitHub repositories.

## System Requirements

Intel&reg; clGPU supports Intel&reg; HD Graphics and Intel&reg; Iris&reg; Graphics and is optimized for

- Codename *Skylake*:
  - Intel&reg; HD Graphics 510 (GT1, *client* market)
  - Intel&reg; HD Graphics 515 (GT2, *client* market)
  - Intel&reg; HD Graphics 520 (GT2, *client* market)
  - Intel&reg; HD Graphics 530 (GT2, *client* market)
  - Intel&reg; Iris&reg; Graphics 540 (GT3e, *client* market)
  - Intel&reg; Iris&reg; Graphics 550 (GT3e, *client* market)
  - Intel&reg; Iris&reg; Pro Graphics 580 (GT4e, *client* market)
  - Intel&reg; HD Graphics P530 (GT2, *server* market)
  - Intel&reg; Iris&reg; Pro Graphics P555 (GT3e, *server* market)
  - Intel&reg; Iris&reg; Pro Graphics P580 (GT4e, *server* market)
- Codename *Apollolake*:
  - Intel&reg; HD Graphics 500
  - Intel&reg; HD Graphics 505
- Codename *Kabylake*:
  - Intel&reg; HD Graphics 610 (GT1, *client* market)
  - Intel&reg; HD Graphics 615 (GT2, *client* market)
  - Intel&reg; HD Graphics 620 (GT2, *client* market)
  - Intel&reg; HD Graphics 630 (GT2, *client* market)
  - Intel&reg; Iris&reg; Graphics 640 (GT3e, *client* market)
  - Intel&reg; Iris&reg; Graphics 650 (GT3e, *client* market)
  - Intel&reg; HD Graphics P630 (GT2, *server* market)
  - Intel&reg; Iris&reg; Pro Graphics 630 (GT2, *server* market)

Intel&reg; clGPU currently uses OpenCL&trade; with multiple Intel&reg; OpenCL&trade; extensions and requires Intel&reg; Graphics Driver to run.

Intel&reg; clGPU requires CPU with Intel&reg; SSE/Intel&reg; AVX support.

---

The software dependencies are:

- [CMake\*](https://cmake.org/download/) 3.5 or later
- C++ compiler with partiall or full C++14 standard support compatible with:
  - GNU\* Compiler Collection 5.2 or later
  - Visual C++ 2015 (MSVC++ 19.0) or later
- [python&trade;](https://www.python.org/downloads/) 2.7 or later (scripts are both compatible with python&trade; 2.7.x and python&trade; 3.x)
- *(optional)* [Doxygen\*](http://www.stack.nl/~dimitri/doxygen/download.html) 1.8.13 or later
    Needed for manual generation of documentation from inline comments.

---

- The software was validated on:
  - CentOS* 7.2 with GNU* Compiler Collection 5.2 (64-bit only), using Intel&reg; Graphics Compute Runtime for OpenCL&trade; [Linux driver package](https://github.com/intel/compute-runtime) .
  - Windows&reg; 10 and Windows&reg; Server 2012 R2 with MSVC 14.0, using [Intel&reg; Graphics Driver for Windows&reg; [15.60] driver package](https://downloadcenter.intel.com/download/27412).

    More information on Intel&reg; OpenCL&trade; drivers can be found [here](https://software.intel.com/en-us/articles/opencl-drivers).

We recommend to use these drivers.

## Building

Download [Intel&reg; clGPU source code](https://github.com/intel/clGPU/archive/master.zip)
or clone the repository to your system:

```shellscript
    git clone  https://github.com/intel/clGPU.git
```

Satisfy all software dependencies and ensure that the versions are correct before building.

Intel&reg; clGPU uses multiple 3<sup>rd</sup>-party components. They are stored in binary form in `common` subdirectory. Currently they are prepared for MSVC++ and GCC\*. They will be cloned with repository.

---

Intel&reg; clGPU uses a CMake-based build system. You can use CMake command-line tool or CMake GUI (`cmake-gui`) to generate required solution.
For Windows system, you can call in `cmd` (or `powershell`):

```shellscript
    @REM Generate 64-bit solution (solution contains multiple build configurations)...
    cmake -E make_directory build && cd build && cmake -G "Visual Studio 14 2015 Win64" ..
```

Created solution can be opened in Visual Studio 2015 or built using appropriate `msbuild` tool
(you can also use `cmake --build .` to select build tool automatically).

For Unix and Linux systems:

```shellscript
    @REM Create GNU makefile for release and build it...
    cmake -E make_directory build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make
    @REM Create Ninja makefile for debug and build it...
    cmake -E make_directory build && cd build && cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug .. && ninja -k 20
```

CMake solution offers multiple options which you can specify using normal CMake syntax (`-D<option-name>=<value>`):

| CMake option                              | Type     | Description                                                                  |
|:------------------------------------------|:---------|:-----------------------------------------------------------------------------|
| CMAKE\_BUILD\_TYPE                        | STRING   | Build configuration that will be used by generated makefiles (it does not affect multi-configuration generators like generators for Visual Studio solutions). Currently supported: `Debug` (default), `Release` |
| ICLGPU\_\_ARCHITECTURE\_TARGET            | STRING   | Architecture of target system (where binary output will be deployed). CMake will try to detect it automatically (based on selected generator type, host OS and compiler properties). Specify this option only if CMake has problem with detection. Currently supported: `Windows32`, `Windows64`, `Linux64` |
| ICLGPU\_\_OUTPUT\_DIR                     | PATH     | Location where built artifacts will be written to. It is set automatically to roughly `build/out/<arch-target>/<build-type>` subdirectory. |

### Generating documentation

Documentation is provided inline and can be generated in HTML format with Doxygen. We recommend to use latest
[Doxygen\*](http://www.stack.nl/~dimitri/doxygen/download.html) and [GraphViz\*](http://www.graphviz.org/download/).

Documentation templates and configuration files are stored in `docs` subdirectories.
You can generate documentation for the core library and Intel&reg; clBLAS separately.

---

\* Other names and brands may be claimed as the property of others.

Copyright &copy; 2018, Intel&reg; Corporation
