// Copyright (c) 2017-2018 Intel Corporation
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//      http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iclBLAS.h"
#include "functions/copy_interleave.hpp"
#include "functions/Sscal.hpp"
#include "functions/Saxpy.hpp"
#include "functions/Snrm2.hpp"
#include "functions/Srotmg.hpp"
#include "functions/Isamax.hpp"
#include "functions/Isamin.hpp"
#include "functions/swap_interleave.hpp"
#include "functions/Srot.hpp"
#include "functions/Sasum.hpp"
#include "functions/Srotm.hpp"
#include "functions/Sdot.hpp"
#include "functions/Srotg.hpp"
#include "functions/Strsv.hpp"
#include "functions/Stbsv.hpp"
#include "functions/Stpsv.hpp"
#include "functions/Ssyr2.hpp"
#include "functions/Sger.hpp"
#include "functions/Ssyr.hpp"
#include "functions/Strmv.hpp"
#include "functions/Stbmv.hpp"
#include "functions/Sgbmv.hpp"
#include "functions/Stpmv.hpp"
#include "functions/Ssbmv.hpp"
#include "functions/Sspmv.hpp"
#include "functions/Ssymm.hpp"
#include "functions/Ssyrk.hpp"
#include "functions/Ssyr2k.hpp"
#include "functions/Sspr2.hpp"
#include "functions/Sspr.hpp"
#include "functions/Ssymv.hpp"
#include "functions/Sgemv.hpp"
#include "functions/Sgemm.hpp"
#include "functions/Strsm.hpp"
#include "functions/Strmm.hpp"

#include "iclBLASImpl.hpp"

// implement C API iclblasScopy by delegation to internal C++ implementation wrapped by exception_to_iclblas_status()
extern "C"
iclblasStatus_t iclblasScopy(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::copy_interleave::params params = { x, y, n, sizeof(float), incx, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::copy_interleave>(handle, params);
    });
}

// implement C API iclblasSscal using lambda
extern "C"
iclblasStatus_t iclblasSscal(iclblasHandle_t handle, int n, const float* alpha, float *x, int incx)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sscal::params params = { n, *alpha, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sscal>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSaxpy(iclblasHandle_t handle, int n, const float* alpha, float * x, int incx, float * y, int incy)
{
    if (n <= 0 || *alpha == 0.f)
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Saxpy::params params = { n, *alpha, x, incx, y, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Saxpy>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSnrm2(iclblasHandle_t handle, int n, float * x, int incx, float * result)
{
    if (n <= 0 || incx <= 0) {
        result[0] = 0.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Snrm2::params params = { n, x, incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Snrm2>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSrotmg(iclblasHandle_t handle, float * d1, float * d2, float * x1, const float * y1, float* par)
{
    // Tests for not standard parameters in the same order as in LAPACK reference implementation
    if (d1[0] < 0.f) {
        par[0] = -1.f;
        par[1] = 0.f;
        par[2] = 0.f;
        par[3] = 0.f;
        par[4] = 0.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    if (d2[0] == 0.f || y1[0] == 0.f) {
        par[0] = -2.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    // Case d1 >= 0 && fabs(d2*y1^2) >= fabs(d1*x1^2) && d2 < 0
    // reduces to  -d2*y1^2 >= d1*x1^2
    if (-d2[0] * y1[0] * y1[0] >= d1[0] * x1[0] * x1[0]) {
        par[0] = -1.f;
        par[1] = 0.f;
        par[2] = 0.f;
        par[3] = 0.f;
        par[4] = 0.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Srotmg::params params = { d1, d2, x1, *y1, par };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Srotmg>(handle, params);
    });
}

// implement C API iclblasIsamax
extern "C"
iclblasStatus_t iclblasIsamax(iclblasHandle_t handle, int n, float* x, int incx, int* result)
{
    if (n <= 0 || incx <= 0)
    {
        *result = 0;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Isamax::params params = { n, x, incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Isamax>(handle, params);
    });        
}

extern "C"
iclblasStatus_t iclblasIsamin(iclblasHandle_t handle, int n, float* x, int incx, int* result)
{
    if (n <= 0 || incx <= 0)
    {
        *result = 0;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Isamin::params params = { n, x, incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Isamin>(handle, params);
    });
}


extern "C"
iclblasStatus_t iclblasSswap(iclblasHandle_t handle, int n, float * x, int incx, float * y, int incy)
{
    if (n <= 0)
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::swap_interleave::params params = { n, x, incx, y, incy, sizeof(float) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::swap_interleave>(handle, params);
    });
}
// implement C API iclblasSrot
extern "C"
iclblasStatus_t iclblasSrot(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, float c, float s)
{
    if (n <= 0 || (c == 1 && s == 0))
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Srot::params params = { n, x, incx, y, incy, c, s };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Srot>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSrotm(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, float* param)
{
    if (n <= 0 || param[0] == -2.f)
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Srotm::params params = { n, x, incx, y, incy, param };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Srotm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSasum(iclblasHandle_t handle, int n, float* x, int incx, float* result)
{
    if (n <= 0 || incx <= 0)
    {
        result[0] = 0.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sasum::params params = { n, x, incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sasum>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSdot(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, float* result)
{
    if (n <= 0)
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sdot::params params = { n, x, incx, y, incy, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sdot>(handle, params);
    });
}

// implement C API iclblasSrotg
extern "C"
iclblasStatus_t iclblasSrotg(iclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Srotg::params params = { a, b, c, s };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Srotg>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasStrsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float * A, int lda, float* x, int incx) {
    if (n == 0) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Strsv::params params = { uplo, trans, diag, n, A, lda, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Strsv>(handle, params);
    });

}

extern "C"
iclblasStatus_t iclblasStbsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, float * A, int lda, float * x, int incx)
{
    if (n == 0) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || k < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Stbsv::params params = { uplo, trans, diag, n, k, A, lda, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Stbsv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasStpsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* AP, float* x, int incx) {
    if (n == 0) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Stpsv::params params = { uplo, trans, diag, n, AP, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Stpsv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSger(iclblasHandle_t handle, int m, int n, const float* alpha, float* x, int incx, float* y, int incy, float* A, int lda) {
    if (m == 0 || n == 0 || alpha[0] == 0.f) return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sger::params params = { m, n, *alpha, x, incx, y, incy, A, lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sger>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSsyr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float * x, int incx, float * A, int lda)
{
    if (n == 0 || alpha[0] == 0.f) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssyr::params params = { uplo, n, *alpha, x, incx, A, lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssyr>(handle, params);
    });
}

// implement C API iclblasStrmv
extern "C"
iclblasStatus_t iclblasStrmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* A, int lda, float* x, int incx)
{
    if (n == 0) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;
    
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Strmv::params params = { uplo, trans, diag, n, A, lda, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Strmv>(handle, params);
    });
}

// implement C API iclblasStbmv
extern "C"
iclblasStatus_t iclblasStbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, float* A, int lda, float* x, int incx)
{
    if (n == 0) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || k < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Stbmv::params params = { uplo, trans, diag, n, k, A, lda, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Stbmv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSgbmv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, float* A, int lda, float* x, int incx, const float* beta, float* y, int incy) {
    if (m == 0 || n == 0 || (alpha[0] == 0.f && beta[0] == 1.f)) return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0 || incy == 0 || ku < 0 || kl < 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sgbmv::params params = { trans, m, n, kl, ku, *alpha, A, lda, x, incx, *beta, y, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sgbmv>(handle, params);
    });

}

// implement C API iclblasStpmv
extern "C"
iclblasStatus_t iclblasStpmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* AP, float* x, int incx)
{
    if (n == 0) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Stpmv::params params = { uplo, trans, diag, n,  AP, x, incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Stpmv>(handle, params);
    });
}

// implement C API iclblasSsbmv
extern "C"
iclblasStatus_t iclblasSsbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, char n, char k, const float* alpha, float* A, int lda, float* x, int incx, const float* beta, float* y, int incy)
{
    if (n == 0 || (alpha[0] == 0.f && beta[0] == 1.f)) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || k < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssbmv::params params = { uplo, n, k, *alpha, A, lda, x, incx, *beta, y, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssbmv>(handle, params);
    });
}

// implement C API iclblasSspmv
extern "C"
iclblasStatus_t iclblasSspmv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float* AP, float* x, int incx, const float* beta, float* y, int incy)
{
    if (n == 0 || (alpha[0] == 0.f && beta[0] == 1.f)) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sspmv::params params = { uplo, n, *alpha, AP, x, incx, *beta, y, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sspmv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSsymm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, int m, int n, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc)
{
    if (n == 0 || m == 0 || (*alpha == 0.f && *beta == 1.f))
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssymm::params params = { side, uplo, m, n, *alpha, A, lda, B, ldb, *beta, C, ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssymm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSsyrk(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const float* alpha, float* A, int lda, const float* beta, float* C, int ldc) {
    if (n == 0 || k == 0 || (*alpha == 0.f && *beta == 1.f))
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssyrk::params params = { uplo, trans, n, k, *alpha, A, lda, *beta, C, ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssyrk>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSsyr2k(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc)
{
    if (n == 0 || k == 0 || (*alpha == 0.f && *beta == 1.f))
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssyr2k::params params = { uplo, trans, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssyr2k>(handle, params);
    });
}

// implement C API iclblasSspr2
extern "C"
iclblasStatus_t iclblasSspr2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *x, int incx, float* y, int incy, float* AP)
{
    if (n == 0 || alpha[0] == 0.f) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sspr2::params params = { uplo, n, *alpha, x, incx, y, incy, AP };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sspr2>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSspr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *x, int incx, float* AP) 
{
    if (n == 0 || alpha[0] == 0.f) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sspr::params params = { uplo, n, *alpha, x, incx, AP };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sspr>(handle, params);
    });
}

// implement C API iclblasSsymv
extern "C"
iclblasStatus_t iclblasSsymv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *A, int lda, float *x, int incx, const float* beta, float *y, int incy)
{
    if (n == 0 || (alpha[0] == 0.f && beta[0] == 1.f)) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssymv::params params = { uplo, n, *alpha, A, lda, x, incx, *beta, y, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssymv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSsyr2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *x, int incx, float* y, int incy, float* A, int lda) {
    if (n == 0 || alpha[0] == 0.f) return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ssyr2::params params = { uplo, n, *alpha, x, incx, y, incy, A, lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ssyr2>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasSgemv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, const float* alpha, float *A, int lda, float *x, int incx, const float* beta, float *y, int incy)
{
    if (m == 0 || n == 0 || (alpha[0] == 0.f && beta[0] == 1.f)) return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0 || incy == 0) return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sgemv::params params = { trans, m, n, *alpha, A, lda, x, incx, *beta, y, incy};
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sgemv>(handle, params);
    });
}

// implement C API iclblasSgemm
extern "C"
iclblasStatus_t iclblasSgemm(iclblasHandle_t handle, iclblasOperation_t transa, iclblasOperation_t transb, int m, int n, int k, const float* alpha, float* A, int lda, float* B, int ldb, const float* beta, float* C, int ldc)
{
    if (m < 0 || n < 0 || k < 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Sgemm::params params = { transa, transb, m, n,k, *alpha, A, lda, B, ldb, *beta, C, ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Sgemm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasStrsm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int m, int n, const float* alpha, float * A, int lda, float * B, int ldb)
{
    if (m == 0 || n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Strsm::params params = { side, uplo, trans, diag, m, n, *alpha, A, lda, B, ldb };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Strsm>(handle, params);
    });
}

// implement C API iclblasStrmm
extern "C"
iclblasStatus_t iclblasStrmm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t transa, iclblasDiagType_t diag, int m, int n, const float* alpha, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Strmm::params params = { side, uplo, transa, diag, m, n, *alpha, A, lda, B, ldb, C, ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Strmm>(handle, params);
    });
}
