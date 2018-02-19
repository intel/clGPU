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
#include "functions/swap_interleave.hpp"
#include "functions/Cscal.hpp"
#include "functions/Cdotu.hpp"
#include "functions/Scasum.hpp"
#include "functions/Scnrm2.hpp"
#include "functions/Icamax.hpp"
#include "functions/Icamin.hpp"
#include "functions/Caxpy.hpp"
#include "functions/Crotg.hpp"
#include "functions/Cgbmv.hpp"
#include "functions/Cgeru.hpp"
#include "functions/Cgerc.hpp"
#include "functions/Cgemv.hpp"
#include "functions/Cher.hpp"
#include "functions/Chemv.hpp"
#include "functions/Ctrmv.hpp"
#include "functions/Cdotc.hpp"
#include "functions/Crot.hpp"
#include "functions/Csrot.hpp"
#include "functions/Cher2.hpp"
#include "functions/Chpmv.hpp"
#include "functions/Chpr.hpp"
#include "functions/Chpr2.hpp"
#include "functions/Ctpmv.hpp"
#include "functions/Ctbmv.hpp"
#include "functions/Ctbsv.hpp"
#include "functions/Ctrsv.hpp"
#include "functions/Cgemm.hpp"
#include "functions/Csymm.hpp"
#include "functions/Csyr2k.hpp"
#include "functions/Csyrk.hpp"
#include "functions/Ctpsv.hpp"
#include "functions/Ctrsm.hpp"
#include "functions/Cherk.hpp"
#include "functions/Cher2k.hpp"
#include "functions/Ctrmm.hpp"
#include "functions/Chemm.hpp"
#include "functions/Chbmv.hpp"
#include "functions/Csyr.hpp"

#include "iclBLASImpl.hpp"

extern "C"
iclblasStatus_t iclblasCcopy(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::copy_interleave::params params = { x, y, n, sizeof(oclComplex_t), incx, incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::copy_interleave>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCscal(iclblasHandle_t handle, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cscal::params params = { n, *iclblas::complex_cast(alpha), iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cscal>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCsscal(iclblasHandle_t handle, int n, const float* alpha, oclComplex_t* x, int incx)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cscal::params params = { n, {*alpha, 0.f }, iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cscal>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCswap(iclblasHandle_t handle, int n, oclComplex_t * x, int incx, oclComplex_t * y, int incy)
{
    if (n <= 0)
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::swap_interleave::params params = { n, x, incx, y, incy, sizeof(oclComplex_t) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::swap_interleave>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCdotu(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* result) {
    if (n <= 0) {
        result->val[0] = 0.f;
        result->val[1] = 0.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cdotu::params params = {n, iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, iclblas::complex_cast(result)};
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cdotu>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasScasum(iclblasHandle_t handle, int n, oclComplex_t *x, int incx, float* result)
{
        return iclblas::exception_to_iclblas_status([=]() {
            iclgpu::functions::Scasum::params params = { n, iclblas::complex_cast(x), incx, result };
            iclblas::iclblasTemplate_impl<iclgpu::functions::Scasum>(handle, params);
        });
}

// implement C API iclblasScnrm2
extern "C"
iclblasStatus_t iclblasScnrm2(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, float* result)
{
    if (n <= 0 || incx <= 0)
    {
        *result = 0;
        return ICLBLAS_STATUS_SUCCESS;
    }
    
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Scnrm2::params params = { n, iclblas::complex_cast(x), incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Scnrm2>(handle, params);
    });
}

// implement C API iclblasIcamax
extern "C"
iclblasStatus_t iclblasIcamax(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, int* result)
{
    if (n <= 0 || incx <= 0)
    {
        *result = 0;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Icamax::params params = { n, iclblas::complex_cast(x), incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Icamax>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasIcamin(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, int* result)
{
    if (n <= 0 || incx <= 0)
    {
        *result = 0;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Icamin::params params = { n, iclblas::complex_cast(x), incx, result };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Icamin>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCaxpy(iclblasHandle_t handle, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy) {
    if (n <= 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f)) {
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Caxpy::params params = { n, iclblas::complex_cast(*alpha), iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Caxpy>(handle, params);
    });
}

// implement C API iclblasCrotg
extern "C"
iclblasStatus_t iclblasCrotg(iclblasHandle_t handle, oclComplex_t* a, oclComplex_t* b, float* c, oclComplex_t* s)
{
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Crotg::params params = { iclblas::complex_cast(a), iclblas::complex_cast(b), c, iclblas::complex_cast(s) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Crotg>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCgbmv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, int kl, int ku, const oclComplex_t* alpha, oclComplex_t * A, int lda, oclComplex_t * x, int incx, const oclComplex_t* beta, oclComplex_t * y, int incy)
{
    if (m == 0 || n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cgbmv::params params = { trans, m, n, kl, ku, *iclblas::complex_cast(alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx, *iclblas::complex_cast(beta), iclblas::complex_cast(y), incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cgbmv>(handle, params);
    });

}

extern "C"
iclblasStatus_t iclblasCgeru(iclblasHandle_t handle, int m, int n, const oclComplex_t* alpha, oclComplex_t * x, int incx, oclComplex_t * y, int incy, oclComplex_t * A, int lda)
{
    if (m == 0 || n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cgeru::params params = { m, n, *iclblas::complex_cast(alpha), iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, iclblas::complex_cast(A), lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cgeru>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCgerc(iclblasHandle_t handle, int m, int n, const oclComplex_t* alpha, oclComplex_t * x, int incx, oclComplex_t * y, int incy, oclComplex_t * A, int lda)
{
    if (m == 0 || n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cgerc::params params = { m, n, *iclblas::complex_cast(alpha), iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, iclblas::complex_cast(A), lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cgerc>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCgemv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy)
{
    if (m == 0 || n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (m < 0 || n < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cgemv::params params = { trans, m, n, *iclblas::complex_cast(alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx, *iclblas::complex_cast(beta), iclblas::complex_cast(y), incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cgemv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCher(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, oclComplex_t* x, int incx, oclComplex_t* A, int lda)
{
    if (n == 0 || *alpha == 0.f)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cher::params params = {uplo,  n, *alpha, iclblas::complex_cast(x), incx, iclblas::complex_cast(A), lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cher>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasChemv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy)
{
    if (n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Chemv::params params = { uplo, n, *iclblas::complex_cast(alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx, *iclblas::complex_cast(beta), iclblas::complex_cast(y), incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Chemv>(handle, params);
    });
}

// implement C API iclblasCtrmv
extern "C"
iclblasStatus_t iclblasCtrmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* A, int lda, oclComplex_t* x, int incx)
{
    if (n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctrmv::params params = { uplo, trans, diag, n, iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctrmv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCdotc(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* result) {
    if (n <= 0) {
        result->val[0] = 0.f;
        result->val[1] = 0.f;
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cdotc::params params = { n, iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, iclblas::complex_cast(result) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cdotc>(handle, params);
    });
}

// implement C API iclblasCrot
extern "C"
iclblasStatus_t iclblasCrot(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, const float* c, const oclComplex_t* s)
{
    if (n <= 0 || (*c == 1 && (s->val[0] == 0 && s->val[1] == 0)))
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Crot::params params = { n, iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, *c, iclblas::complex_cast(*s) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Crot>(handle, params);
    });
}

// implement C API iclblasCsrot
extern "C"
iclblasStatus_t iclblasCsrot(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, const float* c, const float* s)
{
    if (n <= 0 || (*c == 1 && *s == 0))
        return ICLBLAS_STATUS_SUCCESS;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Csrot::params params = { n, iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, *c, *s };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Csrot>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCher2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* A, int lda)
{
    if (n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cher2::params params = { uplo, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, iclblas::complex_cast(A), lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cher2>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasChpmv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* AP, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy)
{
    if (n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Chpmv::params params = { uplo, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(AP), iclblas::complex_cast(x), incx,
            iclblas::complex_cast(*beta), iclblas::complex_cast(y), incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Chpmv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasChpr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, oclComplex_t*x, int incx, oclComplex_t* AP)
{
    if (n == 0 || *alpha == 0.f)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Chpr::params params = { uplo, n, *alpha, iclblas::complex_cast(x), incx, iclblas::complex_cast(AP) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Chpr>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasChpr2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* AP)
{
    if (n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Chpr2::params params = { uplo, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(x), incx, iclblas::complex_cast(y), incy, iclblas::complex_cast(AP) };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Chpr2>(handle, params);
    });
}
// implement C API iclblasCtpmv
extern "C"
iclblasStatus_t iclblasCtpmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* AP, oclComplex_t* x, int incx)
{
    if (n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctpmv::params params = { uplo, trans, diag, n, iclblas::complex_cast(AP), iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctpmv>(handle, params);
    });
}

// implement C API iclblasCtbmv
extern "C"
iclblasStatus_t iclblasCtbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, oclComplex_t* A, int lda, oclComplex_t* x, int incx)
{
    if (n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || k < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctbmv::params params = { uplo, trans, diag, n, k, iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctbmv>(handle, params);
    });
}

// implement C API iclblasCtbsv
extern "C"
iclblasStatus_t iclblasCtbsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, oclComplex_t* A, int lda, oclComplex_t* x, int incx)
{
    if (n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || k < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctbsv::params params = { uplo, trans, diag, n, k, iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctbsv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCtrsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* A, int lda, oclComplex_t* x, int incx)
{
    if (n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctrsv::params params = { uplo, trans, diag, n, iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctrsv>(handle, params);
    });
}

// implement C API iclblasCgemm
extern "C"
iclblasStatus_t iclblasCgemm(iclblasHandle_t handle, iclblasOperation_t transa, iclblasOperation_t transb, int m, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc)
{
    if (m < 0 || n < 0 || k < 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cgemm::params params = { transa, transb, m, n, k, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(B), ldb, iclblas::complex_cast(*beta), iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cgemm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCsymm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc) {
    if (n == 0 || m == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f)) {
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Csymm::params params = { side, uplo, m, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda,
            iclblas::complex_cast(B), ldb, iclblas::complex_cast(*beta), iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Csymm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCsyr2k(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc) {
    if (n == 0 || k == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f)) {
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Csyr2k::params params = { uplo, trans, n, k, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda,
            iclblas::complex_cast(B), ldb, iclblas::complex_cast(*beta), iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Csyr2k>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCsyrk(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, const oclComplex_t* beta, oclComplex_t* C, int ldc) {
    if (n == 0 || k == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f)) {
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Csyrk::params params = { uplo, trans, n, k, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda,
            iclblas::complex_cast(*beta), iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Csyrk>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCtpsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* AP, oclComplex_t* x, int incx)
{
    if (n == 0)
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctpsv::params params = { uplo, trans, diag, n, iclblas::complex_cast(AP), iclblas::complex_cast(x), incx };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctpsv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCtrsm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb) {
    if (m == 0 || n == 0) {
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctrsm::params params = { side, uplo, trans, diag, m, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(B), ldb };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctrsm>(handle, params);
    });
}

// implement C API iclblasCherk
extern "C"
iclblasStatus_t iclblasCherk(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const float* alpha, oclComplex_t* A, int lda, const float* beta, oclComplex_t* C, int ldc)
{
    if (n < 0 || k < 0)
        return ICLBLAS_STATUS_INVALID_VALUE;
    
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cherk::params params = { uplo, trans, n, k, *alpha, iclblas::complex_cast(A), lda, *beta, iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cherk>(handle, params);
    });
}

// implement C API iclblasCher2k
extern "C"
iclblasStatus_t iclblasCher2k(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const float* beta, oclComplex_t* C, int ldc)
{
    if (n < 0 || k < 0)
        return ICLBLAS_STATUS_INVALID_VALUE;
    
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Cher2k::params params = { uplo, trans, n, k, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(B), ldb, *beta, iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Cher2k>(handle, params);
    });
}


// implement C API iclblasCtrmm
extern "C"
iclblasStatus_t iclblasCtrmm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t transa, iclblasDiagType_t diag, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, oclComplex_t* C, int ldc)
{
    if (n <= 0 || m <= 0 || (uplo != 0 && uplo != 1))
        return ICLBLAS_STATUS_SUCCESS;
    
    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Ctrmm::params params = { side, uplo, transa, diag, m, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(B), ldb, iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Ctrmm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasChemm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc) {
    if (m == 0 || n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f)) {
        return ICLBLAS_STATUS_SUCCESS;
    }

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Chemm::params params = { side, uplo, m, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(A), lda,
            iclblas::complex_cast(B), ldb, iclblas::complex_cast(*beta), iclblas::complex_cast(C), ldc };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Chemm>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasChbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy)
{
    if (n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f && beta->val[0] == 1.f && beta->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || k < 0 || incx == 0 || incy == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Chbmv::params params = { uplo, n, k, *iclblas::complex_cast(alpha), iclblas::complex_cast(A), lda, iclblas::complex_cast(x), incx, *iclblas::complex_cast(beta), iclblas::complex_cast(y), incy };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Chbmv>(handle, params);
    });
}

extern "C"
iclblasStatus_t iclblasCsyr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* A, int lda) {
    if (n == 0 || (alpha->val[0] == 0.f && alpha->val[1] == 0.f))
        return ICLBLAS_STATUS_SUCCESS;

    if (n < 0 || incx == 0)
        return ICLBLAS_STATUS_INVALID_VALUE;

    return iclblas::exception_to_iclblas_status([=]() {
        iclgpu::functions::Csyr::params params = { uplo, n, iclblas::complex_cast(*alpha), iclblas::complex_cast(x), incx, iclblas::complex_cast(A), lda };
        iclblas::iclblasTemplate_impl<iclgpu::functions::Csyr>(handle, params);
    });
}
