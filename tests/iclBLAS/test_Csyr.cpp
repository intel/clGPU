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

#include <gtest/gtest.h>
#include <iclBLAS.h>
#include <complex>

#define IDX(m, n, ld) (n)*(ld) + (m)

#define EXPECT_COMPLEX_EQ(expected, result) \
    EXPECT_FLOAT_EQ(expected.real(), result.real()); \
    EXPECT_FLOAT_EQ(expected.imag(), result.imag())

using complex_f = std::complex<float>;

std::vector<complex_f> cpuCsyr_upper(const int n, const complex_f alpha, const std::vector<complex_f> x, const int incx, const std::vector<complex_f> A, const int lda) {
    auto result = A;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            result[IDX(j, i, lda)] += alpha * x[j * incx] * x[i * incx];
        }
    }
    return result;
}

std::vector<complex_f> cpuCsyr_lower(const int n, const complex_f alpha, const std::vector<complex_f> x, const int incx, const std::vector<complex_f> A, const int lda) {
    auto result = A;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            result[IDX(j, i, lda)] += alpha * x[j * incx] * x[i * incx];
        }
    }
    return result;
}

TEST(Csyr, 6x5_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 5;
    const int incx = 1;
    const int lda = 6;

    complex_f alpha = { 1.1f, 1.3f };

    std::vector<complex_f> x(n * incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = { 1.f * i, 1.f * i + 1 };
    }
    std::vector<complex_f> A(n * lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[IDX(j, i, lda)] = { 1.f*i, 1.f*j };
        }
    }

    auto expected = cpuCsyr_upper(n, alpha, x, incx, A, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyr(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(A.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], A[i]);
    }
}

TEST(Csyr, 17x17_up_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 17;
    const int incx = 2;
    const int lda = 17;

    complex_f alpha = { 1.1f, 1.3f };

    std::vector<complex_f> x(n * incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = { 1.f * i, 1.f * i + 1 };
    }
    std::vector<complex_f> A(n * lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[IDX(j, i, lda)] = { 1.f*i, 1.f*j };
        }
    }

    auto expected = cpuCsyr_upper(n, alpha, x, incx, A, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyr(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(A.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], A[i]);
    }
}

TEST(Csyr, 17x16_low_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 16;
    const int incx = 1;
    const int lda = 17;

    complex_f alpha = { 1.1f, 1.3f };

    std::vector<complex_f> x(n * incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = { 1.f * i, 1.f * i + 1 };
    }
    std::vector<complex_f> A(n * lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[IDX(j, i, lda)] = { 1.f*i, 1.f*j };
        }
    }

    auto expected = cpuCsyr_lower(n, alpha, x, incx, A, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyr(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(A.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], A[i]);
    }
}

TEST(Csyr, 16x16_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 16;
    const int incx = 2;
    const int lda = 16;

    complex_f alpha = { 1.1f, 1.3f };

    std::vector<complex_f> x(n * incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = { 1.f * i, 1.f * i + 1 };
    }
    std::vector<complex_f> A(n * lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[IDX(j, i, lda)] = { 1.f*i, 1.f*j };
        }
    }

    auto expected = cpuCsyr_lower(n, alpha, x, incx, A, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyr(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(A.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], A[i]);
    }
}
