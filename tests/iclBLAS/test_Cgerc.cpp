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

#define ACCESS(a, m, n, N) a[(n)*(N) + (m)]

#define EXPECT_COMPLEX_EQ(expected, result) \
    EXPECT_FLOAT_EQ(expected.real(), result.real()); \
    EXPECT_FLOAT_EQ(expected.imag(), result.imag())

TEST(Cgerc, naive_3x5_incx2) {
    const int m = 3;
    const int n = 5;
    const int lda = 3;
    const int incx = 2;
    const int incy = 1;
    std::complex<float> alpha = { 1.1f, -10.1f };
    std::complex<float> x[m*incx] = { { 1.f, 2.f }, { -1.f, -1.f }, { 3.f, 4.f}, { -1.f, -1.f }, { 5.f, 6.f },{ -1.f, -1.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { 13.f, 14.f }, { 15.f, 16.f } };
    std::complex<float> a[lda*n] = { { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, 
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f } };

    std::complex<float> expected[lda*n];

    for (int i = 0; i < lda*n; i++) expected[i] = a[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ACCESS(expected, j, i, lda) += alpha * x[j*incx] * std::conj(y[i*incy]);
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgerc(handle, m, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x), incx,
        reinterpret_cast<oclComplex_t*>(y), incy, reinterpret_cast<oclComplex_t*>(a), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cgerc, naive_3x5_no_inc) {
    const int m = 3;
    const int n = 5;
    const int lda = 4;
    const int incx = 1;
    const int incy = 1;
    std::complex<float> alpha = { 1.1f, 2.1f };
    std::complex<float> x[m*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { 13.f, 14.f }, { 15.f, 16.f } };
    std::complex<float> a[lda*n] = { { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { 1.f, 2.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { 1.f, 2.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { 1.f, 2.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { 1.f, 2.f },
                                     { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { 1.f, 2.f } };

    std::complex<float> expected[lda*n];

    for (int i = 0; i < lda*n; i++) expected[i] = a[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ACCESS(expected, j, i, lda) += alpha * x[j*incx] * std::conj(y[i*incy]);
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgerc(handle, m, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x), incx,
        reinterpret_cast<oclComplex_t*>(y), incy, reinterpret_cast<oclComplex_t*>(a), lda);    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}
