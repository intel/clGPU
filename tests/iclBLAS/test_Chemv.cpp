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

#define ACCESS(a, m, n, lda) a[(n)*(lda) + (m)]

#define EXPECT_COMPLEX_EQ(expected, result) \
    EXPECT_FLOAT_EQ(expected.real(), result.real()); \
    EXPECT_FLOAT_EQ(expected.imag(), result.imag())

TEST(Chemv, naive_up_4x3) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 3;
    const int lda = 4;
    const int incx = 1;
    const int incy = 1;

    std::complex<float> alpha(1.1f, -1.2f);
    std::complex<float> beta(1.3f, 2.3f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f },
                                     { 3.f, 4.f }, { 5.f, 6.f }, { -7.f, -8.f }, { -9.f, -10.f },
                                     { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { -11.f, -12.f } };
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f } };

    std::complex<float> expected[n*incy];
    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] *= beta;
        std::complex<float> prod;
        for (int j = 0; j < n; j++) {
            if (j > i) {
                prod += ACCESS(a, i, j, lda)*x[j*incx];
            }
            else if (j == i) {
                prod += std::complex<float>(ACCESS(a, i, j, lda).real(), 0.f)*x[j*incx];
            }
            else {
                prod += conj(ACCESS(a, j, i, lda))*x[j*incx];
            }
        }
        expected[i*incy] += alpha*prod;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChemv(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Chemv, naive_up_4x3_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 3;
    const int lda = 4;
    const int incx = 2;
    const int incy = 1;

    std::complex<float> alpha(1.1f, -1.2f);
    std::complex<float> beta(1.3f, 2.3f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f },
                                     { 3.f, 4.f }, { 5.f, 6.f }, { -7.f, -8.f }, { -9.f, -10.f },
                                     { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { -11.f, -12.f } };
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f } };
    std::complex<float> y[n*incy] = { { 13.f, 14.f }, { 15.f, 16.f }, { 17.f, 18.f } };

    std::complex<float> expected[n*incy];
    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] *= beta;
        std::complex<float> prod;
        for (int j = 0; j < n; j++) {
            if (j > i) {
                prod += ACCESS(a, i, j, lda)*x[j*incx];
            }
            else if (j == i) {
                prod += std::complex<float>(ACCESS(a, i, j, lda).real(), 0.f)*x[j*incx];
            }
            else {
                prod += conj(ACCESS(a, j, i, lda))*x[j*incx];
            }
        }
        expected[i*incy] += alpha*prod;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChemv(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Chemv, naive_low_3x3) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 3;
    const int lda = 3;
    const int incx = 1;
    const int incy = 1;

    std::complex<float> alpha(-0.1f, -0.2f);
    std::complex<float> beta(1.3f, 2.3f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f },
                                     { -1.f, -2.f }, { 7.f, 8.f }, { 9.f, 10.f },
                                     { -3.f, -4.f }, { -5.f, -6.f }, { 11.f, 12.f } };
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f } };

    std::complex<float> expected[n*incy];
    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] *= beta;
        std::complex<float> prod;
        for (int j = 0; j < n; j++) {
            if (j < i) {
                prod += ACCESS(a, i, j, lda)*x[j*incx];
            }
            else if (j == i) {
                prod += std::complex<float>(ACCESS(a, i, j, lda).real(), 0.f)*x[j*incx];
            }
            else {
                prod += conj(ACCESS(a, j, i, lda))*x[j*incx];
            }
        }
        expected[i*incy] += alpha*prod;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChemv(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Chemv, naive_low_4x3_beta0) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 3;
    const int lda = 4;
    const int incx = 1;
    const int incy = 1;

    std::complex<float> alpha(1024.f, -128.f);
    std::complex<float> beta(0.f, 0.f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { -1.f, -2.f },
                                     { -3.f, -4.f }, { 7.f, 8.f },{ 9.f, 10.f }, { -5.f, -6.f },
                                     { -7.f, -8.f }, { -9.f, -10.f }, { 11.f, 12.f } };
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f } };
    std::complex<float> y[n*incy] = { { -7.f, 8.f }, { -9.f, 10.f }, { -11.f, 12.f } };

    std::complex<float> expected[n*incy];
    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] *= beta;
        std::complex<float> prod;
        for (int j = 0; j < n; j++) {
            if (j < i) {
                prod += ACCESS(a, i, j, lda)*x[j*incx];
            }
            else if (j == i) {
                prod += std::complex<float>(ACCESS(a, i, j, lda).real(), 0.f)*x[j*incx];
            }
            else {
                prod += conj(ACCESS(a, j, i, lda))*x[j*incx];
            }
        }
        expected[i*incy] += alpha*prod;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChemv(handle, uplo, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}
