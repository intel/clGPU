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

TEST(Cgbmv, 5x5_ntrans_u1l1) {
    const auto trans = ICLBLAS_OP_N;

    const int m = 5;
    const int n = 5;
    const int ku = 1;
    const int kl = 1;
    std::complex<float> alpha = { 1.1f, -2.1f };
    const int lda = 3;
    const int incx = 1;
    std::complex<float> beta = { -1.3f, 2.2f };
    const int incy = 1;

    std::complex<float> a[lda*n] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f },
                              { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f },
                              { 13.f, 14.f }, { 15.f, 16.f }, { 17.f, 18.f },
                              { 19.f, 20.f }, { 21.f, 22.f }, { 23.f, 24.f },
                              { 25.f, 26.f }, { 27.f, 28.f }, { 29.f, 30.f } };
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f }, { 9.f, 10.f } };
    std::complex<float> y[m*incy] = { { 11.f, 12.f },{ 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f } };
    std::complex<float> expected[m*incy];

    for (int i = 0; i < m*incy; i++) expected[i] = y[i];
    for (int i = 0; i < m; i++) {
        expected[i*incy] = { 0.f, 0.f };
        for (int j = std::max(i - kl, 0); j<std::min(i + ku + 1, n); j++) {
            expected[i*incy] += ACCESS(a, ku + i - j, j, lda) * x[j*incx];
        }
        expected[i*incy] *= alpha;
        expected[i*incy] += beta * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgbmv(handle, trans, m, n, kl, ku, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Cgbmv, 3x2_ntrans_u1l1) {
    const auto trans = ICLBLAS_OP_N;

    const int m = 3;
    const int n = 2;
    const int ku = 1;
    const int kl = 1;
    std::complex<float> alpha = { 1.1f, -2.1f };
    const int lda = 4;
    const int incx = 1;
    std::complex<float> beta = { -1.3f, 2.2f };
    const int incy = 1;

    std::complex<float> a[lda*n] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { -1.f, -2.f },
                              { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { -3.f, -4.f } };
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f } };
    std::complex<float> y[m*incy] = { { 5.f, 6.f }, { 7.f, 8.f }, { 9.f, 10.f } };
    std::complex<float> expected[m*incy];
    
    for (int i = 0; i < m*incy; i++) expected[i] = y[i];
    for (int i = 0; i < m; i++) {
        expected[i*incy] = { 0.f, 0.f };
        for (int j = std::max(i - kl, 0); j<std::min(i + ku + 1, n); j++) {
            expected[i*incy] += ACCESS(a, ku + i - j, j, lda) * x[j*incx];
        }
        expected[i*incy] *= alpha;
        expected[i*incy] += beta * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgbmv(handle, trans, m, n, kl, ku, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Cgbmv, 3x3_trans_inc2) {
    const auto trans = ICLBLAS_OP_T;

    const int m = 3;
    const int n = 3;
    const int ku = 0;
    const int kl = 1;
    std::complex<float> alpha = { 1.1f, -2.1f };
    const int lda = 2;
    const int incx = 2;
    std::complex<float> beta = { -1.3f, 2.2f };
    const int incy = 2;

    std::complex<float> a[lda*n] = { { 1.f, 2.f }, { 3.f, 4.f },
                              { 5.f, 6.f }, { 7.f, 8.f },
                              { 9.f, 10.f }, { 11.f, 12.f } };
    std::complex<float> x[m*incx] = { { 1.f, 2.f }, { -11.f, -11.f }, { 3.f, 4.f }, { -11.f, 11.f }, { 5.f, 6.f }, { -11.f, 11.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f }, { -11.f, -11.f }, { 9.f, 10.f }, { -11.f, -11.f }, { 11.f, 12.f }, { -11.f, -11.f } };
    std::complex<float> expected[n*incy];

    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] = { 0.f, 0.f };
        for (int j = std::max(i - ku, 0); j<std::min(i + kl + 1, m); j++) {
            expected[i*incy] += ACCESS(a, ku + j - i, i, lda) * x[j*incx];
        }
        expected[i*incy] *= alpha;
        expected[i*incy] += beta * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgbmv(handle, trans, m, n, kl, ku, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Cgbmv, 3x3_trans_u2l1) {
    const auto trans = ICLBLAS_OP_T;

    const int m = 3;
    const int n = 3;
    const int ku = 2;
    const int kl = 1;
    std::complex<float> alpha = { 1.1f, -2.1f };
    const int lda = 4;
    const int incx = 1;
    std::complex<float> beta = { -1.3f, 2.2f };
    const int incy = 1;

    std::complex<float> a[lda*n] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f },
                                     { 9.f, 10.f }, { 11.f, 12.f }, { 13.f, 14.f }, { 15.f, 16.f },
                                     { 17.f, 18.f }, { 19.f, 20.f }, { 21.f, 22.f }, { 23.f, 24.f } };
    std::complex<float> x[m*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f } };
    std::complex<float> expected[n*incy];

    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] = { 0.f, 0.f };
        for (int j = std::max(i - ku, 0); j<std::min(i + kl + 1, m); j++) {
            expected[i*incy] += ACCESS(a, ku + j - i, i, lda) * x[j*incx];
        }
        expected[i*incy] *= alpha;
        expected[i*incy] += beta * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgbmv(handle, trans, m, n, kl, ku, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Cgbmv, 3x3_conjg_u2l1) {
    const auto trans = ICLBLAS_OP_C;

    const int m = 3;
    const int n = 3;
    const int ku = 2;
    const int kl = 1;
    std::complex<float> alpha = { -0.1f, 2.1f };
    const int lda = 4;
    const int incx = 1;
    std::complex<float> beta = { 1.1f, 1.1f };
    const int incy = 1;

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },
                                     { 9.f, 10.f },{ 11.f, 12.f },{ 13.f, 14.f },{ 15.f, 16.f },
                                     { 17.f, 18.f },{ 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f } };
    std::complex<float> x[m*incx] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f } };
    std::complex<float> y[n*incy] = { { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };
    std::complex<float> expected[n*incy];

    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] = { 0.f, 0.f };
        for (int j = std::max(i - ku, 0); j<std::min(i + kl + 1, m); j++) {
            expected[i*incy] += conj(ACCESS(a, ku + j - i, i, lda)) * x[j*incx];
        }
        expected[i*incy] *= alpha;
        expected[i*incy] += beta * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgbmv(handle, trans, m, n, kl, ku, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}

TEST(Cgbmv, 5x4_conjg_u1l1) {
    const auto trans = ICLBLAS_OP_C;

    const int m = 5;
    const int n = 4;
    const int ku = 1;
    const int kl = 1;
    std::complex<float> alpha = { 100.f, 200.f };
    const int lda = 3;
    const int incx = 1;
    std::complex<float> beta = { 300.f, 400.f };
    const int incy = 1;

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                              { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },
                              { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                              { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f } };
    std::complex<float> x[m*incx] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f }, { 9.f, 10.f } };
    std::complex<float> y[n*incy] = { { 11.f, 12.f },{ 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f } };
    std::complex<float> expected[n*incy];

    for (int i = 0; i < n*incy; i++) expected[i] = y[i];
    for (int i = 0; i < n; i++) {
        expected[i*incy] = { 0.f, 0.f };
        for (int j = std::max(i - ku, 0); j<std::min(i + kl + 1, m); j++) {
            expected[i*incy] += conj(ACCESS(a, ku + j - i, i, lda)) * x[j*incx];
        }
        expected[i*incy] *= alpha;
        expected[i*incy] += beta * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgbmv(handle, trans, m, n, kl, ku, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(y), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_COMPLEX_EQ(expected[i], y[i]);
    }
}
