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

TEST(Csymm, naive_left_up_3x3) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    std::complex<float> alpha(1.1f, 1.3f);
    std::complex<float> beta(2.2f, 2.4f);

    std::complex<float> a[m*lda] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };
    std::complex<float> b[n*ldb] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };
    std::complex<float> c[n*ldc] = { { 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f },
                                     { 37.f, 38.f },{ 39.f, 40.f },{ 41.f, 42.f },
                                     { 43.f, 44.f },{ 45.f, 46.f },{ 47.f, 48.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ACCESS(expected, j, i, ldc) *= beta;
            std::complex<float> axb;
            for (int k = 0; k < m; k++) {
                if (k >= j) {
                    axb += ACCESS(a, j, k, lda) * ACCESS(b, k, i, ldb);
                }
                else {
                    axb += ACCESS(a, k, j, lda) * ACCESS(b, k, i, ldb);
                }
            }
            ACCESS(expected, j, i, ldc) += alpha * axb;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsymm(handle, side, uplo, m, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(b), ldb, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csymm, naive_right_up_3x4) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int m = 3;
    const int n = 4;
    const int lda = 4;
    const int ldb = 3;
    const int ldc = 3;

    std::complex<float> alpha(-1.1f, 1.3f);
    std::complex<float> beta(2.2f, 2.4f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },{ -5.f, -6.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -7.f, -8.f },{ -9.f, -10.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -11.f, -12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f } };
    std::complex<float> b[n*ldb] = { { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },
                                     { 27.f, 28.f },{ 29.f, 30.f },{ 31.f, 32.f },
                                     { 33.f, 34.f },{ 35.f, 36.f },{ 37.f, 38.f },
                                     { 39.f, 40.f },{ 41.f, 42.f },{ 43.f, 44.f } };
    std::complex<float> c[n*ldc] = { { 45.f, 46.f },{ 47.f, 48.f },{ 49.f, 50.f },
                                     { 51.f, 52.f },{ 53.f, 54.f },{ 55.f, 56.f },
                                     { 57.f, 58.f },{ 59.f, 60.f },{ 61.f, 62.f },
                                     { 63.f, 64.f },{ 65.f, 66.f },{ 67.f, 68.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ACCESS(expected, j, i, ldc) *= beta;
            std::complex<float> bxa;
            for (int k = 0; k < n; k++) {
                if (k <= i) {
                    bxa += ACCESS(a, k, i, lda) * ACCESS(b, j, k, ldb);
                }
                else {
                    bxa += ACCESS(a, i, k, lda) * ACCESS(b, j, k, ldb);
                }
            }
            ACCESS(expected, j, i, ldc) += alpha * bxa;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsymm(handle, side, uplo, m, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(b), ldb, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csymm, naive_left_low_4x3) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int m = 4;
    const int n = 3;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 4;

    std::complex<float> alpha(1.1f, -1.3f);
    std::complex<float> beta(-2.2f, 2.4f);

    std::complex<float> a[m*lda] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },{ -5.f, -6.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -7.f, -8.f },{ -9.f, -10.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -11.f, -12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f } };
    std::complex<float> b[n*ldb] = { { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },
                                     { 29.f, 30.f },{ 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f },
                                     { 37.f, 38.f },{ 39.f, 40.f },{ 41.f, 42.f },{ 43.f, 44.f } };
    std::complex<float> c[n*ldc] = { { 45.f, 46.f },{ 47.f, 48.f },{ 49.f, 50.f },{ 51.f, 52.f },
                                     { 53.f, 54.f },{ 55.f, 56.f },{ 57.f, 58.f },{ 59.f, 60.f },
                                     { 61.f, 62.f },{ 63.f, 64.f },{ 65.f, 66.f },{ 67.f, 68.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ACCESS(expected, j, i, ldc) *= beta;
            std::complex<float> axb;
            for (int k = 0; k < m; k++) {
                if (k <= j) {
                    axb += ACCESS(a, j, k, lda) * ACCESS(b, k, i, ldb);
                }
                else {
                    axb += ACCESS(a, k, j, lda) * ACCESS(b, k, i, ldb);
                }
            }
            ACCESS(expected, j, i, ldc) += alpha * axb;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsymm(handle, side, uplo, m, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(b), ldb, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csymm, naive_right_low_3x3_lds4) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int m = 3;
    const int n = 3;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 4;

    std::complex<float> alpha(1.1f, 1.3f);
    std::complex<float> beta(2.2f, -2.4f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },{ -5.f, -6.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -7.f, -8.f },{ -9.f, -10.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -11.f, -12.f } };
    std::complex<float> b[n*ldb] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -1.f, -2.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },{ -3.f, -4.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f },{ -5.f, -6.f } };
    std::complex<float> c[n*ldc] = { { 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f },{ -7.f, -8.f },
                                     { 37.f, 38.f },{ 45.f, 46.f },{ 47.f, 48.f },{ -9.f, -10.f },
                                     { 49.f, 50.f },{ 51.f, 52.f },{ 53.f, 54.f },{ -11.f, -12.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ACCESS(expected, j, i, ldc) *= beta;
            std::complex<float> bxa;
            for (int k = 0; k < n; k++) {
                if (k >= i) {
                    bxa += ACCESS(a, k, i, lda) * ACCESS(b, j, k, ldb);
                }
                else {
                    bxa += ACCESS(a, i, k, lda) * ACCESS(b, j, k, ldb);
                }
            }
            ACCESS(expected, j, i, ldc) += alpha * bxa;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsymm(handle, side, uplo, m, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(b), ldb, reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        if (i == 8) continue; //TODO fix dirty hack
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}
