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

TEST(Csyrk, 3x3_up_ntrans) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const int n = 3;
    const int k = 3;
    const int lda = 3;
    const int ldc = 3;

    std::complex<float> alpha(1.1f, 1.3f);
    std::complex<float> beta(2.2f, 1.4f);

    std::complex<float> a[k*lda] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f } };
    std::complex<float> c[n*ldc] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            std::complex<float> this_sum;
            for (int l = 0; l < k; l++) {
                this_sum += ACCESS(a, j, l, lda) * ACCESS(a, i, l, lda);
            }
            this_sum *= alpha;
            ACCESS(expected, j, i, ldc) = beta * ACCESS(expected, j, i, ldc) + this_sum;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyrk(handle, uplo, trans, n, k, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csyrk, 4x3_up_trans) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const int n = 4;
    const int k = 3;
    const int lda = 3;
    const int ldc = 4;

    std::complex<float> alpha(-1.1f, 1.3f);
    std::complex<float> beta(2.2f, 1.4f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f }, { 23.f, 24.f } };
    std::complex<float> c[n*ldc] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },{ -5.f, -6.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -7.f, -8.f },{ -9.f, -10.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -11.f, -12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            std::complex<float> this_sum;
            for (int l = 0; l < k; l++) {
                this_sum += ACCESS(a, l, j, lda) * ACCESS(a, l, i, lda);
            }
            this_sum *= alpha;
            ACCESS(expected, j, i, ldc) = beta * ACCESS(expected, j, i, ldc) + this_sum;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyrk(handle, uplo, trans, n, k, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csyrk, 3x4_up_conj) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_C;
    const int n = 3;
    const int k = 4;
    const int lda = 4;
    const int ldc = 3;

    std::complex<float> alpha(-1.1f, -1.3f);
    std::complex<float> beta(2.2f, 1.4f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },
                                     { 9.f, 10.f },{ 11.f, 12.f },{ 13.f, 14.f },{ 15.f, 16.f },
                                     { 17.f, 18.f },{ 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f } };
    std::complex<float> c[n*ldc] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            std::complex<float> this_sum;
            for (int l = 0; l < k; l++) {
                this_sum += std::conj(ACCESS(a, l, j, lda)) * std::conj(ACCESS(a, l, i, lda));
            }
            this_sum *= alpha;
            ACCESS(expected, j, i, ldc) = beta * ACCESS(expected, j, i, ldc) + this_sum;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyrk(handle, uplo, trans, n, k, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csyrk, 3x3_low_ntrans) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const int n = 3;
    const int k = 3;
    const int lda = 3;
    const int ldc = 3;

    std::complex<float> alpha(1.1f, 1.3f);
    std::complex<float> beta(-2.2f, 1.4f);

    std::complex<float> a[k*lda] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f } };
    std::complex<float> c[n*ldc] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { -1.f, -2.f },{ 7.f, 8.f },{ 9.f, 10.f },
                                     { -3.f, -4.f },{ -5.f, -6.f },{ 11.f, 12.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            std::complex<float> this_sum;
            for (int l = 0; l < k; l++) {
                this_sum += ACCESS(a, j, l, lda) * ACCESS(a, i, l, lda);
            }
            this_sum *= alpha;
            ACCESS(expected, j, i, ldc) = beta * ACCESS(expected, j, i, ldc) + this_sum;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyrk(handle, uplo, trans, n, k, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csyrk, 3x4_low_trans) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_T;
    const int n = 3;
    const int k = 4;
    const int lda = 4;
    const int ldc = 3;

    std::complex<float> alpha(-1.1f, 1.3f);
    std::complex<float> beta(-2.2f, 1.4f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ -1.f, -2.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -3.f, -4.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -5.f, -6.f } };
    std::complex<float> c[n*ldc] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { -1.f, -2.f },{ 7.f, 8.f },{ 9.f, 10.f },
                                     { -3.f, -4.f },{ -5.f, -6.f },{ 11.f, 12.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            std::complex<float> this_sum;
            for (int l = 0; l < k; l++) {
                this_sum += ACCESS(a, l, j, lda) * ACCESS(a, l, i, lda);
            }
            this_sum *= alpha;
            ACCESS(expected, j, i, ldc) = beta * ACCESS(expected, j, i, ldc) + this_sum;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyrk(handle, uplo, trans, n, k, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}

TEST(Csyrk, 4x3_low_conj_lds5) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const int n = 4;
    const int k = 3;
    const int lda = 5;
    const int ldc = 5;

    std::complex<float> alpha(1.1f, -1.3f);
    std::complex<float> beta(-2.2f, 1.4f);

    std::complex<float> a[n*lda] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -5.f, -6.f },{ -7.f, -8.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -9.f, -10.f },{ -11.f, -12.f },
                                     { 19.f, 20.f }, { 21.f, 22.f }, { 23.f, 24.f}, { -13.f, -14.f }, { -15.f, -16.f } };
    std::complex<float> c[n*ldc] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ -1.f, -2.f },
                                     { -3.f, -4.f },{ 9.f, 10.f },{ 11.f, 12.f },{ 13.f, 14.f },{ -5.f, -6.f },
                                     { -7.f, -8.f },{ -9.f, -10.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -11.f, -12.f },
                                     { -13.f, -14.f },{ -15.f, -16.f },{ -17.f, -18.f },{ 19.f, 20.f },{ -19.f, -20.f } };

    std::complex<float> expected[n*ldc];
    for (int i = 0; i < n*ldc; i++) expected[i] = c[i];
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            std::complex<float> this_sum;
            for (int l = 0; l < k; l++) {
                this_sum += std::conj(ACCESS(a, l, j, lda)) * std::conj(ACCESS(a, l, i, lda));
            }
            this_sum *= alpha;
            ACCESS(expected, j, i, ldc) = beta * ACCESS(expected, j, i, ldc) + this_sum;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCsyrk(handle, uplo, trans, n, k, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(a), lda,
        reinterpret_cast<oclComplex_t*>(&beta), reinterpret_cast<oclComplex_t*>(c), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_COMPLEX_EQ(expected[i], c[i]);
    }
}
