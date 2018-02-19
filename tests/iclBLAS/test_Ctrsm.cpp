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

#define EPS 4.e-4f

#define EXPECT_COMPLEX_NEAR(expected, result) \
{\
    if(expected.real() != 0)\
        EXPECT_NEAR(1.f, result.real()/expected.real(), EPS);\
    else\
        EXPECT_NEAR(0.f, result.real(), EPS);\
    \
    if(expected.imag() != 0)\
        EXPECT_NEAR(1.f, result.imag()/expected.imag(), EPS);\
    else\
        EXPECT_NEAR(0.f, result.imag(), EPS);\
}

TEST(Ctrsm, 3x3_left_up_ntrans_ndiag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    std::complex<float> alpha(1.1f, 2.3f);

    std::complex<float> a[lda*m] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = m - 1; i >= 0; i--) {
        for (int j = 0; j<n; j++) {
            std::complex<float> this_x = ACCESS(expected, i, j, ldb);
            this_x /= ACCESS(a, i, i, lda);
            for (int k = 0; k<i; k++) {
                ACCESS(expected, k, j, ldb) -= ACCESS(a, k, i, lda)*this_x;
            }
            ACCESS(expected, i, j, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 4x3_left_up_trans_diag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 4;
    const int n = 3;
    const int lda = 4;
    const int ldb = 4;
    std::complex<float> alpha(-1.1f, 2.3f);

    std::complex<float> a[lda*m] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },{ -5.f, -6.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -7.f, -8.f },{ -9.f, -10.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -11.f, -12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f }, { 19.f, 20.f } };
    std::complex<float> b[ldb*n] = { { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },
                                     { 29.f, 30.f },{ 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f },
                                     { 37.f, 38.f },{ 39.f, 40.f },{ 41.f, 42.f },{ 43.f, 44.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j<n; j++) {
            std::complex<float> this_x = ACCESS(expected, i, j, ldb);
            for (int k = i+1; k<m; k++) {
                ACCESS(expected, k, j, ldb) -= ACCESS(a, i, k, lda)*this_x;
            }
            ACCESS(expected, i, j, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x4_left_up_conj_ndiag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 4;
    const int lda = 3;
    const int ldb = 3;
    std::complex<float> alpha(-1.1f, -2.3f);

    std::complex<float> a[lda*m] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f },
                                     { 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j<n; j++) {
            std::complex<float> this_x = ACCESS(expected, i, j, ldb);
            this_x /= std::conj(ACCESS(a, i, i, lda));
            for (int k = i + 1; k<m; k++) {
                ACCESS(expected, k, j, ldb) -= std::conj(ACCESS(a, i, k, lda))*this_x;
            }
            ACCESS(expected, i, j, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x3_left_low_ntrans_diag_lda4) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 4;
    const int ldb = 3;
    std::complex<float> alpha(1.1f, -2.3f);

    std::complex<float> a[lda*m] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ -11.f, -12.f },
                                     { -1.f, -2.f },{ 7.f, 8.f },{ 9.f, 10.f },{ -11.f, -12.f },
                                     { -3.f, -4.f },{ -5.f, -6.f },{ 11.f, 12.f },{ -11.f, -12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j<n; j++) {
            std::complex<float> this_x = ACCESS(expected, i, j, ldb);
            for (int k = i + 1; k<m; k++) {
                ACCESS(expected, k, j, ldb) -= ACCESS(a, k, i, lda)*this_x;
            }
            ACCESS(expected, i, j, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x4_left_low_trans_ndiag_ldb4) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 4;
    const int lda = 3;
    const int ldb = 4;
    std::complex<float> alpha(2.1f, 1.3f);

    std::complex<float> a[lda*m] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { -1.f, -2.f },{ 7.f, 8.f },{ 9.f, 10.f },
                                     { -3.f, -4.f },{ -5.f, -6.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -1.f, -2.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },{ -3.f, -4.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f },{ -5.f, -6.f },
                                     { 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f },{ -7.f, -8.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = m-1; i >= 0; i--) {
        for (int j = 0; j<n; j++) {
            std::complex<float> this_x = ACCESS(expected, i, j, ldb);
            this_x /= ACCESS(a, i, i, lda);
            for (int k = 0; k<i; k++) {
                ACCESS(expected, k, j, ldb) -= ACCESS(a, i, k, lda)*this_x;
            }
            ACCESS(expected, i, j, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 4x3_left_low_conj_diag_lds5) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 4;
    const int n = 3;
    const int lda = 5;
    const int ldb = 5;
    std::complex<float> alpha(-2.1f, 1.3f);

    std::complex<float> a[lda*m] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ -1.f, -2.f },
                                     { -3.f, -4.f },{ 9.f, 10.f },{ 11.f, 12.f },{ 13.f, 14.f },{ -5.f, -6.f },
                                     { -7.f, -8.f },{ -9.f, -10.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -11.f, -12.f },
                                     { -13.f, -14.f },{ -15.f, -16.f },{ -17.f, -18.f },{ 19.f, 20.f },{ -19.f, -20.f } };
    std::complex<float> b[ldb*n] = { { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },{ -1.f, -2.f },
                                     { 29.f, 30.f },{ 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f },{ -3.f, -4.f },
                                     { 37.f, 38.f },{ 39.f, 40.f },{ 41.f, 42.f },{ 43.f, 44.f },{ -5.f, -6.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = m - 1; i >= 0; i--) {
        for (int j = 0; j<n; j++) {
            std::complex<float> this_x = ACCESS(expected, i, j, ldb);
            for (int k = 0; k<i; k++) {
                ACCESS(expected, k, j, ldb) -= std::conj(ACCESS(a, i, k, lda))*this_x;
            }
            ACCESS(expected, i, j, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x3_right_up_ntrans_ndiag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    std::complex<float> alpha(-2.1f, -1.3f);

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j<m; j++) {
            std::complex<float> this_x = ACCESS(expected, j, i, ldb);
            this_x /= ACCESS(a, i, i, lda);
            for (int k = i+1; k<n; k++) {
                ACCESS(expected, j, k, ldb) -= ACCESS(a, i, k, lda)*this_x;
            }
            ACCESS(expected, j, i, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 4x3_right_up_trans_ndiag_ldb5) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 4;
    const int n = 3;
    const int lda = 3;
    const int ldb = 4;
    std::complex<float> alpha(2.1f, -1.3f);

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -5.f, -6.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f },
                                     { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },
                                     { 29.f, 30.f },{ 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = n-1; i >= 0; i--) {
        for (int j = 0; j<m; j++) {
            std::complex<float> this_x = ACCESS(expected, j, i, ldb);
            this_x /= ACCESS(a, i, i, lda);
            for (int k = 0; k<i; k++) {
                ACCESS(expected, j, k, ldb) -= ACCESS(a, k, i, lda)*this_x;
            }
            ACCESS(expected, j, i, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x4_right_up_conj_diag_lda5) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 3;
    const int n = 4;
    const int lda = 4;
    const int ldb = 3;
    std::complex<float> alpha(1.3f, 2.3f);

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ -1.f, -2.f },{ -3.f, -4.f },{ -5.f, -6.f },
                                     { 3.f, 4.f },{ 5.f, 6.f },{ -7.f, -8.f },{ -9.f, -10.f },
                                     { 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },{ -11.f, -12.f },
                                     { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f } };
    std::complex<float> b[ldb*n] = { { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },
                                     { 27.f, 28.f },{ 29.f, 30.f },{ 31.f, 32.f },
                                     { 33.f, 34.f },{ 35.f, 36.f },{ 37.f, 38.f },
                                     { 39.f, 40.f },{ 41.f, 42.f },{ 43.f, 44.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = n - 1; i >= 0; i--) {
        for (int j = 0; j<m; j++) {
            std::complex<float> this_x = ACCESS(expected, j, i, ldb);
            for (int k = 0; k<i; k++) {
                ACCESS(expected, j, k, ldb) -= std::conj(ACCESS(a, k, i, lda))*this_x;
            }
            ACCESS(expected, j, i, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x3_right_low_ntrans_diag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    std::complex<float> alpha(-1.3f, -2.3f);

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { -1.f, -2.f },{ 7.f, 8.f },{ 9.f, 10.f },
                                     { -3.f, -4.f },{ -5.f, -6.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                     { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },
                                     { 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = n-1; i >= 0; i--) {
        for (int j = 0; j<m; j++) {
            std::complex<float> this_x = ACCESS(expected, j, i, ldb);
            for (int k = 0; k<i; k++) {
                ACCESS(expected, j, k, ldb) -= ACCESS(a, i, k, lda)*this_x;
            }
            ACCESS(expected, j, i, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 3x4_right_low_trans_ndiag_lds5) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 4;
    const int lda = 5;
    const int ldb = 5;
    std::complex<float> alpha(1.3f, -2.3f);

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ -1.f, -2.f },
                                     { -3.f, -4.f },{ 9.f, 10.f },{ 11.f, 12.f },{ 13.f, 14.f },{ -5.f, -6.f },
                                     { -7.f, -8.f },{ -9.f, -10.f },{ 15.f, 16.f },{ 17.f, 18.f },{ -11.f, -12.f },
                                     { -13.f, -14.f },{ -15.f, -16.f },{ -17.f, -18.f },{ 19.f, 20.f },{ -19.f, -20.f } };
    std::complex<float> b[ldb*n] = { { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ -1.f, -2.f },{ -3.f, -4.f },
                                     { 27.f, 28.f },{ 29.f, 30.f },{ 31.f, 32.f },{ -5.f, -6.f },{ -7.f, -8.f },
                                     { 33.f, 34.f },{ 35.f, 36.f },{ 37.f, 38.f },{ -9.f, -10.f },{ -11.f, -12.f },
                                     { 39.f, 40.f },{ 41.f, 42.f },{ 43.f, 44.f },{ -13.f, -14.f },{ -15.f, -16.f }, };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j<m; j++) {
            std::complex<float> this_x = ACCESS(expected, j, i, ldb);
            this_x /= ACCESS(a, i, i, lda);
            for (int k = i + 1; k<n; k++) {
                ACCESS(expected, j, k, ldb) -= ACCESS(a, k, i, lda)*this_x;
            }
            ACCESS(expected, j, i, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}

TEST(Ctrsm, 4x3_right_low_conj_ndiag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 4;
    const int n = 3;
    const int lda = 3;
    const int ldb = 4;
    std::complex<float> alpha(1.3f, 2.3f);

    std::complex<float> a[lda*n] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },
                                     { -1.f, -2.f },{ 7.f, 8.f },{ 9.f, 10.f },
                                     { -3.f, -4.f },{ -5.f, -6.f },{ 11.f, 12.f } };
    std::complex<float> b[ldb*n] = { { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f },
                                     { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },
                                     { 29.f, 30.f },{ 31.f, 32.f },{ 33.f, 34.f },{ 35.f, 36.f } };

    std::complex<float> expected[ldb*n];
    for (int i = 0; i < ldb*n; i++) expected[i] = b[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j<m; j++) {
            std::complex<float> this_x = ACCESS(expected, j, i, ldb);
            this_x /= std::conj(ACCESS(a, i, i, lda));
            for (int k = i + 1; k<n; k++) {
                ACCESS(expected, j, k, ldb) -= std::conj(ACCESS(a, k, i, lda))*this_x;
            }
            ACCESS(expected, j, i, ldb) = this_x*alpha;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<oclComplex_t*>(&alpha),
        reinterpret_cast<oclComplex_t*>(a), lda, reinterpret_cast<oclComplex_t*>(b), ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        EXPECT_COMPLEX_NEAR(expected[i], b[i]);
    }
}
