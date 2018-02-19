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
#include <cmath>

#define ADAPTIVE_FLOAT_EQ(expected, result) if (std::abs((double)(expected)) < 1.) EXPECT_NEAR((expected), (result), 4.*FLT_EPSILON); else EXPECT_FLOAT_EQ((expected), (result))
/* Check if absolute value of expected is less than 1, because ULP(1) is equal FLT_EPSILON.
   Below this threshold instead of checking if values are within 4 ULPs, we check if they are within 4 FLT_EPSILONs, that is machine epsilon for floats */

TEST(Strsm, 3x3_left_up_ntrans_ndiag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    float alpha = 1.1f;

    float A[lda*m] = { 1.f, -1.f, -2.f,
                       2.f, 3.f, -3.f,
                       4.f, 5.f, 6.f };
    float B[ldb*n] = { 7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f,
                       13.f, 14.f, 15.f };
    float expected[ldb*n] = { 1.1f*2.f/3.f, 1.1f*1.f/6.f, 1.1f*3.f/2.f,
                              1.1f*4.f/3.f, 1.1f*1.f/3.f, 1.1f*2.f, 
                              1.1f*2.f, 1.1f*1.f/2.f, 1.1f*5.f/2.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n,&alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 3x3_right_up_ntrans_ndiag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    float alpha = 1.1f;

    float A[lda*n] = { 1.f, -1.f, -2.f,
        2.f, 3.f, -3.f,
        4.f, 5.f, 6.f };
    float B[ldb*n] = { 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f,
        13.f, 14.f, 15.f };
    float expected[ldb*n] = { 1.1f*7.f, 1.1f*8.f, 1.1f*9.f,
                             -1.1f*4.f / 3.f, -1.1f*5.f / 3.f, -1.1f*6.f/3.f, 
                             -1.1f*25.f / 18.f, -1.1f*29.f / 18.f, -1.1f*33.f / 18.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 3x3_left_low_ntrans_ndiag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 4;
    const int ldb = 3;
    float alpha = -1.1f;

    float A[lda*m] = { 1.f, 2.f, 3.f, -1.f,
                      -2.f, 4.f, 5.f, -3.f,
                      -4.f, -5.f, 6.f, -7.f };
    float B[ldb*n] = { 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f,
        13.f, 14.f, 15.f };
    float expected[ldb*n] = { -1.1f*7.f, 1.1f*3.f/2.f, 1.1f*9.f/12.f ,
                              -1.1f*10.f, 1.1f*9.f/4.f, 1.1f*27.f/24.f,
                              -1.1f*13.f, 1.1f*3.f, 1.1f*9.f/6.f};

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 3x3_right_low_ntrans_ndiag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 4;
    float alpha = 0.1f;

    float A[lda*n] = { 1.f, 2.f, 3.f,
                      -1.f, 4.f, 5.f,
        -             -2.f, -3.f, 6.f };
    float B[ldb*n] = { 7.f, 8.f, 9.f, -1.f,
                      10.f, 11.f, 12.f, -2.f,
                      13.f, 14.f, 15.f, -3.f };
    float expected[ldb*n] = { 0.1f*11.f/12.f, 0.1f*4.f/3.f, 0.1f*7.f/4.f, -1.f,
                              -0.1f*5.f/24.f, -0.1f*1.f/6.f, -0.1f*1.f/8.f, -2.f,
                              0.1f*13.f/6.f, 0.1f*14.f/6.f, 0.1f*15.f/6.f, -3.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 4x3_left_up_trans_ndiag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 4;
    const int n = 3;
    const int lda = 4;
    const int ldb = 4;
    float alpha = 1.3f;

    float A[lda*m] = { 1.f, -1.f, -2.f, -3.f,
                       2.f, 3.f, -4.f, -5.f,
                       4.f, 5.f, 6.f, -6.f,
                       7.f, 8.f, 9.f, 10.f };
    float B[ldb*n] = { 11.f, 12.f, 13.f, 14.f,
                       15.f, 16.f, 17.f, 18.f,
                       19.f, 20.f, 21.f, 22.f };
    float expected[ldb*n] = { 1.3f*11.f, -1.3f*10.f/3.f, -1.3f*43.f/18.f, -1.3f*89.f/60.f,
                              1.3f*15.f, -1.3f*14.f/3.f, -1.3f*59.f/18.f, -1.3f*121.f/60.f,
                              1.3f*19.f, -1.3f*18.f/3.f, -1.3f*75.f/18.f, -1.3f*153.f/60.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 3x4_right_up_trans_ndiag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;

    const int m = 3;
    const int n = 4;
    const int lda = 4;
    const int ldb = 3;
    float alpha = 1.f;

    float A[lda*n] = { 1.f, -1.f, -2.f, -3.f,
                       2.f, 3.f, -4.f, -5.f,
                       4.f, 5.f, 6.f, -6.f,
                       7.f, 8.f, 9.f, 10.f };
    float B[ldb*n] = { 11.f, 12.f, 13.f,
                       14.f, 15.f, 16.f,
                       17.f, 18.f, 19.f,
                       20.f, 21.f, 22.f };
    float expected[ldb*n] = { -14.f/9.f, -14.f/10.f, -56.f/45.f,
         -7.f/18.f, -21.f/60.f, -14.f/45.f,
        -1.f/6.f, -9.f/60.f, -8.f/60.f,
        20.f/10.f, 21.f/10.f, 22.f/10.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 3x3_left_low_trans_diag) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 3;
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    float alpha = 0.1f;

    float A[lda*m] = { 1.f, 2.f, 3.f,
        -2.f, 4.f, 5.f,
        -4.f, -5.f, 6.f };
    float B[ldb*n] = { 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f,
        13.f, 14.f, 15.f };
    float expected[ldb*n] = { 0.1f*54.f, -0.1f*37.f, 0.1f*9.f,
                              0.1f*72.f, -0.1f*49.f, 0.1f*12.f,
                              0.1f*90.f, -0.1f*61.f, 0.1f*15.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}

TEST(Strsm, 5x2_right_low_trans_diag) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_UNIT;

    const int m = 5;
    const int n = 2;
    const int lda = 3;
    const int ldb = 6;
    float alpha = 0.1f;

    float A[lda*n] = { 1.f, 2.f, -1.f,
                       -2.f, 3.f, -3.f};
    float B[ldb*n] = { 4.f, 5.f, 6.f, 7.f, 8.f, -1.f,
                       10.f, 11.f, 12.f, 13.f, 14.f, -2.f };
    float expected[ldb*n] = { 0.1f*4.f, 0.1f*5.f, 0.1f*6.f, 0.1f*7.f, 0.1f*8.f, -1.f,
                              0.1f*2.f, 0.1f*1.f, 0.f, -0.1f*1.f, -0.1f*2.f, -2.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldb*n; i++) {
        ADAPTIVE_FLOAT_EQ(expected[i], B[i]);
    }
}
