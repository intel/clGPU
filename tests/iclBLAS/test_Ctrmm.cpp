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
#include <gtest_utils.hpp>

TEST(Ctrmm, 3x3_Left_Upper_N_NonUnit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { {0, 60},{ 4, 4 },{ 7, 7 },
    { 0, 72 },{ 0, 162 },{ 8, 8 },
    { 0, 84 },{ 0, 192 },{ 0, 300 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Left_Upper_N_Unit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { 1, 59 },{ 4, 4 },{ 7, 7 },
    { 2, 70 },{ 5, 117 },{ 8, 8 },
    { 3, 81 },{ 6, 138 },{ 9, 147 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Left_Lower_N_Unit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { 1, 59 },{ 4, 96 },{ 7, 85 },
    { 2, 2 },{ 5, 117 },{ 8, 116 },
    { 3, 3 },{ 6, 6 },{ 9, 147 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Left_Upper_T_NonUnit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { 0, 132 },{ 4, 4 },{ 7, 7 },
    { 0, 156 },{ 0, 186 },{ 8, 8 },
    { 0, 180 },{ 0, 216 },{ 0, 252 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Left_Lower_T_NonUnit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { 0, 132 },{ 0, 156 },{ 0, 180 },
    { 2, 2 },{ 0, 186 },{ 0, 216 },
    { 3, 3 },{ 6, 6 },{ 0, 252 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Right_Upper_N_NonUnit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 4,6 },
    { 3,3 },{ 8,2 },{ 8,8 },
    { 2,2 },{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { -6, 48 },{ 4, 4 },{ 7, 7 },
    { 12, 74 },{ 30, 170 },{ 8, 8 },
    { 0, 82 },{ 0, 184 },{ 0, 286 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Right_Upper_T_NonUnit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3},{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { 0, 28 },{ 4, 4 },{ 7, 7 },
    { 0, 64 },{ 0, 154 },{ 8, 8 },
    { 0, 100 },{ 0, 244 },{ 0, 388 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}

TEST(Ctrmm, 3x3_Right_Lower_T_NonUnit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    oclComplex_t alpha = { 1, 0 };
    const int lda = 3;
    const int ldb = 3;
    const int ldc = 3;

    oclComplex_t a[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3},{ 6,6 },{ 9,9 } };

    oclComplex_t b[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t c[9] = { { 1,1 },{ 4,4 },{ 7,7 },
    { 2,2 },{ 5,5 },{ 8,8 },
    { 3,3 },{ 6,6 },{ 9,9 } };

    oclComplex_t expected[9] = { { 0, 28 },{ 0, 64 },{ 0, 100 },
    { 2, 2 },{ 0, 154 },{ 0, 244 },
    { 3, 3 },{ 6, 6 },{ 0, 388 }, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, expected, c);
}
