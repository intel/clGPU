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

TEST(Strmm, 3x3_Left_Lowerr_N_NonUnit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f};
    
    float expected[9] = { 60, 132, 204,
            2, 162, 252,
            3, 6, 300, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb); // Using b as c, to use this function like in fortran
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++) 
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Left_Upper_N_Unit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 1.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 30, 4, 7,
        36, 61, 8.f,
        42, 72, 78 };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x2_Left_Upper_N_Unit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 2;
    const int m = 3;
    float alpha = 1.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 2.f, 8.f, 14.f,
        4.f, 10.f, 16.f };

    float expected[9] = { 60.f, 8.f, 14.f,
        72.f, 162.f, 16.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 6; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Right_Lower_N_Unit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 60, 132, 204,
                          2, 122, 188, 
                          3, 6, 156};

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Right_Lower_N_NonUnit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 60, 132, 204,
        2, 162, 252,
        3, 6, 300 };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Left_Lowerr_T_NonUnit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 132, 156, 180,
        2, 186, 216,
        3, 6, 252, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Left_Lowerr_N_Unit) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 60, 100, 92,
        2, 122, 124,
        3, 6, 156, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Right_Upper_N_NonUnit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 60.f, 4.f, 7.f,
        72.f, 162.f, 8,
        84.f, 192.f, 300.f, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Right_Upper_N_Unit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 2.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 60.f, 4.f, 7.f,
        56, 122, 8,
        36, 96, 156, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Right_Upper_T_NONUnit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 1.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 14, 4, 7,
        32, 77, 8,
        50, 122, 194, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}

TEST(Strmm, 3x3_Right_Upper_T_Unit) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto transa = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int m = 3;
    float alpha = 1.f;
    const int lda = 3;
    const int ldb = 3;

    float a[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float b[9] = { 1.f, 4.f, 7.f,
        2.f, 5.f, 8.f,
        3.f, 6.f, 9.f };

    float expected[9] = { 14, 4, 7,
        24, 57, 8,
        26, 74, 122, };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmm(handle, side, uplo, transa, diag, m, n, &alpha, a, lda, b, ldb, b, ldb);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 9; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], b[i]);
    }
}
