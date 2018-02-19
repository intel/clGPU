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

TEST(Strmv, Strmv_up_ndiag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int lda = 4;

    const int incx = 2;

    float A[n * lda] = { 1.f, 0.f, 0.f, -1.f,
                         2.f, 4.f, 0.f, -1.f,
                         3.f, 5.f, 6.f, -1.f };
    float x[n * incx] = { 1.f, 0.f, 1.5f, -1.f, 4.f, 0.f };

    float ref_x[n * incx] = { 16.f, 0.f, 26.f, -1.f, 24.f, 0.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_up_diag_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    float A[n * lda] = { 1.f, 4.f, 6.f, 0.f,
                         2.f, 5.f, 7.f, 9.f,
                         3.f, 0.f, 8.f, 5.f,
                         0.f, 0.f, 3.f, 5.f };
    float x[n * incx] = { 1.f, 1.f, 2.f, 0.f };

    float ref_x[n * incx] = { 9.f, 1.f, 2.f, 0.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_low_diag_3x3)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 3;
    const int lda = 4;

    const int incx = 1;

    float A[n * lda] = { 1.9f, 0.5f, 0.7f, -1.f,
                         1.6f, 4.f, 14.25f, 0.f,
                         3.f, 5.25f, 6.f, 1.f };
    float x[n * incx] = { 1.9f, 0.7f, 1.35f };

    float ref_x[n * incx] = { 1.9f, 1.65f, 12.655f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_low_ndiag_2x2)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 2;
    const int lda = 2;

    const int incx = 1;

    float A[n * lda] = { 1.4f, 37.f,
                         1.7f, 1.3f };
    float x[n * incx] = { 1.7f, 1.5f };

    float ref_x[n * incx] = {2.38f, 64.85f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_low_ndiag_trans_3x3)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 3;
    const int lda = 5;

    const int incx = 2;

    float A[n * lda] = { 6.f, 5.f, 3.f, 5.f, 3.f,
                         0.f, 4.f, 2.f, 4.f, 2.f,
                         0.f, 0.f, 1.f, 0.f, 1.f };
    float x[n * incx] = { 1.f, .5f, 1.f, 1.f, .75f, 1.f };

    float ref_x[n * incx] = { 13.25f, 0.5f, 5.5f, 1.f, 0.75f, 1.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_up_ndiag_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    float A[n * lda] = { 1.f, 2.f, 3.f, 5.f,
                         4.f, 5.f, 0.f, 1.f,
                         6.f , 1.f, 23.f, 3.f,
                         0.5f, 4.f, 5.f, 5.f };
    float x[n * incx] = { 1.5f, 2.f, 3.f, 0.f };

    float ref_x[n * incx] = { 27.5f, 13.f, 69.f, 0.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_up_ndiag_trans_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    float A[n * lda] = { 1.f, 2.f, 3.f, 5.f,
                        4.f, 5.f, 0.f, 1.f,
                        6.f , 1.f, 23.f, 3.f,
                        0.5f, 4.f, 5.f, 5.f };
    float x[n * incx] = { 1.5f, 2.f, 3.f, 0.f };

    float ref_x[n * incx] = { 1.5f, 16.f, 80.f, 23.75f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_low_trans_3x3)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 3;
    const int lda = 3;

    const int incx = 1;

    /*
    * | 2    0    0 |   | 1 |   | 2 |
    * | 1    2    0 | * | 2 | = | 5 |
    * | 1    1    2 |   | 3 |   | 9 |
    *        A            x     ref_x
    */
    float A[n * lda] = { 2.f, 1.f, 1.f,
                        0.f, 2.f, 1.f,
                        0.f , 0.f, 2.f };

    float x[n * incx] = { 1.f, 2.f, 3.f };

    float ref_x[n * incx] = { 7.f, 7.f, 6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_low_ndiag_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    float A[n * lda] = { 1.f, 4.f, 6.f, 0.f,
                        2.f, 5.f, 7.f, 9.f,
                        3.f, 0.f, 8.f, 5.f,
                        0.f, 0.f, 3.f, 5.f };
    float x[n * incx] = { 1.f, 1.f, 2.f, 0.f };

    float ref_x[n * incx] = { 1.f, 9.f, 29.f, 19.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}

TEST(Strmv, Strmv_low_trans_ndiag_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    float A[n * lda] = { 1.f, 4.f, 6.f, 0.f,
        2.f, 5.f, 7.f, 9.f,
        3.f, 0.f, 8.f, 5.f,
        0.f, 0.f, 3.f, 5.f };
    float x[n * incx] = { 1.f, 1.f, 2.f, 0.f };

    float ref_x[n * incx] = { 17.f, 19.f, 16.f, 0.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_x[i * incx], x[i * incx]);
}
