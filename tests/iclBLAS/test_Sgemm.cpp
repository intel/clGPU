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

TEST(Sgemm, ntransAB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    float alpha = 1; float beta = 1;
    const int m = 3; const int n = 3; const int k = 3;

    const int lda = 3; const int ldb = 3; const int ldc = 3;

    float A[m * lda] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float B[k * ldb] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float C[m * ldc] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float ref_C[m * ldc] = { 7.f, 14.f, 21.f,
                            7.f, 14.f, 21.f,
                            7.f, 14.f, 21.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
            EXPECT_FLOAT_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
}

TEST(Sgemm, transAB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_T;
    iclblasOperation_t transb = ICLBLAS_OP_T;

    float alpha = 1; float beta = 1;
    const int m = 3; const int n = 3; const int k = 3;

    const int lda = 3; const int ldb = 3; const int ldc = 3;

    float A[lda * m] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float B[ldb * k] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float C[ldc * n] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float ref_C[ldc * n] = { 7.f, 8.f, 9.f,
                            13.f, 14.f, 15.f,
                            19.f, 20.f, 21.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
            EXPECT_FLOAT_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
}

TEST(Sgemm, ntransA_transB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_T;

    float alpha = 1; float beta = 1;
    const int m = 3; const int n = 3; const int k = 3;

    const int lda = 3; const int ldb = 3; const int ldc = 3;

    float A[lda * k] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float B[ldb * k] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float C[ldc * n] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float ref_C[ldc * n] = { 4.f, 8.f, 12.f,
                            7.f, 14.f, 21.f,
                            10.f, 20.f, 30.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
            EXPECT_FLOAT_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
}

TEST(Sgemm, transA_ntransB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_T;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    float alpha = 1; float beta = 1;
    const int m = 3; const int n = 3; const int k = 3;

    const int lda = 3; const int ldb = 3; const int ldc = 3;

    float A[lda * m] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float B[ldb * n] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float C[ldc * n] = { 1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f,
                        1.f, 2.f, 3.f };

    float ref_C[ldc * n] = { 15.f, 16.f, 17.f,
                            15.f, 16.f, 17.f,
                            15.f, 16.f, 17.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
            EXPECT_FLOAT_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
}

TEST(Sgemm, ntransAB_ldx_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    float alpha = 1; float beta = 1;
    const int m = 3; const int n = 3; const int k = 3;

    const int lda = 4; const int ldb = 5; const int ldc = 4;

    float A[lda * k] = { 1.f, 2.f, 3.f, -1.f,
                        1.f, 2.f, 3.f, -1.f,
                        1.f, 2.f, 3.f, -1.f, };

    float B[ldb * n] = { 1.f, 2.f, 3.f, -1.f, -1.f,
                        1.f, 2.f, 3.f, -1.f, -1.f,
                        1.f, 2.f, 3.f, -1.f, -1.f };

    float C[ldc * n] = { 1.f, 2.f, 3.f, -1.f,
                        1.f, 2.f, 3.f, -1.f,
                        1.f, 2.f, 3.f, -1.f };

    float ref_C[ldc * n] = { 7.f, 14.f, 21.f, -1.f,
                            7.f, 14.f, 21.f, -1.f,
                            7.f, 14.f, 21.f, -1.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            EXPECT_FLOAT_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
}

TEST(Sgemm, ntransAB_optim)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    float alpha = 1; float beta = 1;
    const int m = 64; const int n = 64; const int k = 64;
    const int lda = 64; const int ldb = 64; const int ldc = 64;

    float A[lda * k];
    float B[ldb * n];
    float C[ldc * n];
    float ref_C[ldc * n];

    for (int i = 0; i < lda * k; i++)
        A[i] = i * 1.25f;

    for (int i = 0; i < ldb * n; i++)
        B[i] = i * 1.5f;

    for (int i = 0; i < ldc * n; i++)
        C[i] = i * 1.75f;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float value = 0.f;

            for (int l = 0; l < k; l++)
            {
                value += A[l * lda + i] * B[j * ldb + l];
            }

            //ref_C[j * ldc + i] = value;
            ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            EXPECT_FLOAT_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
}
