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

TEST(Cgemm, ntransAB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    oclComplex_t alpha = { 1.f, 0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[m * lda] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t B[k * ldb] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t C[m * ldc] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t ref_C[m * ldc] = { {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f},
                                    {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f},
                                    {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, transAB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_T;
    iclblasOperation_t transb = ICLBLAS_OP_T;

    oclComplex_t alpha = { 1.f,0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[lda * m] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t B[ldb * k] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t C[ldc * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t ref_C[ldc * n] = { {7.f, 0.f}, {8.f, 0.f}, {9.f, 0.f},
                                    {13.f, 0.f}, {14.f, 0.f}, {15.f, 0.f},
                                    {19.f, 0.f}, {20.f, 0.f}, {21.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, ntransA_transB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_T;

    oclComplex_t alpha = { 1.f, 0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[lda * k] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t B[ldb * k] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t C[ldc * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t ref_C[ldc * n] = { {4.f, 0.f}, {8.f, 0.f}, {12.f, 0.f},
                                    {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f},
                                    {10.f, 0.f}, {20.f, 0.f}, {30.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, transA_ntransB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_T;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    oclComplex_t alpha = { 1.f, 0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[lda * m] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t B[ldb * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t C[ldc * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f} };

    oclComplex_t ref_C[ldc * n] = { {15.f, 0.f}, {16.f, 0.f}, {17.f, 0.f},
                                    {15.f, 0.f}, {16.f, 0.f}, {17.f, 0.f},
                                    {15.f, 0.f}, {16.f, 0.f}, {17.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, ntransAB_ldx_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    oclComplex_t alpha = { 1.f, 0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 4; const int ldb = 5; const int ldc = 4;

    oclComplex_t A[lda * k] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f} };

    oclComplex_t B[ldb * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f}, {-1.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f}, {-1.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f}, {-1.f, 0.f} };

    oclComplex_t C[ldc * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f},
                                {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {-1.f, 0.f} };

    oclComplex_t ref_C[ldc * n] = { {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f}, {-1.f, 0.f},
                                    {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f}, {-1.f, 0.f},
                                    {7.f, 0.f}, {14.f, 0.f}, {21.f, 0.f}, {-1.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, hermitAB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_C;
    iclblasOperation_t transb = ICLBLAS_OP_C;

    oclComplex_t alpha = { 1.f,0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[lda * m] = { { 1.f, 0.f },{ 2.f, -1.f },{ 3.f, 5.f },
                                { 2.f, 1.f },{ 2.f, 0.f },{ 3.f, 1.f },
                                { 3.f, -5.f },{ 3.f, 1.f },{ 6.f, 0.f } };

    oclComplex_t B[ldb * k] = { { 1.f, 0.f },{ 1.f, 1.f },{ 3.f, 5.f },
                                { 1.f, -1.f },{ 2.f, 0.f },{ 3.f, -2.f },
                                { 3.f, -5.f },{ 3.f, 2.f },{ 3.f, 0.f } };

    oclComplex_t C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f } };

    oclComplex_t ref_C[ldc * n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            oclComplex_t value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                oclComplex_t tmp = std::conj(A[i * lda + l]) * std::conj(B[l * ldb + j]);

                value += tmp;
            }

            oclComplex_t tmp1 = alpha * value;
            oclComplex_t tmp2 = beta * C[j * ldc + i];

            ref_C[j * ldc + i] = tmp1 + tmp2;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, hermitA_ntransB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_C;
    iclblasOperation_t transb = ICLBLAS_OP_N;

    oclComplex_t alpha = { 1.f,0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[lda * m] = { { 1.f, 0.f },{ 2.f, -1.f },{ 3.f, 5.f },
                                { 2.f, 1.f },{ 2.f, 0.f },{ 3.f, 1.f },
                                { 3.f, -5.f },{ 3.f, 1.f },{ 6.f, 0.f } };

    oclComplex_t B[ldb * k] = { { 1.f, 0.f },{ 1.f, 1.f },{ 3.f, 5.f },
                                { 1.f, -1.f },{ 2.f, 0.f },{ 3.f, -2.f },
                                { 3.f, -5.f },{ 3.f, 2.f },{ 3.f, 0.f } };

    oclComplex_t C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f } };

    oclComplex_t ref_C[ldc * n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            oclComplex_t value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                oclComplex_t tmp = std::conj(A[i * lda + l]) * B[j * ldb + l];

                value += tmp;
            }

            oclComplex_t tmp1 = alpha * value;
            oclComplex_t tmp2 = beta * C[j * ldc + i];

            ref_C[j * ldc + i] = tmp1 + tmp2;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}

TEST(Cgemm, ntransA_hermitB_3x3)
{
    iclblasOperation_t transa = ICLBLAS_OP_N;
    iclblasOperation_t transb = ICLBLAS_OP_C;

    oclComplex_t alpha = { 1.f,0.f }; oclComplex_t beta = { 1.f, 0.f };

    const int m = 3; const int n = 3; const int k = 3;
    const int lda = 3; const int ldb = 3; const int ldc = 3;

    oclComplex_t A[lda * m] = { { 1.f, 0.f },{ 2.f, -1.f },{ 3.f, 5.f },
                                { 2.f, 1.f },{ 2.f, 0.f },{ 3.f, 1.f },
                                { 3.f, -5.f },{ 3.f, 1.f },{ 6.f, 0.f } };

    oclComplex_t B[ldb * k] = { { 1.f, 0.f },{ 1.f, 1.f },{ 3.f, 5.f },
                                { 1.f, -1.f },{ 2.f, 0.f },{ 3.f, -2.f },
                                { 3.f, -5.f },{ 3.f, 2.f },{ 3.f, 0.f } };

    oclComplex_t C[ldc * n] = { { 1.f, -1.f },{ 2.f, 5.f },{ 3.f, -8.f },
                                { 1.f, 2.f },{ 2.f, -18.f },{ 3.f, 1.f },
                                { 1.f, 15.f },{ 2.f, 7.f },{ 3.f, -3.f } };

    oclComplex_t ref_C[ldc * n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            oclComplex_t value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                oclComplex_t tmp = A[l * lda + i] * std::conj(B[l * ldb + j]);

                value += tmp;
            }

            oclComplex_t tmp1 = alpha * value;
            oclComplex_t tmp2 = beta * C[j * ldc + i];

            ref_C[j * ldc + i] = tmp1 + tmp2;
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < ldc; j++)
        {
            EXPECT_COMPLEX_EQ(C[i * ldc + j], ref_C[i * ldc + j]);
        }
}
