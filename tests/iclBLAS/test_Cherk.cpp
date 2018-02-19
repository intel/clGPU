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

TEST(Cherk, up_ntrans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3; const int k = 3;
    const int lda = 3; const int ldc = 3;
    float alpha = 1.f; float beta = 1.f;

    std::complex<float> A[lda * k] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f } };
    std::complex<float> C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 2.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 3.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += A[i * lda + l] * std::conj(A[i * lda + l]);
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}

TEST(Cherk, up_ntrans_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 4; const int k = 4;
    const int lda = 4; const int ldc = 4;
    float alpha = 1.f; float beta = 1.f;

    std::complex<float> A[lda * k] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },
                                        { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },
                                        { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },
                                        { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f } };
    std::complex<float> C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f }, { 4.f, 0.f },
                                        { 2.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f }, { 4.f, 0.f },
                                        { 3.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f }, { 4.f, 0.f },
                                        { 4.f, 0.f }, { 4.f, 0.f }, { 4.f, 0.f }, { 5.f, 0.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += A[i * lda + l] * std::conj(A[i * lda + l]);
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}

TEST(Cherk, low_ntrans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3; const int k = 3;
    const int lda = 3; const int ldc = 3;
    float alpha = 1.f; float beta = 1.f;

    std::complex<float> A[lda * k] = { { 1.f, 0.f },{ 2.f, 1.f },{ 5.f, -2.f },
                                        { 2.f, -1.f },{ 2.f, -15.f },{ 3.f, 0.f },
                                        { 3.f, -2.f },{ 4.f, 0.f },{ 3.f, 7.f } };
    std::complex<float> C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 2.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 3.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += (A[i * lda + l] * std::conj(A[i * lda + l]));
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}

TEST(Cherk, up_trans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasOperation_t trans = ICLBLAS_OP_C;

    const int n = 3; const int k = 3;
    const int lda = 3; const int ldc = 3;
    float alpha = 1.f; float beta = 1.f;

    std::complex<float> A[lda * n] = { { 1.f, 0.f },{ 2.f, 1.f },{ 5.f, -2.f },
                                        { 2.f, -1.f },{ 2.f, -15.f },{ 3.f, 0.f },
                                        { 3.f, -2.f },{ 4.f, 0.f },{ 3.f, 7.f } };
    std::complex<float> C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 2.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },
                                        { 3.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += std::conj(A[l * lda + i]) * A[l * lda + i];
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}

TEST(Cherk, low_trans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasOperation_t trans = ICLBLAS_OP_C;

    const int n = 3; const int k = 3;
    const int lda = 3; const int ldc = 3;
    float alpha = 1.f; float beta = 1.f;

    std::complex<float> A[lda * n] = { { 1.f, 0.f },{ 2.f, 1.f },{ 5.f, -2.f },
                                        { 2.f, -1.f },{ 2.f, -15.f },{ 3.f, 0.f },
                                        { 3.f, -2.f },{ 4.f, 0.f },{ 3.f, 7.f } };
    std::complex<float> C[ldc * n] = { { 1.f, -1.f },{ 2.f, 2.f },{ 3.f, 7.f },
                                        { 2.f, -2.f },{ 2.f, -15.f },{ 3.f, -2.f },
                                        { 3.f, -7.f },{ 3.f, 2.f },{ 4.f, 6.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += (std::conj(A[l * lda + i]) * A[l * lda + i]);
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}

TEST(Cherk, up_ntrans_4x4_lda)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 4; const int k = 4;
    const int lda = 6; const int ldc = 4;
    float alpha = 1.35f; float beta = 2.4f;

    std::complex<float> A[lda * k] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },{ -1.f,-1.f },{ -1.f,-1.f },
                                    { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },{ -1.f,-1.f },{ -1.f,-1.f },
                                    { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },{ -1.f,-1.f },{ -1.f,-1.f },
                                    { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 3.f, 0.f },{ -1.f,-1.f },{ -1.f,-1.f } };
    std::complex<float> C[ldc * n] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f },
                                    { 2.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f },
                                    { 3.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f },{ 4.f, 0.f },
                                    { 4.f, 0.f },{ 4.f, 0.f },{ 4.f, 0.f },{ 5.f, 0.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += A[i * lda + l] * std::conj(A[i * lda + l]);
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}

TEST(Cherk, low_trans_3x3_lda)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasOperation_t trans = ICLBLAS_OP_C;

    const int n = 3; const int k = 3;
    const int lda = 5; const int ldc = 4;
    float alpha = 0.5f; float beta = 1.7f;

    std::complex<float> A[lda * n] = { { 1.f, 0.f },{ 2.f, 1.f },{ 5.f, -2.f },{ -1.f,-1.f },{ -1.f,-1.f },
                                        { 2.f, -1.f },{ 2.f, -15.f },{ 3.f, 0.f },{ -1.f,-1.f },{ -1.f,-1.f },
                                        { 3.f, -2.f },{ 4.f, 0.f },{ 3.f, 7.f },{ -1.f,-1.f },{ -1.f,-1.f } };
    std::complex<float> C[ldc * n] = { { 1.f, -1.f },{ 2.f, 2.f },{ 3.f, 7.f },{ -1.f,-1.f },
                                        { 2.f, -2.f },{ 2.f, -15.f },{ 3.f, -2.f },{ -1.f,-1.f },
                                        { 3.f, -7.f },{ 3.f, 2.f },{ 4.f, 6.f },{ -1.f,-1.f } };
    std::complex<float> ref_C[ldc * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ref_C[i * ldc + j] = C[i * ldc + j];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::complex<float> value = { 0.f, 0.f };

            for (int l = 0; l < k; l++)
            {
                value += (std::conj(A[l * lda + i]) * A[l * lda + i]);
            }

            if (j * ldc == i * ldc)
                C[j * ldc + i].imag(0.f);

            if (uplo == ICLBLAS_FILL_MODE_UPPER && j * ldc >= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }

            if (uplo == ICLBLAS_FILL_MODE_LOWER && j * ldc <= i * ldc)
            {
                ref_C[j * ldc + i] = alpha * value + beta * C[j * ldc + i];
            }
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCherk(handle, uplo, trans, n, k, &alpha, reinterpret_cast<oclComplex_t*>(A), lda, &beta, reinterpret_cast<oclComplex_t*>(C), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].real(), C[j * ldc + i].real());
            EXPECT_FLOAT_EQ(ref_C[j * ldc + i].imag(), C[j * ldc + i].imag());
        }
}
