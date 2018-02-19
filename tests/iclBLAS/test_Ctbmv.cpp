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

TEST(Ctbmv, Ctbmv_up_ndiag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {2.f, 0.f}, {2.f, 0.f}, {1.5f, 0.f} };
    oclComplex_t ref_A[n * n] = { {1.f, 0.f}, {2.f, 0.f}, {0.f, 0.f},
                                    {0.f, 0.f}, {1.f, 0.f}, {2.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f} };

    oclComplex_t A[n * lda] = { {-1.f, 0.f}, {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {2.f + 4.f, 0.f}, {2.f + 3.f, 0.f}, {1.5f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_up_diag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_A[n * n] = { {15.f, 0.f}, {2.f, 0.f}, {0.f, 0.f},
                                    {0.f, 0.f}, {15.f, 0.f}, {2.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {15.f, 0.f} };

    oclComplex_t A[n * lda] = { {0.f, 0.f}, {15.f, 0.f}, {2.f, 0.f}, {15.f, 0.f}, {2.f, 0.f}, {15.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {3.f, 0.f}, {3.f, 0.f}, {1.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_up_ndiag_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 4;
    const int k = 2;
    const int lda = 3;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {4.35f, 0.f}, {1.25f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_A[n * n] = { {5.f, 0.f}, {2.f, 0.f}, {4.f, 0.f}, {0.f, 0.f},
                                    {0.f, 0.f}, {5.f, 0.f}, {6.f, 0.f}, {1.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {3.f, 0.f}, {1.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {4.f, 0.f} };

    oclComplex_t A[n * lda] = { {0.f, 0.f}, {0.f, 0.f}, {5.f, 0.f}, {0.f, 0.f}, {2.f, 0.f}, {5.f, 0.f}, {4.f, 0.f}, {6.f, 0.f}, {3.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {4.f, 0.f} };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
    {
        int kCount = 0;

        for (int j = 0; j < n; j++)
            if (j >= i && kCount <= k)
            {
                kCount += 1;

                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_low_ndiag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {2.f, 0.f}, {1.5f, 0.f} };
    oclComplex_t ref_A[n * n] = { {1.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {2.f, 0.f}, {1.f, 0.f}, {0.f, 0.f},
                                    {0.f, 0.f}, {2.f, 0.f}, {1.f, 0.f} };

    oclComplex_t A[n * lda] = { {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {0.f, 0.f} };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j <= i)
            {
                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_low_diag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.5f, 0.f} };
    oclComplex_t ref_A[n * n] = { {1.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {2.f, 0.f}, {1.f, 0.f}, {0.f, 0.f},
                                    {0.f, 0.f}, {2.f, 0.f}, {1.f, 0.f} };

    oclComplex_t A[n * lda] = { {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {0.f, 0.f} };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j <= i)
            {
                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_up_ndiag_trans_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_T;

    const int n = 4;
    const int k = 2;
    const int lda = 3;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.5f, 0.f}, {2.15f, 0.f}, {.45f, 0.f}, {1.7f, 0.f} };
    oclComplex_t ref_A[n * n] = { {5.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {2.f, 0.f}, {5.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {4.f, 0.f}, {6.f, 0.f}, {3.f, 0.f}, {0.f, 0.f},
                                    {0.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {4.f, 0.f} };

    oclComplex_t A[n * lda] = { {-1.f, 0.f}, {-1.f, 0.f}, {5.f, 0.f}, {-1.f, 0.f}, {2.f, 0.f}, {5.f, 0.f}, {4.f, 0.f}, {6.f, 0.f}, {3.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {4.f, 0.f} };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j <= i)
            {
                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_low_ndiag_trans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_T;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_A[n * n] = { {1.f, 0.f}, {2.f, 0.f}, {0.f, 0.f},
                                {0.f, 0.f}, {1.f, 0.f}, {2.f, 0.f},
                                {0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f} };

    oclComplex_t A[n * lda] = { {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {2.f, 0.f}, {1.f, 0.f}, {0.f, 0.f} };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j >= i)
            {
                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_low_ndiag_trans_hermit_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_C;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_A[n * n] = { { 1.f, 0.f },{ 2.f, -2.f },{ 0.f, 0.f },
                                { 0.f, 0.f },{ 1.f, 0.f },{ 2.f, 1.f },
                                { 0.f, 0.f },{ 0.f, 0.f },{ 1.f, 0.f } };

    oclComplex_t A[n * lda] = { { 1.f, 0.f },{ 2.f, -2.f },{ 1.f, 0.f },{ 2.f, 1.f },{ 1.f, 0.f },{ 0.f, 0.f } };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j >= i)
            {
                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_A[i * n + j].val[1] *= ref_A[i * n + j].val[1] == 0.f ? (1.f) : (-1.f);

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbmv, Ctbmv_up_ndiag_trans_hermit_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_T;

    const int n = 4;
    const int k = 2;
    const int lda = 3;

    const int incx = 1;

    oclComplex_t x[n * incx] = { { 1.5f, 0.f },{ 2.15f, 0.f },{ .45f, 0.f },{ 1.7f, 0.f } };
    oclComplex_t ref_A[n * n] = { { 5.f, 0.f },{ 2.f, -1.f },{ 4.f, 5.f },{ 0.f, 0.f },
                                    { 2.f, 1.f },{ 5.f, 0.f },{ 6.f, -1.f },{ 1.f, 0.f },
                                    { 4.f, -5.f },{ 6.f, 1.f },{ 3.f, 0.f },{ 0.f, -1.f },
                                    { 0.f, 0.f },{ 1.f, 0.f },{ 0.f, 1.f },{ 4.f, 0.f } };

    oclComplex_t A[n * lda] = { { -1.f, 0.f },{ -1.f, 0.f },{ 5.f, 0.f },{ -1.f, 0.f },{ 2.f, -1.f },{ 5.f, 0.f },{ 4.f, 5.f },{ 6.f, -1.f },{ 3.f, 0.f },{ 1.f, 0.f },{ 0.f, -1.f },{ 4.f, 0.f } };
    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j <= i)
            {
                if (j == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[j * incx].val[0];
                    ref_x[i * incx].val[1] += x[j * incx].val[1];
                    continue;
                }

                ref_A[i * n + j].val[1] *= ref_A[i * n + j].val[1] == 0.f ? (1.f) : (-1.f);

                ref_x[i * incx].val[0] += (ref_A[i * n + j].val[0] * x[j * incx].val[0] - ref_A[i * n + j].val[1] * x[j * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * n + j].val[0] * x[j * incx].val[1] + ref_A[i * n + j].val[1] * x[j * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}
