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
#include <iostream>

TEST(Ctrmv, Ctrmv_up_ndiag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int lda = 4;

    const int incx = 2;

    oclComplex_t A[n * lda] = { {1.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {-1.f, 0.f},
                                {2.f, 0.f}, {4.f, 0.f}, {0.f, 0.f}, {-1.f, 0.f},
                                {3.f, 0.f}, {5.f, 0.f}, {6.f, 0.f}, {-1.f, 0.f} };
    oclComplex_t x[n * incx] = { {1.f, 0.f}, {0.f, 0.f}, {1.35f, 0.f}, {-1.f, 0.f}, {4.89f, 0.f}, {0.f, 0.f} };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (A[i * lda + k].val[0] * x[k * incx].val[0] - A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (A[i * lda + k].val[0] * x[k * incx].val[1] + A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_up_diag_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    oclComplex_t A[n * lda] = { {1.f,0.f}, {4.f, 0.f}, {6.f, 0.f}, {0.f, 0.f},
                                {2.7f,0.f}, {5.1f, 0.f}, {1.f, 0.f}, {4.f, 0.f},
                                {3.f, 0.f}, {0.f, 0.f}, {23.f, 0.f}, {5.5f, 0.f},
                                {0.4f, 0.f}, {0.f, 0.f}, {3.f, 0.f}, {5.7f, 0.f} };
    oclComplex_t x[n * incx] = { {1.7f, 0.f}, {1.5f, 0.f}, {1.3f, 0.f}, {0.f,0.f} };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (A[i * lda + k].val[0] * x[k * incx].val[0] - A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (A[i * lda + k].val[0] * x[k * incx].val[1] + A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_low_diag_3x3)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 3;
    const int lda = 4;

    const int incx = 1;

    oclComplex_t A[n * lda] = { {1.9f, 0.f}, {0.5f, 0.f}, {0.7f, 0.f}, {-1.f, 0.f},
                                {1.6f, 0.f}, {4.f, 0.f}, {14.35f, 0.f}, {0.f, 0.f},
                                {3.f, 0.f}, {5.155f, 0.f}, {6.f, 0.f}, {1.f, 0.f} };
    oclComplex_t x[n * incx] = { {1.9f, 0.f}, {0.7f, 0.f}, {1.35f, 0.f} };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (A[i * lda + k].val[0] * x[k * incx].val[0] - A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (A[i * lda + k].val[0] * x[k * incx].val[1] + A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_low_ndiag_2x2)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 2;
    const int lda = 2;

    const int incx = 1;

    oclComplex_t A[n * lda] = { {1.4f, 0.f}, {37.f, 0.f},
                                {1.7f, 0.f}, {1.3f, 0.f} };
    oclComplex_t x[n * incx] = { {1.7f, 0.f}, {1.54f, 0.f} };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1]; 
                    continue;
                }

                ref_x[i * incx].val[0] += (A[i * lda + k].val[0] * x[k * incx].val[0] - A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (A[i * lda + k].val[0] * x[k * incx].val[1] + A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_low_ndiag_trans_3x3)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 3;
    const int lda = 5;

    const int incx = 2;

    oclComplex_t A[n * lda] = { {6.f, 0.f}, {5.f, 0.f}, {3.f, 0.f}, {5.f, 0.f}, {3.f, 0.f},
                                {0.f, 0.f}, {4.f, 0.f}, {2.f, 0.f}, {4.f, 0.f}, {2.f, 0.f},
                                {0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f}, {0.f, 0.f}, {1.f, 0.f} };
    oclComplex_t x[n * incx] = { {1.f, 0.f}, {.5f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {.75f, 0.f}, {1.f, 0.f} };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1]; 
                    continue;
                }

                ref_x[i * incx].val[0] += (A[i * lda + k].val[0] * x[k * incx].val[0] - A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (A[i * lda + k].val[0] * x[k * incx].val[1] + A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_up_diag_trans_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    oclComplex_t A[n * lda] = { {1.f, 0.f}, {2.7f, 0.f}, {3.f, 0.f}, {0.4f, 0.f},
                                {4.f, 0.f}, {5.1f, 0.f}, {0.f, 0.f}, {0.9f, 0.f},
                                {6.f, 0.f}, {1.f, 0.f}, {23.f, 0.f}, {3.f, 0.f},
                                {0.75f, 0.f}, {4.f, 0.f}, {5.5f, 0.f}, {5.7f, 0.f} };
    oclComplex_t x[n * incx] = { {1.7f, 0.f}, {1.5f, 0.f}, {1.3f, 0.f}, {0.1f, 0.f} };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1]; 
                    continue;
                }

                ref_x[i * incx].val[0] += (A[i * lda + k].val[0] * x[k * incx].val[0] - A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (A[i * lda + k].val[0] * x[k * incx].val[1] + A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_low_ndiag_trans_hermit_3x3)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_C;

    const int n = 3;
    const int lda = 5;

    const int incx = 2;

    oclComplex_t A[n * lda] = { { 6.f, 0.f },{ 5.f, 1.f },{ 3.f, 0.f },{ 5.f, -4.f },{ 3.f, 0.f },
                                { 5.f, -1.f },{ 4.f, 0.f },{ 2.f, -0.5f },{ 4.f, 3.f },{ 2.f, 0.f },
                                { 3.f, 0.f },{ 2.f, 0.5f },{ 1.f, 0.f },{ 0.f, 0.f },{ 1.f, 0.f } };

    oclComplex_t ref_A[n * lda] = { { 6.f, 0.f },{ 5.f, 1.f },{ 3.f, 0.f },{ 5.f, -4.f },{ 3.f, 0.f },
                                    { 5.f, -1.f },{ 4.f, 0.f },{ 2.f, -0.5f },{ 4.f, 3.f },{ 2.f, 0.f },
                                    { 3.f, 0.f },{ 2.f, 0.5f },{ 1.f, 0.f },{ 0.f, 0.f },{ 1.f, 0.f } };

    oclComplex_t x[n * incx] = { { 1.f, 0.f },{ .5f, 0.f },{ 1.f, 0.f },{ 1.f, 0.f },{ .75f, 0.f },{ 1.f, 0.f } };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1];
                    continue;
                }

                ref_A[i * lda + k].val[1] *= ref_A[i * lda + k].val[1] == 0.f ? (1.f) : (-1.f);

                ref_x[i * incx].val[0] += (ref_A[i * lda + k].val[0] * x[k * incx].val[0] - ref_A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * lda + k].val[0] * x[k * incx].val[1] + ref_A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctrmv, Ctrmv_up_diag_trans_hermit_4x4)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_C;

    const int n = 4;
    const int lda = 4;

    const int incx = 1;

    oclComplex_t A[n * lda] = { { 1.f, 0.f },{ 2.7f, -1.f },{ 3.f, 0.5f },{ 0.75f, 0.f },
                                { 2.7f, 1.f },{ 5.1f, 0.f },{ 0.f, 2.f },{ 0.9f, -1.f },
                                { 3.f, -0.5f },{ 0.f, -2.f },{ 23.f, 0.f },{ 5.5f, 14.f },
                                { 0.75f, 0.f },{ 0.9f, 1.f },{ 5.5f, -14.f },{ 5.7f, 0.f } };

    oclComplex_t ref_A[n * lda] = { { 1.f, 0.f },{ 2.7f, -1.f },{ 3.f, 0.5f },{ 0.75f, 0.f },
                                    { 2.7f, 1.f },{ 5.1f, 0.f },{ 0.f, 2.f },{ 0.9f, -1.f },
                                    { 3.f, -0.5f },{ 0.f, -2.f },{ 23.f, 0.f },{ 5.5f, 14.f },
                                    { 0.75f, 0.f },{ 0.9f, 1.f },{ 5.5f, -14.f },{ 5.7f, 0.f } };

    oclComplex_t x[n * incx] = { { 1.7f, 0.f },{ 1.5f, 0.f },{ 1.3f, 0.f },{ 0.1f, 0.f } };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i * incx].val[0] += x[k * incx].val[0];
                    ref_x[i * incx].val[1] += x[k * incx].val[1];
                    continue;
                }

                ref_A[i * lda + k].val[1] *= ref_A[i * lda + k].val[1] == 0.f ? (1.f) : (-1.f);

                ref_x[i * incx].val[0] += (ref_A[i * lda + k].val[0] * x[k * incx].val[0] - ref_A[i * lda + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_A[i * lda + k].val[0] * x[k * incx].val[1] + ref_A[i * lda + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}
