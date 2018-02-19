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

TEST(Ctpmv, Ctpmv_up_diag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int incx = 1;

    oclComplex_t ref_AP[n * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
                                    {0.f, 0.f}, {4.f, 0.f}, {5.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {6.f, 0.f} };

    oclComplex_t AP[(n*(n + 1)) / 2] = { {1.f, 0.f}, {2.f, 0.f}, {4.f, 0.f}, {3.f, 0.f}, {5.f, 0.f}, {6.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {6.f, 0.f}, {6.f, 0.f}, {1.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_low_diag_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 3;
    const int incx = 1;

    oclComplex_t ref_AP[n * n] = { {1.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {2.f, 0.f}, {4.f, 0.f}, {0.f, 0.f},
                                    {3.f, 0.f}, {5.f, 0.f}, {6.f, 0.f} };

    oclComplex_t AP[(n*(n + 1)) / 2] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f}, {5.f, 0.f}, {6.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {1.f, 0.f}, {3.f, 0.f}, {9.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_up_ndiag_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 4;
    const int incx = 2;

    oclComplex_t ref_AP[n * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f},
                                    {0.f, 0.f}, {5.f, 0.f}, {6.f, 0.f}, {7.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {8.f, 0.f}, {9.f, 0.f},
                                    {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {10.f, 0.f} };

    oclComplex_t AP[(n*(n + 1)) / 2] = { {1.f, 0.f}, {2.f, 0.f}, {5.f, 0.f}, {3.f, 0.f}, {6.f, 0.f}, {8.f, 0.f}, {4.f, 0.f}, {7.f, 0.f}, {9.f, 0.f}, {10.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {0.5f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {9.f, 0.f}, {1.f, 0.f}, {15.5f, 0.f}, {1.f, 0.f}, {17.f, 0.f}, {1.f, 0.f}, {10.f, 0.f}, {1.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_low_ndiag_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 4;
    const int incx = 1;

    oclComplex_t ref_AP[n * n] = { {1.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {2.f, 0.f}, {5.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {3.f, 0.f}, {6.f, 0.f}, {8.f, 0.f}, {0.f, 0.f},
                                    {4.f, 0.f}, {7.f, 0.f}, {9.f, 0.f}, {10.f, 0.f} };

    oclComplex_t AP[(n*(n + 1)) / 2] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f}, {5.f, 0.f}, {6.f, 0.f}, {7.f, 0.f}, {8.f, 0.f}, {9.f, 0.f}, {10.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {.25f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {1.f, 0.f}, {7.f, 0.f}, {17.f, 0.f}, {22.5f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_up_ndiag_nxn)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 25;
    const int incx = 1;

    oclComplex_t ref_AP[n * n];
    oclComplex_t x[n * incx];

    int idx = 0;
    const int APn = (n*(n + 1)) / 2;
    oclComplex_t AP[APn];

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            ref_AP[i * n + k] = { static_cast<float>(std::rand() % 10), static_cast<float>(std::rand() % 10) };
        }

        x[i] = { 1.f, 0.f };
    }

    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            if (k >= i)
            {
                AP[idx++] = ref_AP[i * n + k];
            }
        }
    }

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i].val[0] += x[k].val[0];
                    ref_x[i].val[1] += x[k].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_AP[i * n + k].val[0] * x[k * incx].val[0] - ref_AP[i * n + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_AP[i * n + k].val[0] * x[k * incx].val[1] + ref_AP[i * n + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_low_ndiag_nxn)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_N;

    const int n = 25;
    const int incx = 1;

    oclComplex_t ref_AP[n * n];
    oclComplex_t x[n * incx];

    int idx = 0;
    const int APn = (n*(n + 1)) / 2;
    oclComplex_t AP[APn];

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            ref_AP[i * n + k] = { static_cast<float>(std::rand() % 10),static_cast<float>(std::rand() % 10) };
        }

        x[i] = { 1.f, 0.f };
    }

    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            if (k <= i)
            {
                AP[idx++] = ref_AP[i * n + k];
            }
        }
    }

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i].val[0] += x[k].val[0];
                    ref_x[i].val[1] += x[k].val[1];
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_AP[i * n + k].val[0] * x[k * incx].val[0] - ref_AP[i * n + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_AP[i * n + k].val[0] * x[k * incx].val[1] + ref_AP[i * n + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_low_ndiag_trans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_T;

    const int n = 3;
    const int incx = 1;

    oclComplex_t ref_AP[n * n] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f},
    {0.f, 0.f}, {4.f, 0.f}, {5.f, 0.f},
    {0.f, 0.f}, {0.f, 0.f}, {6.f, 0.f} };

    oclComplex_t AP[(n*(n + 1)) / 2] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f}, {5.f, 0.f}, {6.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {6.f, 0.f}, {9.f, 0.f}, {6.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_low_ndiag_trans_nxn)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_T;

    const int n = 23;
    const int incx = 1;

    oclComplex_t ref_AP[n * n];
    oclComplex_t tmp_AP[n * n];
    oclComplex_t x[n];

    oclComplex_t AP[(n*(n + 1)) / 2];

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
            tmp_AP[i * n + k] = { static_cast<float>(std::rand() % 10),static_cast<float>(std::rand() % 10) };

        x[i] = { 1.f, 0.f };
    }

    int idx = 0;

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if (k <= i)
                AP[idx++] = tmp_AP[i * n + k];

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            ref_AP[i * n + k] = tmp_AP[k * n + i];

    oclComplex_t ref_x[n];

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i].val[0] += x[k].val[0];
                    ref_x[i].val[1] += x[k].val[1]; 
                    continue;
                }

                ref_x[i * incx].val[0] += (ref_AP[i * n + k].val[0] * x[k * incx].val[0] - ref_AP[i * n + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_AP[i * n + k].val[0] * x[k * incx].val[1] + ref_AP[i * n + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_up_ndiag_trans_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_T;

    const int n = 3;
    const int incx = 2;

    oclComplex_t ref_AP[n * n] = { {1.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
                                    {2.f, 0.f}, {4.f, 0.f}, {0.f, 0.f},
                                    {3.f, 0.f}, {5.f, 0.f}, {6.f, 0.f} };

    oclComplex_t AP[(n*(n + 1)) / 2] = { {1.f, 0.f}, {2.f, 0.f}, {4.f, 0.f}, {3.f, 0.f}, {5.f, 0.f}, {6.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f}, {1.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {1.f, 0.f}, {1.f, 0.f}, {6.f, 0.f}, {1.f, 0.f}, {14.f, 0.f}, {1.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_up_ndiag_trans_hermit_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_C;

    const int n = 3;
    const int incx = 2;

    oclComplex_t ref_AP[n * n] = { { 1.f, 0.f },{ 2.f, -1.f },{ 3.f, -3.f },
                                    { 2.f, 1.f },{ 4.f, 0.f },{ 5.f, 2.f },
                                    { 3.f, 3.f },{ 5.f, -2.f },{ 6.f, 0.f } };

    oclComplex_t AP[(n*(n + 1)) / 2] = { { 1.f, 0.f },{ 2.f, 1.f },{ 4.f, 0.f },{ 3.f, 3.f },{ 5.f, -2.f },{ 6.f, 0.f } };
    oclComplex_t x[n * incx] = { { 1.f, 0.f },{ 1.f, 0.f },{ 1.f, 0.f },{ 1.f, 0.f },{ 1.f, 0.f },{ 1.f, 0.f } };

    oclComplex_t ref_x[n * incx];

    for (int i = 0; i < n * incx; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i].val[0] += x[k].val[0];
                    ref_x[i].val[1] += x[k].val[1];
                    continue;
                }

                ref_AP[i * n + k].val[1] *= (ref_AP[i * n + k].val[1] == 0.f) ? 1.f : -1.f;

                ref_x[i * incx].val[0] += (ref_AP[i * n + k].val[0] * x[k * incx].val[0] - ref_AP[i * n + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_AP[i * n + k].val[0] * x[k * incx].val[1] + ref_AP[i * n + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctpmv, Ctpmv_low_ndiag_trans_hermit_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    iclblasDiagType_t diag = ICLBLAS_DIAG_NON_UNIT;
    iclblasOperation_t trans = ICLBLAS_OP_C;

    const int n = 3;
    const int incx = 1;

    oclComplex_t ref_AP[n * n] = { { 1.f, 0.f },{ 2.f, -2.f },{ 3.f, -15.f },
                                    { 2.f, 2.f },{ 4.f, 0.f },{ 5.f, -1.5f },
                                    { 3.f, 15.f },{ 5.f, -1.5f },{ 6.f, 0.f } };

    oclComplex_t AP[(n*(n + 1)) / 2] = { { 1.f, 0.f },{ 2.f, -2.f },{ 3.f, -15.f },{ 4.f, 0.f },{ 5.f, -1.5f },{ 6.f, 0.f } };
    
    oclComplex_t ref_x[n * incx];
    oclComplex_t x[n * incx] = { { 1.f, 0.f },{ 1.f, 0.f },{ 1.f, 0.f } };

    for (int i = 0; i < n; i++)
        ref_x[i] = { 0.f, 0.f };

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k >= i)
            {
                if (k == i && (diag == ICLBLAS_DIAG_UNIT))
                {
                    ref_x[i].val[0] += x[k].val[0];
                    ref_x[i].val[1] += x[k].val[1];
                    continue;
                }

                ref_AP[i * n + k].val[1] *= (ref_AP[i * n + k].val[1] == 0.f) ? 1.f : -1.f;

                ref_x[i * incx].val[0] += (ref_AP[i * n + k].val[0] * x[k * incx].val[0] - ref_AP[i * n + k].val[1] * x[k * incx].val[1]);
                ref_x[i * incx].val[1] += (ref_AP[i * n + k].val[0] * x[k * incx].val[1] + ref_AP[i * n + k].val[1] * x[k * incx].val[0]);
            }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        ASSERT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}
