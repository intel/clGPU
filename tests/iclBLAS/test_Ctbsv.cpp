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

oclComplex_t cdiv(oclComplex_t a, oclComplex_t b)
{
    float divisor = b.val[0] * b.val[0] + b.val[1] * b.val[1];
    return { (a.val[0] * b.val[0] + a.val[1] * b.val[1]) / divisor, (a.val[1] * b.val[0] - a.val[0] * b.val[1]) / divisor };
}

TEST(Ctbsv, 3x4_up_ntrans_ndiag_k1)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 4;
    const int k = 1;
    const int lda = 3;
    const int incx = 1;

    oclComplex_t a[n * lda] = { {-1.f, 0.f}, {4.f, 0.f}, {-2.f, 0.f},
                                {1.f, 0.f}, {5.f, 0.f}, {-3.f, 0.f},
                                {2.f, 0.f}, {6.f, 0.f}, {-4.f, 0.f},
                                {3.f, 0.f}, {7.f, 0.f}, {-5.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { cdiv({6.f, 0.f}, {35.f, 0.f}), cdiv({11.f, 0.f}, {35.f, 0.f}), cdiv({3.f, 0.f}, {14.f, 0.f}), cdiv({4.f, 0.f}, {7.f, 0.f}) };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbsv, 3x4_low_ntrans_ndiag_k2)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_N;

    const int n = 4;
    const int k = 2;
    const int lda = 3;
    const int incx = 1;

    oclComplex_t a[n * lda] = { {1.f, 0.f}, {5.f, 0.f}, {9.f, 0.f},
                                {2.f, 0.f}, {6.f, 0.f}, {10.f, 0.f},
                                {3.f, 0.f}, {7.f, 0.f}, {-11.f, 0.f},
                                {4.f, 0.f}, {-8.f, 0.f}, {-12.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {1.f, 0.f}, cdiv({-3.f, 0.f}, {2.f, 0.f}), {1.f, 0.f}, {3.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbsv, 3x4_low_trans_diag)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_T;

    const int n = 4;
    const int k = 2;
    const int lda = 3;
    const int incx = 1;

    oclComplex_t a[n * lda] = { {1.f, 0.f}, {5.f, 0.f}, {9.f, 0.f},
                                {2.f, 0.f}, {6.f, 0.f}, {10.f, 0.f},
                                {3.f, 0.f}, {7.f, 0.f}, {-11.f, 0.f},
                                {4.f, 0.f}, {-8.f, 0.f}, {-12.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { {-334.f, 0.f}, {112.f, 0.f}, {-25.f, 0.f}, {4.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbsv, 2x4_up_trans_ndiag_2incx_k1)
{
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const auto trans = ICLBLAS_OP_C;

    const int n = 3;
    const int k = 1;
    const int lda = 2;
    const int incx = 2;

    oclComplex_t a[n * lda] = { {-1.f, 0.f}, {3.f, 0.f},
                                {1.f, 0.f}, {4.f, 0.f},
                                {2.f, 0.f}, {5.f, 0.f} };

    oclComplex_t x[n * incx] = { {1.f, 0.f}, {-2.f, 0.f}, {3.f, 0.f}, {-4.f, 0.f}, {5.f, 0.f}, {-6.f, 0.f} };
    oclComplex_t ref_x[n * incx] = { cdiv({1.f, 0.f}, {3.f, 0.f}), {-2.f, 0.f}, cdiv({8.f, 0.f}, {12.f, 0.f}), {-4.f, 0.f}, cdiv({22.f, 0.f}, {30.f, 0.f}), {-6.f, 0.f} };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}

TEST(Ctbsv, 3x4_low_trans_hermit_diag)
{
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const auto trans = ICLBLAS_OP_C;

    const int n = 4;
    const int k = 2;
    const int lda = 3;
    const int incx = 1;

    oclComplex_t a[n * lda] = { { 1.f, 1.f },{ 5.f, -5.f },{ 9.f, 20.f },
                                { 2.f, -1.f },{ 6.f, 0.f },{ 10.f, -4.f },
                                { 3.f, -12.f },{ 7.f, 0.f },{ -11.f, 7.f },
                                { 4.f, 1.f },{ -8.f, 0.f },{ -12.f, 0.f } };

    oclComplex_t x[n * incx] = { { 1.f, 0.f },{ 2.f, 0.f },{ 3.f, 0.f },{ 4.f, 0.f } };
    oclComplex_t ref_x[n * incx] = { { -414.f, -980.f },{ 112.f, -16.f },{ -25.f, 0.f },{ 4.f, 0.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[0], x[i * incx].val[0]);
        EXPECT_FLOAT_EQ(ref_x[i * incx].val[1], x[i * incx].val[1]);
    }
}
