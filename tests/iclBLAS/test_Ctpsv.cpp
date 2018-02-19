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

TEST(Ctpsv, Naive_Upper_N_Unit)
{
    auto uplo = ICLBLAS_FILL_MODE_UPPER;
    auto trans = ICLBLAS_OP_N;
    auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int incx = 1;
    const int lda = 3;
    oclComplex_t x[n * incx] = { { 1,0 },{ 1,0 },{ 1,0 }};
    oclComplex_t AP[n*n] =
    {
        { 1,0 },
        { 3,4 },{ 1,0 },
        { 7,8 },{ 9,10 },{ 1,0 },
    };

    oclComplex_t ex_result[n * incx] = { { -22, 54 },{ -8, -10 },{ 1,0 }};

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpsv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i].val[0], x[i].val[0]);
        EXPECT_FLOAT_EQ(ex_result[i].val[1], x[i].val[1]);
    }
}

TEST(Ctpsv, Naive_Lower_T_Unit)
{
    auto uplo = ICLBLAS_FILL_MODE_LOWER;
    auto trans = ICLBLAS_OP_T;
    auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int incx = 1;
    const int lda = 3;
    oclComplex_t x[n * incx] = { { 1,0 },{ 1,0 },{ 1,0 } };
    oclComplex_t AP[n*n] =
    {
        { 1,0 },
        { 2,2 },{ 1,0 },
        { 4,4 },{ 5,5 },{ 1,0 },
    };

    oclComplex_t ex_result[n * incx] = { { -2, 18 },{ -4, -5 },{ 1, 0 } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpsv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i].val[0], x[i].val[0]);
        EXPECT_FLOAT_EQ(ex_result[i].val[1], x[i].val[1]);
    }
}

TEST(Ctpsv, Naive_Lower_T_Unit_incx)
{
    auto uplo = ICLBLAS_FILL_MODE_LOWER;
    auto trans = ICLBLAS_OP_T;
    auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 3;
    const int incx = 2;
    const int lda = 3;
    oclComplex_t x[n * incx] = { { 1,0 },{ 1,0 },{ 1,0 }, {1,2}, {2,3}, {4,5} };
    oclComplex_t AP[n*n] =
    {
        { 1,0 },
        { 2,2 },{ 1,0 },
        { 4,4 },{ 5,5 },{ 1,0 },
    };

    oclComplex_t ex_result[n * incx] = { { -63, 35 },{ 1, 0 },{ 6, -25 },{ 1, 2 },{ 2, 3 },{ 4, 5 } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCtpsv(handle, uplo, trans, diag, n, AP, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incx; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i].val[0], x[i].val[0]);
        EXPECT_FLOAT_EQ(ex_result[i].val[1], x[i].val[1]);
    }
}