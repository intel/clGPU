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

TEST(Stbsv, 3x4_up_ntrans_ndiag_k1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int k = 1;
    const int lda = 3;
    const int incx = 1;

    float a[lda*n] = { -1.f, 4.f, -2.f,
                        1.f, 5.f, -3.f,
                        2.f, 6.f, -4.f,
                        3.f, 7.f, -5.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { 6.f/35.f, 11.f/35.f, 3.f/14.f, 4.f/7.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Stbsv, 3x4_low_ntrans_ndiag_k2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int k = 2;
    const int lda = 3;
    const int incx = 1;

    float a[lda*n] = { 1.f, 5.f, 9.f,
                       2.f, 6.f, 10.f,
                       3.f, 7.f, -11.f,
                       4.f, -8.f, -12.f };

    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { 1.f, -3.f/2.f, 1.f, 3.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Stbsv, 3x4_low_trans_diag) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 4;
    const int k = 2;
    const int lda = 3;
    const int incx = 1;

    float a[lda*n] = { 1.f, 5.f, 9.f,
                       2.f, 6.f, 10.f,
                       3.f, 7.f, -11.f,
                       4.f, -8.f, -12.f };

    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { -334.f, 112.f, -25.f, 4.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Stbsv, 2x4_up_trans_ndiag_2incx_k1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int k = 1;
    const int lda = 2;
    const int incx = 2;

    float a[lda*n] = { -1.f, 3.f,
                        1.f, 4.f,
                        2.f, 5.f };

    float x[n*incx] = { 1.f, -2.f, 3.f, -4.f, 5.f, -6.f };
    const float expected[n*incx] = { 1.f / 3.f, -2.f, 8.f/12.f, -4.f, 22.f / 30.f, -6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStbsv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}
