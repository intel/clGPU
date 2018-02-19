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

TEST(Strsv, 5x4_low_ntrans_ndiag) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int lda = 5;
    const int incx = 1;
    
    float a[n*lda] = { 2.f, 3.f, 4.f, 5.f, -1.f,
                      -2.f, 6.f, 7.f, 8.f, -3.f,
                      -4.f, -5.f, 9.f, 10.f, -6.f,
                      -7.f, -8.f, -9.f, 11.f, -10.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { 1.f/2.f, 1.f/12.f, 5.f/108.f, 40.f / 1188.f };
    
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsv(handle, uplo, trans, diag, n, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Strsv, 5x4_up_ntrans_ndiag) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int lda = 5;
    const int incx = 1;
    
    float a[n*lda] = { 1.f, -1.f, -2.f, -3.f, -4.f,
                       2.f, 3.f, -5.f, -6.f, -7.f,
                       4.f, 5.f, 6.f, -8.f, -9.f,
                       7.f, 8.f, 9.f, 10.f, -10.f };

    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { -14.f/15.f, -7.f/30.f, -1.f / 10.f, 4.f / 10.f };
    
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsv(handle, uplo, trans, diag, n, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Strsv, 5x4_up_trans_diag) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 4;
    const int lda = 5;
    const int incx = 1;

    float a[n*lda] = { 1.f, -1.f, -2.f, -3.f, -4.f,
                      2.f, 3.f, -5.f, -6.f, -7.f,
                      4.f, 5.f, 6.f, -8.f, -9.f,
                      7.f, 8.f, 9.f, 10.f, -10.f };

    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { 1.f, 0.f, -1.f, 6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsv(handle, uplo, trans, diag, n, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Strsv, 3x3_low_trans_ndiag_2incx) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 3;
    const int lda = 3;
    const int incx = 2;

    float a[n*lda] = { 2.f, 3.f, 4.f,
        -             -1.f, 5.f, 6.f,
                      -2.f, -3.f, 7.f };

    float x[n*incx] = { 1.f, -2.f, 3.f, -4.f, 5.f, -6.f };
    const float expected[n*incx] = { -38.f/70.f, -2.f, -9.f/35.f, -4.f, 5.f/7.f, -6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStrsv(handle, uplo, trans, diag, n, a, lda, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], x[i]);
    }
}
