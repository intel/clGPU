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

TEST(Stpsv, naive_n4_up_ntrans_ndiag) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int incx = 1;
    float ap[n*(n + 1) / 2] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};
    float x[n*incx] = {1.f, 2.f, 3.f, 4.f};
    const float expected[n*incx] = { -28.f/30.f, -7.f/30.f,-0.1f, 2.f/5.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStpsv(handle, uplo, trans, diag, n, ap, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++) {
        ASSERT_FLOAT_EQ(expected[i], x[i]);
    }

}

TEST(Stpsv, naive_n4_low_ntrans_diag_2inc) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const auto diag = ICLBLAS_DIAG_UNIT;
    const int n = 4;
    const int incx = 2;
    float ap[n*(n + 1) / 2] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f };
    float x[n*incx] = { 1.f, -2.f, 3.f,- 4.f, 5.f, -6.f, 7.f, -8.f};
    const float expected[n*incx] = { 1.f, -2.f, 1.f, -4.f, -4.f,-6.f, 32.f, -8.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStpsv(handle, uplo, trans, diag, n, ap, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++) {
        ASSERT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Stpsv, naive_n4_up_trans_ndiag) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int incx = 1;
    float ap[n*(n + 1) / 2] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { 1.f, 0.f, -1.f/6.f, -3.f / 20.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStpsv(handle, uplo, trans, diag, n, ap, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++) {
        ASSERT_FLOAT_EQ(expected[i], x[i]);
    }
}

TEST(Stpsv, naive_n4_low_trans_ndiag) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const auto diag = ICLBLAS_DIAG_NON_UNIT;
    const int n = 4;
    const int incx = 1;
    float ap[n*(n + 1) / 2] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f };
    const float expected[n*incx] = { -47.f / 200.f, -7.f / 100.f, -3.f / 40.f, 4.f / 10.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasStpsv(handle, uplo, trans, diag, n, ap, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    ASSERT_FLOAT_EQ(expected[2], x[2]);
    for (int i = 0; i < n*incx; i++) {
        ASSERT_FLOAT_EQ(expected[i], x[i]);
    }
}
