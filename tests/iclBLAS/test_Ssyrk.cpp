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

TEST(Ssyrk, 3x3_up_ntrans) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const int n = 3;
    const int k = 3;

    const int lda = 4;
    float alpha = 1.1f;
    float a[k*lda] = { 1.f, 2.f, 3.f, -1.f,
                       4.f, 5.f, 6.f, -2.f,
                       7.f, 8.f, 9.f, -3.f };

    const int ldc = 3;
    float beta = 1.3f;
    float c[n*ldc] = { 10.f, -1.f, -2.f,
                       11.f, 12.f, -3.f,
                       13.f, 14.f, 15.f };

    float expected[n*ldc] = { 1.3f*10.f + 1.1f*66.f, -1.f, -2.f,
                              1.3f*11.f + 1.1f*78.f, 1.3f*12.f + 1.1f*93.f, -3.f,
                              1.3f*13.f + 1.1f*90.f, 1.3f*14.f + 1.1f*108.f, 1.3f*15.f + 1.1f*126.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyrk(handle, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], c[i]);
    }
}

TEST(Ssyrk, 3x4_low_ntrans) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const int n = 3;
    const int k = 4;

    const int lda = 3;
    float alpha = -1.1f;
    float a[k*lda] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f };

    const int ldc = 3;
    float beta = 1.3f;
    float c[n*ldc] = { 10.f, 11.f, 12.f,
                       -1.f, 13.f, 14.f,
                       -2.f, -3.f, 15.f };

    float expected[n*ldc] = { 1.3f*10.f - 1.1f*166.f, 1.3f*11.f - 1.1f*188.f, 1.3f*12.f - 1.1f*210.f,
                              -1.f, 1.3f*13.f - 1.1f*214.f, 1.3f*14.f - 1.1f*240.f,
                              -2.f, -3.f, 1.3f*15.f - 1.1f*270.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyrk(handle, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], c[i]);
    }
}

TEST(Ssyrk, 3x3_up_trans) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const int n = 3;
    const int k = 3;

    const int lda = 3;
    float alpha = 1.1f;
    float a[n*lda] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f };

    const int ldc = 4;
    float beta = -1.3f;
    float c[n*ldc] = { 10.f, -1.f, -2.f, -3.f,
        11.f, 12.f, -4.f, -5.f,
        13.f, 14.f, 15.f, -6.f };

    float expected[n*ldc] = { -1.3f*10.f + 1.1f*14.f, -1.f, -2.f, -3.f,
                              -1.3f*11.f + 1.1f*32.f, -1.3f*12.f + 1.1f*77.f, -4.f, -5.f,
                              -1.3f*13.f + 1.1f*50.f, -1.3f*14.f + 1.1f*122.f, -1.3f*15.f + 1.1f*194.f, -6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyrk(handle, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], c[i]);
    }
}

TEST(Ssyrk, 4x3_low_trans) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const int n = 4;
    const int k = 3;

    const int lda = 3;
    float alpha = 1.1f;
    float a[n*lda] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f };

    const int ldc = 4;
    float beta = 1.3f;
    float c[n*ldc] = { 13.f, 14.f, 15.f, 16.f,
                       -1.f, 17.f, 18.f, 19.f,
                       -2.f, -3.f, 20.f, 21.f,
                       -4.f, -5.f, -6.f, 22.f };

    float expected[n*ldc] = { 1.3f*13.f + 1.1f*14.f, 1.3f*14.f + 1.1f*32.f, 1.3f*15.f + 1.1f*50.f, 1.3f*16.f + 1.1f*68.f,
                             -1.f, 1.3f*17.f + 1.1f*77.f, 1.3f*18.f + 1.1f*122.f, 1.3f*19.f + 1.1f*167.f,
                             -2.f, -3.f, 1.3f*20.f + 1.1f*194.f, 1.3f*21.f + 1.1f*266.f,
                             -4.f, -5.f, -6.f, 1.3f*22.f + 1.1f*365.f};

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyrk(handle, uplo, trans, n, k, &alpha, a, lda, &beta, c, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], c[i]);
    }
}
