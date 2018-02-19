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

TEST(Sgbmv, 5x5_ntrans_u1l1) {
    const auto trans = ICLBLAS_OP_N;

    const int m = 5;
    const int n = 5;
    const int ku = 1;
    const int kl = 1;
    float alpha = 1.1f;
    const int lda = 3;
    const int incx = 1;
    float beta = 1.3f;
    const int incy = 1;

    float a[lda*n] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f,
                       13.f, 14.f, 15.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    float y[m*incy] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    const float expected[m*incy] = { 1.3f + 1.1f*(2.f + 2.f*4.f),
                                     2.f*1.3f + 1.1f*(3.f + 2.f*5.f + 3.f*7.f),
                                     3.f*1.3f + 1.1f*(2.f*6.f + 3.f*8.f + 4.f*10.f),
                                     4.f*1.3f + 1.1f*(3.f*9.f+4.f*11.f + 5.f*13.f),
                                     5.f*1.3f + 1.1f*(4.f*12.f + 5.f*14.f) };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgbmv(handle, trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m*incy; i++) {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}

TEST(Sgbmv, 3x2_ntrans_u1l1) {
    const auto trans = ICLBLAS_OP_N;

    const int m = 3;
    const int n = 2;
    const int ku = 1;
    const int kl = 1;
    float alpha = 1.1f;
    const int lda = 4;
    const int incx = 1;
    float beta = 1.3f;
    const int incy = 1;

    float a[lda*n] = { 1.f, 2.f, 3.f, -1.f,
                       4.f, 5.f, 6.f, -2.f };
    float x[n*incx] = { 1.f, 2.f };
    float y[m*incy] = { 1.f, 2.f, 3.f };
    const float expected[m*incy] = { 1.3f + 1.1f*(2.f + 2.f*4.f),
                                     2.f*1.3f + 1.1f*(3.f + 2.f*5.f),
                                     3.f*1.3f + 1.1f*(2.f*6.f) };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgbmv(handle, trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m*incy; i++) {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}

TEST(Sgbmv, 3x3_trans_inc2) {
    const auto trans = ICLBLAS_OP_T;

    const int m = 3;
    const int n = 3;
    const int ku = 0;
    const int kl = 1;
    float alpha = 1.1f;
    const int lda = 2;
    const int incx = 2;
    float beta = 1.3f;
    const int incy = 2;

    float a[lda*n] = { 1.f, 2.f,
                       3.f, 4.f,
                       5.f, 6.f };
    float x[m*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    float y[n*incy] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    const float expected[n*incy] = { 1.3f + 1.1f*(1.f + 3.f*2.f), 2.f,
                                     3.f*1.3f + 1.1f*(3.f*3.f + 5.f*4.f), 4.f,
                                     5.f*1.3f + 1.1f*(5.f*5.f), 6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgbmv(handle, trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}

TEST(Sgbmv, 3x3_trans_u1l1) {
    const auto trans = ICLBLAS_OP_C;

    const int m = 3;
    const int n = 3;
    const int ku = 1;
    const int kl = 1;
    float alpha = 1.1f;
    const int lda = 3;
    const int incx = 1;
    float beta = 1.3f;
    const int incy = 1;

    float a[lda*n] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f};
    float x[m*incx] = { 1.f, 2.f, 3.f};
    float y[n*incy] = { 4.f, 5.f, 6.f };
    const float expected[n*incy] = { 4*1.3f + 1.1f*(2.f + 2.f*3.f),
                                     5.f*1.3f + 1.1f*(4.f + 2.f*5.f + 3.f*6.f),
                                     6.f*1.3f + 1.1f*(2.f*7.f + 3.f*8.f) };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgbmv(handle, trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}

TEST(Sgbmv, 3x4_trans_u1l1) {
    const auto trans = ICLBLAS_OP_T;

    const int m = 3;
    const int n = 4;
    const int ku = 1;
    const int kl = 1;
    float alpha = 1.1f;
    const int lda = 3;
    const int incx = 1;
    float beta = 1.3f;
    const int incy = 1;

    float a[lda*n] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f };
    float x[m*incx] = { 1.f, 2.f, 3.f };
    float y[n*incy] = { 4.f, 5.f, 6.f, 7.f };
    const float expected[n*incy] = { 4 * 1.3f + 1.1f*(2.f + 2.f*3.f),
        5.f*1.3f + 1.1f*(4.f + 2.f*5.f + 3.f*6.f),
        6.f*1.3f + 1.1f*(2.f*7.f + 3.f*8.f),
        7.f*1.3f + 1.1f*10.f*3.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgbmv(handle, trans, m, n, kl, ku, &alpha, a, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incy; i++) {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}
