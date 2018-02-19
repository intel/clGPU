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

TEST(Ssyr2k, 3x4_up_ntrans) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_N;
    const int n = 3;
    const int k = 4;
    float alpha = 1.1f;
    float beta = 1.3f;

    const int lda = 3;
    float A[lda*k] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f };
    const int ldb = 3;
    float B[ldb*k] = { 13.f, 14.f, 15.f,
                       16.f, 17.f, 18.f,
                       19.f, 20.f, 21.f,
                       22.f, 23.f, 24.f };
    const int ldc = 3;
    float C[ldc*n] = { 1.f, -1.f, -2.f,
                       2.f, 3.f, -3.f,
                       4.f, 5.f, 6.f };

    const float expected[ldc*n] = { 1.3f*1.f + 1.1f*860.f, -1.f, -2.f,
                                    1.3f*2.f + 1.1f*952.f, 1.3f*3.f + 1.1f*1052.f, -3.f,
                                    1.3f*4.f + 1.1f*1044.f, 1.3f*5.f + 1.1f*1152.f, 1.3f*6.f + 1.1f*1260.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr2k(handle, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

TEST(Ssyr2k, 3x3_low_ntrans) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_N;
    const int n = 3;
    const int k = 3;
    float alpha = -1.1f;
    float beta = 1.3f;

    const int lda = 4;
    float A[lda*k] = { 1.f, 2.f, 3.f, -1.f,
        4.f, 5.f, 6.f, -2.f,
        7.f, 8.f, 9.f, -3.f };
    const int ldb = 3;
    float B[ldb*k] = { 10.f, 11.f, 12.f,
                       13.f, 14.f, 15.f,
                       16.f, 17.f, 18.f };
    const int ldc = 4;
    float C[ldc*n] = { 1.f, 2.f, 3.f, -1.f,
                      -2.f, 4.f, 5.f, -3.f,
                      -4.f, -5.f, 6.f, -6.f };

    const float expected[ldc*n] = { 1.3f*1.f - 1.1f*348.f, 1.3f*2.f - 1.1f*399.f, 1.3f*3.f - 1.1f*450.f, -1.f,
                                   -2.f, 1.3f*4.f - 1.1f*456.f, 1.3f*5.f - 1.1f*513.f, -3.f,
                                   -4.f, -5.f, 1.3f*6.f - 1.1f*576.f, -6.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr2k(handle, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

TEST(Ssyr2k, 4x3_up_trans) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const auto trans = ICLBLAS_OP_T;
    const int n = 4;
    const int k = 3;
    float alpha = 1.1f;
    float beta = 1.3f;

    const int lda = 3;
    float A[lda*n] = { 1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f };
    const int ldb = 3;
    float B[ldb*n] = { 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f,
        19.f, 20.f, 21.f,
        22.f, 23.f, 24.f };
    const int ldc = 4;
    float C[ldc*n] = { 1.f, -1.f, -2.f, -3.f,
                       2.f, 3.f, -4.f, -5.f,
                       4.f, 5.f, 6.f, -6.f,
                       7.f, 8.f, 9.f, 10.f };

    const float expected[ldc*n] = { 1.3f*1.f + 1.1f*172.f, -1.f, -2.f, -3.f,
        1.3f*2.f + 1.1f*316.f, 1.3f*3.f + 1.1f*514.f, -4.f, -5.f,
        1.3f*4.f + 1.1f*460.f, 1.3f*5.f + 1.1f*712.f, 1.3f*6.f + 1.1f*964.f, -6.f,
        1.3f*7.f + 1.1f*604.f, 1.3f*8.f + 1.1f*910.f, 1.3f*9.f + 1.1f*1216.f, 1.3f*10.f + 1.1f*1522.f};

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr2k(handle, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

TEST(Ssyr2k, 3x3_low_trans) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const auto trans = ICLBLAS_OP_C;
    const int n = 3;
    const int k = 3;
    float alpha = -1.1f;
    float beta = 0.f;

    const int lda = 3;
    float A[lda*n] = { 1.f, 2.f, 3.f,
                       4.f, 5.f, 6.f,
                       7.f, 8.f, 9.f };
    const int ldb = 3;
    float B[ldb*n] = { 10.f, 11.f, 12.f,
                       13.f, 14.f, 15.f,
                       16.f, 17.f, 18.f };
    const int ldc = 3;
    float C[ldc*n] = { -1.f, -2.f, -3.f,
                       -4.f, -5.f, -6.f,
                       -7.f, -8.f, -9.f };

    const float expected[ldc*n] = { -1.1f*136.f, -1.1f*253.f, -1.1f*370.f,
                                    -4.f, -1.1f*424.f, -1.1f*595.f,
                                    -7.f, -8.f, -1.1f*820.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr2k(handle, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}
