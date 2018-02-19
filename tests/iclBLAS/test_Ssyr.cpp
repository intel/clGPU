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


TEST(Ssyr, naive_6x5_up) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 5;
    const int lda = 6;
    const int incx = 1;
    float alpha = 1.1f;
    float a[lda*n] = { 1.f, -1.f, -2.f, -3.f, -4.f, -5.f,
                       2.f, 3.f, -6.f, -7.f, -8.f, -9.f,
                       4.f, 5.f, 6.f, -10.f, -11.f, -12.f,
                       7.f, 8.f, 9.f, 10.f, -13.f, -14.f,
                       11.f, 12.f, 13.f, 14.f, 15.f, -15.f };
    float x[n*incx] = {1.f, 2.f, 3.f, 4.f, 5.f};
    const float expected[lda*n] = { 1.f + 1.1f, -1.f, -2.f, -3.f, -4.f, -5.f,
                                    2.f+1.1f*2.f, 3.f+1.1f*4.f, -6.f, -7.f, -8.f, -9.f,
                                    4.f+1.1f*3.f, 5.f+1.1f*6.f, 6.f+1.1f*9.f, -10.f, -11.f, -12.f,
                                    7.f+1.1f*4.f, 8.f+1.1f*8.f, 9.f+1.1f*12.f, 10.f+1.1f*16.f, -13.f, -14.f,
                                    11.f+1.1f*5.f, 12.f+1.1f*10.f, 13.f+1.1f*15.f, 14.f+1.1f*20.f, 15.f+1.1f*25.f, -15.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x, incx, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, naive_5x5_low) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 5;
    const int lda = 5;
    const int incx = 1;
    float alpha = 1.1f;
    float a[lda*n] = { 1.f, 2.f, 3.f, 4.f, 5.f,
                      -1.f, 6.f, 7.f, 8.f, 9.f,
                      -2.f, -3.f, 10.f, 11.f, 12.f,
                      -4.f, -5.f, -6.f, 13.f, 14.f,
                      -7.f, -8.f, -9.f, -10.f, 15.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    const float expected[lda*n] = { 1.f+1.1f*1.f, 2.f+1.1f*2.f, 3.f+1.1f*3.f, 4.f+1.1f*4.f, 5.f+1.1f*5.f,
                                   -1.f, 6.f+1.1f*4.f, 7.f+1.1f*6.f, 8.f+1.1f*8.f, 9.f+1.1f*10.f,
                                   -2.f, -3.f, 10.f+1.1f*9.f, 11.f+1.1f*12.f, 12.f+1.1f*15.f,
                                   -4.f, -5.f, -6.f, 13.f+1.1f*16.f, 14.f+1.1f*20.f,
                                   -7.f, -8.f, -9.f, -10.f, 15.f+1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x, incx, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, naive_3x3_low_inc2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 3;
    const int lda = 3;
    const int incx = 2;
    float alpha = 1.1f;
    float a[lda*n] = { 1.f, 2.f, 3.f,
                      -1.f, 4.f, 5.f,
                      -2.f, -3.f, 6.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    const float expected[lda*n] = { 1.f+1.1f*1.f, 2.f+1.1f*3.f, 3.f+1.1f*5.f,
                                   -1.f, 4.f+1.1f*9.f, 5.f+1.1f*15.f,
                                   -2.f, -3.f, 6.f+1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x, incx, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}
