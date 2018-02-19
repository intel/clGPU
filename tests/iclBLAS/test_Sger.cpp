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

TEST(Sger, naive_3x5) {
    const int m = 3;
    const int n = 5;
    const int lda = 3;
    const int incx = 2;
    const int incy = 1;
    float alpha = 1.1f;
    float x[m*incx] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float y[n*incy] = {1.f, 2.f, 3.f, 4.f, 5.f};
    float a[lda*n] = { -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f };

    const float expected[lda*n] = { -1.f+1.1f*1.f, -2.f+1.1f*3.f, -3.f+1.1f*5.f,
                                    -1.f+1.1f*2.f, -2.f+1.1f*6.f, -3.f+1.1f*10.f,
                                    -1.f+1.1f*3.f, -2.f+1.1f*9.f, -3.f+1.1f*15.f,
                                    -1.f+1.1f*4.f, -2.f+1.1f*12.f, -3.f+1.1f*20.f,
                                    -1.f+1.1f*5.f, -2.f+1.1f*15.f, -3.f+1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x, incx, y, incy, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda*n; i++) {
        ASSERT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Sger, no_inc_3x5) {
    const int m = 3;
    const int n = 5;
    const int lda = 3;
    const int incx = 1;
    const int incy = 1;
    float alpha = 1.1f;
    float x[m*incx] = { 1.f, 3.f, 5.f };
    float y[n*incy] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    float a[lda*n] = { -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f };

    const float expected[lda*n] = { -1.f + 1.1f*1.f, -2.f + 1.1f*3.f, -3.f + 1.1f*5.f,
        -1.f + 1.1f*2.f, -2.f + 1.1f*6.f, -3.f + 1.1f*10.f,
        -1.f + 1.1f*3.f, -2.f + 1.1f*9.f, -3.f + 1.1f*15.f,
        -1.f + 1.1f*4.f, -2.f + 1.1f*12.f, -3.f + 1.1f*20.f,
        -1.f + 1.1f*5.f, -2.f + 1.1f*15.f, -3.f + 1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x, incx, y, incy, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda*n; i++) {
        ASSERT_FLOAT_EQ(expected[i], a[i]);
    }
}
