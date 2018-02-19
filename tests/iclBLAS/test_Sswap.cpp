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

TEST(Sswap, n0) {
    const int num = 0;
    const int incx = 1;
    const int incy = 1;

    float x[] = { 1.f, 2.f, 3.f,
                  4.f, 5.f, 6.f,
                  7.f, 8.f, 9.f };

    float expected_x[] = { 1.f, 2.f, 3.f,
                           4.f, 5.f, 6.f,
                           7.f, 8.f, 9.f };

    float y[] = { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
                  6.5f, 7.5f, 8.5f, 9.5f, 10.5f,
                  11.5f, 12.5f, 13.5f, 14.5f, 15.5f };

    float expected_y[] = { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
                           6.5f, 7.5f, 8.5f, 9.5f, 10.5f,
                           11.5f, 12.5f, 13.5f, 14.5f, 15.5f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSswap(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected_x[i], x[i]);
        EXPECT_FLOAT_EQ(expected_y[i], y[i]);
    }
}

TEST(Sswap, 5_incx1_incy2) {
    const int num = 5;
    const int incx = 1;
    const int incy = 2;

    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    
    float y[num*incy+1] = { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
                            6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f};

    float expected_x[num*incx] = { 1.5f, 3.5f, 5.5f, 7.5f, 9.5f };

    float expected_y[num*incy + 1] = { 1.f, 2.5f, 2.f, 4.5f, 3.f,
                           6.5f, 4.f, 8.5f, 5.f, 10.5f, 11.5f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSswap(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected_x[i], x[i]);
        EXPECT_FLOAT_EQ(expected_y[i], y[i]);
    }
}

TEST(Sswap, 11_incs1) {
    const int num = 11;
    const int incx = 1;
    const int incy = 1;

    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
        7.f, 8.f, 9.f, 10.f, 11.f };

    float y[num*incy] = { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
        6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f };

    float expected_x[num*incx] = { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
        6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f };

    float expected_y[num*incy] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
        7.f, 8.f, 9.f, 10.f, 11.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSswap(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_FLOAT_EQ(expected_x[i], x[i]);
        EXPECT_FLOAT_EQ(expected_y[i], y[i]);
    }
}
