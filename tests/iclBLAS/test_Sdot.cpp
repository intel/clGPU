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

TEST(Sdot, naive_5x1)
{
    const int num = 5;
    const int incx = 1;
    const int incy = 1;
    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    float y[num*incy] = { 5.f, 4.f, 1.f, 2.f, 5.f };
    float result[] = { 0.0f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSdot(handle, num, x, incx, y, incy, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(49.f, result[0]);
    
}

TEST(Sdot, naive_6x2)
{
    const int num = 3;
    const int incx = 2;
    const int incy = 2;
    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 8.f };
    float y[num*incy] = { 5.f, 4.f, 1.f, 2.f, 5.f, 2.f };
    float result[] = { 0.0f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSdot(handle, num, x, incx, y, incy, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(33.f, result[0]);
}