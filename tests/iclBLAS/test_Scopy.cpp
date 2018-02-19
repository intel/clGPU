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

TEST(Scopy, naive_5x2)
{
    const int num = 5;
    const int incx = 2;
    const int incy = 2;
    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f };
    float y[num*incy] = { 11.f, 22.f, 33.f, 44.f, 55.f, 66.f, 77.f, 88.f, 99.f, 110.f, };
    float ref_y[num*incy] = { 1.f, 22.f, 3.f, 44.f, 5.f, 66.f, 7.f, 88.f, 9.f, 110.f, };
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScopy(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for(int i = 0; i < num*incy; ++i)
    {
        EXPECT_FLOAT_EQ(ref_y[i], y[i]);
    }
}
