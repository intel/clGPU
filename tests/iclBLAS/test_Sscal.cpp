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

TEST(Sscal, naive_5x2)
{
    const int num = 5;
    const int incx = 2;
    float alpha = 1.5f;
    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f };
    float ref_x[num*incx] = { 1.f * 1.5f, 2.f,
                              3.f * 1.5f, 4.f,
                              5.f * 1.5f, 6.f,
                              7.f * 1.5f, 8.f,
                              9.f * 1.5f, 10.f, };
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSscal(handle, num, &alpha, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_FLOAT_EQ(ref_x[i], x[i]);
    }
}


TEST(Sscal, noinc_10x1)
{
    const int num = 10;
    const int incx = 1;
    float alpha = 1.5f;
    float x[num*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f };
    float ref_x[num*incx] = { 1.f * 1.5f, 2.f * 1.5f,
                              3.f * 1.5f, 4.f * 1.5f,
                              5.f * 1.5f, 6.f * 1.5f,
                              7.f * 1.5f, 8.f * 1.5f,
                              9.f * 1.5f, 10.f * 1.5f, };
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSscal(handle, num, &alpha, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_FLOAT_EQ(ref_x[i], x[i]);
    }
}
