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

TEST(Isamin, naive)
{
    const int n = 512;
    const int incx = 1;

    float x[n * incx];

    for (int i = 0; i < n; ++i)
    {
        x[i] = i + 100.f;
    }

    x[55] = -50.f;

    int result[1] = { 0 };
    int ref = 55;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Isamin, naive_incx)
{
    const int n = 40960;
    const int incx = 2;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = i + 100.f;
    }

    x[56] = -25.f; 
    x[55] = -10.f; //Should omitt this lowest value as incx is 2
    x[23] = -10.f; //Should omitt this lowest value as incx is 2
    x[89] = -10.f; //Should omitt this lowest value as incx is 2

    int result[1] = { 0 };
    int ref = 28; // Normally it should be 56, but we divide that value by incx (which equals 2)


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Isamin, opt_2stage)
{
    const int n = 80000;
    const int incx = 1;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = i + 100.f;
    }

    x[55] = -50.f;

    int result[1] = { 0 };
    int ref = 55;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Isamin, opt_2stage_incx)
{
    const int n = 80000;
    const int incx = 2;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = i + 100.f;
    }

    x[55] = -50.f; //Should omitt this lowest value as incx is 2
    x[57] = -50.f; //Should omitt this lowest value as incx is 2
    x[107] = -10.f; //Should omitt this lowest value as incx is 2
    x[1001] = -10.f; //Should omitt this lowest value as incx is 2
    x[88] = -60.f;

    int result[1] = { 0 };
    int ref = 88/incx;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}
