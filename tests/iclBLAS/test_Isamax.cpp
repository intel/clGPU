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

TEST(Isamax, 256)
{
    const int n = 256;
    const int incx = 1;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i);
    }

    int result[1] = { 0 };

    x[47] = 666.f;
    int ex_res = 47;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ex_res, *result);
}

TEST(Isamax, 256_incx2)
{
    const int n = 256;
    const int incx = 2;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i);
    }

    int result[1] = { 0 };

    x[47] = 777.f; //Should ommit because of incx is 2
    x[48] = 666.f;
    int ex_res = 48 / incx;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ex_res, *result);
}

TEST(Isamax, inc_1)
{
    const int n = 11;
    const int incx = 1;

    float x[n * incx] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f, 14.f };
    
    int zeroValue = 0;
    int *res = &zeroValue;
    int ref = 1;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, res);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *res);
}

TEST(Isamax, inc2)
{
    const int n = 1000;
    const int incx = 2;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i);
    }

    int result[1] = { 0 };

    x[66] = 3000;
    x[67] = 3000;
    x[68] = 3000;
    int ex_res = 33; // Normally it should be 66, but we divide that value by incx (which equals 2)

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ex_res, *result);
}

TEST(Isamax, opt1_test)
{
    const int n = 1024;
    const int incx = 1;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i)/n/n;
    }

    x[55] = -22222.f;
    x[56] = -22222.f;
    x[58] = -22222.f;
    x[111] = -22222.f;
    x[222] = -22222.f;

    int result[1] = { 0 };
    int ref = 55;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Isamax, two_stage_test)
{
    const int n = 65537;
    const int incx = 1;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i) / n / n;
    }

    x[55] = 6553777;
    x[58] = 6553777;
    x[55] = 6553777;
    x[55] = 6553777;


    int result[1] = { 0 };
    int ref = 55;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Isamax, two_stage_test_incx)
{
    const int n = 80000;
    const int incx = 2;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i) / n / n;
    }

    x[5555] = 6553777; //Should ommit, incx = 2
    x[5556] = 6553778; 
    x[5557] = 6553777; //Should ommit, also highest value but higher index


    int result[1] = { 0 };
    int ref = 5556 /incx;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Isamax, two_stage_test_incx_neg)
{
    const int n = 80000;
    const int incx = 1;

    float x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = static_cast<float>(i) / n / n;
    }

    x[5555] = -6553777; //Should ommit, incx = 2
    x[5556] = -6553778;
    x[5557] = -6553777; //Should ommit, also highest value but higher index


    int result[1] = { 0 };
    int ref = 5556 / incx;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIsamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}
