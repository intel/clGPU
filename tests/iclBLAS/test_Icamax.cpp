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

TEST(Icamax, inc_1)
{
    const int n = 11;
    const int incx = 1;

    oclComplex_t x[n * incx] = { {-1.f, 0.f}, {23.f, 0.f}, {3.f, 0.f}, {14.f, 0.f}, {4.f, 0.f}, {8.f, 0.f}, {7.f, 0.f}, {-11.f, 0.f}, {9.f, 0.f}, {10.f, 0.f}, {14.f, 0.f} };

    int res = 0;
    int ref = 1;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIcamax(handle, n, x, incx, &res);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, res);
}

TEST(Icamax, inc_2)
{
    const int n = 128;
    const int incx = 2;

    oclComplex_t x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i].val[0] = i + 100.f;
        x[i].val[1] = i + 100.f;
    }

    x[55].val[0] = 1000.f; //Should omitt this value as incx is 2
    x[55].val[1] = 1000.f; //Should omitt this value as incx is 2

    x[56].val[0] = 1000.f;
    x[56].val[1] = 1000.f;

    x[60].val[0] = 1000.f; //Should omitt
    x[60].val[1] = 1000.f; //Should omitt

    x[64].val[0] = 1000.f; //Should omitt
    x[64].val[1] = 1000.f; //Should omitt

    int result[1] = { 0 };
    int ref = 56;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIcamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Icamax, opt_2stage_test)
{
    const int n = 1000;
    const int incx = 1;

    oclComplex_t x[n];

    for (int i = 0; i < n; ++i)
    {
        x[i].val[0] = static_cast<float>(i);
        x[i].val[1] = static_cast<float>(i);
    }

    x[55].val[0] = 10000.f;
    x[55].val[1] = -10000.f;

    x[60].val[0] = 10000.f; //Should omitt
    x[60].val[1] = -10000.f; //Should omitt

    x[64].val[0] = 10000.f; //Should omitt
    x[64].val[1] = -10000.f; //Should omitt

    int result[1] = { 0 };
    int ref = 55;



    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIcamax(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}
