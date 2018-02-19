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

TEST(Icamin, naive)
{
    const int n = 512;
    const int incx = 1;

    oclComplex_t x[n * incx];

    for (int i = 0; i < n; ++i)
    {
        x[i].val[0] = i + 100.f;
        x[i].val[1] = i + 100.f;
    }

    x[55].val[0] = -55.f;
    x[55].val[1] = -55.f;

    x[88].val[0] = -55.f;
    x[88].val[1] = -55.f;

    x[99].val[0] = -55.f;
    x[99].val[1] = -55.f;

    int result[1] = { 0 };
    int ref = 55;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIcamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Icamin, naive_incx)
{
    const int n = 4096;
    const int incx = 2;

    oclComplex_t x[n * incx];

    for (int i = 0; i < n * incx; ++i)
    {
        x[i].val[0] = i + 100.f;
        x[i].val[1] = i + 100.f;
    }

    x[55].val[0] = -25.f; //Should omitt this lowest value as incx is 2
    x[55].val[1] = -25.f; //Should omitt this lowest value as incx is 2

    x[56].val[0] = -55.f;
    x[56].val[1] = -55.f;

    x[88].val[0] = -55.f; 
    x[88].val[1] = -55.f;

    x[99].val[0] = -55.f; //Should omitt this lowest value as incx is 2
    x[99].val[1] = -55.f; //Should omitt this lowest value as incx is 2

    int result[1] = { 0 };
    int ref = 56;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIcamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);
}

TEST(Icamin, opt_2stage_test)
{
    const int n = 10001;
    const int incx = 1;

    oclComplex_t x[n];

    for (int i = 0; i < n; ++i)
    {
        x[i].val[0] = i + 100.f;
        x[i].val[1] = i + 100.f;
    }

    x[55].val[0] = -20.f;
    x[55].val[1] = -20.f;

    x[1111].val[0] = -20.f;
    x[1111].val[1] = -20.f;

    x[10000].val[0] = -20.f;
    x[10000].val[1] = -20.f;

    int result[1] = { 0 };
    int ref = 55;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasIcamin(handle, n, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(ref, *result);

}
