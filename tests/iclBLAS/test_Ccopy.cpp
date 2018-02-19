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

TEST(Ccopy, n11_noinc)
{
    const int n = 11;
    const int incx = 1;
    const int incy = 1;

    oclComplex_t x[n * incx] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f },{ 4.f, 4.f },
                                { 5.f, 5.f },{ 6.f, 6.f },{ 7.f, 7.f },{ 8.f, 8.f },
                                { 9.f, 9.f },{ 10.f, 10.f },{ 11.f, 11.f } };

    oclComplex_t y[n * incy];
    oclComplex_t ref_y[n * incy] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f },{ 4.f, 4.f },
                                    { 5.f, 5.f },{ 6.f, 6.f },{ 7.f, 7.f },{ 8.f, 8.f },
                                    { 9.f, 9.f },{ 10.f, 10.f },{ 11.f, 11.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCcopy(handle, n, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_y[i * incy].val[0], y[i * incy].val[0]);
        EXPECT_FLOAT_EQ(ref_y[i * incy].val[1], y[i * incy].val[1]);
    }
}

TEST(Ccopy, n_noinc)
{
    const int n = 300;
    const int incx = 1;
    const int incy = 1;

    oclComplex_t x[n * incx];
    oclComplex_t y[n * incy];
    oclComplex_t ref_y[n * incy];

    for (int i = 0; i < n * incx; i++)
        x[i] = { static_cast<float>(std::rand() % 10), static_cast<float>(std::rand() % 10) };

    for (int i = 0; i < n; i++)
        ref_y[i * incy] = x[i * incx];

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCcopy(handle, n, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_y[i * incy].val[0], y[i * incy].val[0]);
        EXPECT_FLOAT_EQ(ref_y[i * incy].val[1], y[i * incy].val[1]);
    }
}

TEST(Ccopy, n_inc)
{
    const int n = 300;
    const int incx = 2;
    const int incy = 3;

    oclComplex_t x[n * incx];
    oclComplex_t y[n * incy];
    oclComplex_t ref_y[n * incy];

    for (int i = 0; i < n * incx; i++)
        x[i] = { static_cast<float>(std::rand() % 10), static_cast<float>(std::rand() % 10) };

    for (int i = 0; i < n; i++)
        ref_y[i * incy] = x[i * incx];

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCcopy(handle, n, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(ref_y[i * incy].val[0], y[i * incy].val[0]);
        EXPECT_FLOAT_EQ(ref_y[i * incy].val[1], y[i * incy].val[1]);
    }
}
