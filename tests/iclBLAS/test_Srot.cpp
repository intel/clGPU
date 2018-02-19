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

TEST(Srot, n1_c2_s1)
{
    const int n = 1;
    const int incx = 1;
    const int incy = 1;

    float x[n] = { 1.f };
    float y[n] = { 2.f };

    float c = 2;
    float s = 1;

    float refx = 4.f;
    float refy = 3.f;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x, incx, y, incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(refx, x[0]);
    EXPECT_FLOAT_EQ(refy, y[0]);
}

TEST(Srot, n11_c2_s1)
{
    const int n = 11;
    const int incx = 1;
    const int incy = 1;

    float x[n * incx] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f, 14.f };
    float y[n * incy] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f, 14.f };

    float c = 2;
    float s = 1;

    float refx[n * incx];
    float refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        float _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1 * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x, incx, y, incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx], x[i * incx]);
        EXPECT_FLOAT_EQ(refy[i * incy], y[i * incy]);
    }
}

TEST(Srot, n5x2_c2_s1)
{
    const int n = 5;
    const int incx = 2;
    const int incy = 2;

    float x[n * incx] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f };
    float y[n * incy] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f };

    float c = 2;
    float s = 1;

    float refx[n * incx];
    float refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        refx[i * incx] = x[i * incx];
        refy[i * incy] = y[i * incy];
    }

    for (int i = 0; i < n; i++)
    {
        float _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1 * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x, incx, y, incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx], x[i * incx]);
        EXPECT_FLOAT_EQ(refy[i * incy], y[i * incy]);
    }
}

TEST(Srot, noincx)
{
    const int n = 12;
    const int incx = 1;
    const int incy = 2;

    float x[n * incx];
    float y[n * incy];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = static_cast<float>(std::rand() % 23);
        y[i * incy] = static_cast<float>(std::rand() % 23);
    }

    float c = 2;
    float s = 1;

    float refx[n * incx];
    float refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        refx[i * incx] = x[i * incx];
        refy[i * incy] = y[i * incy];
    }

    for (int i = 0; i < n; i++)
    {
        float _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1 * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x, incx, y, incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx], x[i * incx]);
        EXPECT_FLOAT_EQ(refy[i * incy], y[i * incy]);
    }
}

TEST(Srot, noincy)
{
    const int n = 12;
    const int incx = 2;
    const int incy = 1;

    float x[n * incx];
    float y[n * incy];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = static_cast<float>(std::rand() % 23);
        y[i * incy] = static_cast<float>(std::rand() % 23);
    }

    float c = 2.25f;
    float s = 0.75f;

    float refx[n * incx];
    float refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        refx[i * incx] = x[i * incx];
        refy[i * incy] = y[i * incy];
    }

    for (int i = 0; i < n; i++)
    {
        float _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1 * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x, incx, y, incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx], x[i * incx]);
        EXPECT_FLOAT_EQ(refy[i * incy], y[i * incy]);
    }
}

TEST(Srot, noinc_optim)
{
    const int n = 65536;
    const int incx = 1;
    const int incy = 1;

    //use vector to avoid stack size limit problem
    std::vector<float> x(n * incx);
    std::vector<float> y(n * incy);

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = static_cast<float>(std::rand() % 15);
        y[i * incy] = static_cast<float>(std::rand() % 15);
    }

    float c = 2;
    float s = 1;

    std::vector<float> refx(n * incx);
    std::vector<float> refy(n * incy);

    for (int i = 0; i < n; i++)
    {
        float _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1 * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x.data(), incx, y.data(), incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx], x[i * incx]);
        EXPECT_FLOAT_EQ(refy[i * incy], y[i * incy]);
    }
}

TEST(Srot, noincx_optim)
{
    const int n = 16384;
    const int incx = 1;
    const int incy = 3;

    //use vector to avoid stack size limit problem
    std::vector<float> x(n * incx);
    std::vector<float> y(n * incy);

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = static_cast<float>(std::rand() % 15);
        y[i * incy] = static_cast<float>(std::rand() % 15);
    }

    float c = 2.4f;
    float s = 0.25f;

    std::vector<float> refx(n * incx);
    std::vector<float> refy(n * incy);

    for (int i = 0; i < n; i++)
    {
        float _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1 * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrot(handle, n, x.data(), incx, y.data(), incy, c, s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx], x[i * incx]);
        EXPECT_FLOAT_EQ(refy[i * incy], y[i * incy]);
    }
}
