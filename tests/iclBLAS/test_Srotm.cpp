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

TEST(Srotm, n11)
{
    const int n = 11;
    const int incx = 1;
    const int incy = 1;

    float x[n] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f, 14.f };
    float y[n] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f, 14.f };

    float param[5] = { -1.f, -1.f, 3.f, 4.f, 1.f };

    float refx[n];
    float refy[n];

    for (int i = 0; i < n; i++)
    {
        float _x = param[1] * x[i] + param[2] * y[i];
        refy[i] = param[3] * x[i] + param[4] * y[i];
        refx[i] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotm(handle, n, x, incx, y, incy, param);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i], x[i]);
        EXPECT_FLOAT_EQ(refy[i], y[i]);
    }
}

TEST(Srotm, inc2)
{
    const int n = 5;
    const int incx = 2;
    const int incy = 2;

    float x[n * incx] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f };
    float y[n * incy] = { -1.f, 23.f, 3.f, 14.f, 4.f, 8.f, 7.f, -11.f, 9.f, 10.f };

    float param[5] = { -1.f, -1.f, 3.f, 4.f, 1.f };

    float refx[n * incx];
    float refy[n * incy];

    for (int i = 0; i < 2 * n; i++)
    {
        refx[i] = x[i];
        refy[i] = y[i];
    }

    for (int i = 0; i < n; i++)
    {
        float _x = param[1] * x[i * incx] + param[2] * y[i * incy];
        refy[i * incy] = param[3] * x[i * incx] + param[4] * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotm(handle, n, x, incx, y, incy, param);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < 2 * n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i], x[i]);
        EXPECT_FLOAT_EQ(refy[i], y[i]);
    }
}

TEST(Srotm, 11_incx1) {
    const int n = 11;
    const int incx = 1;
    const int incy = 2;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = 1.f * i;
    }
    std::vector<float> y(n*incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = 1.f * n - i;
    }
    float param[5] = { -1.f, -1.f, 3.f, 4.f, 1.f };

    auto expected_x = x;
    auto expected_y = y;
    for (int i = 0; i < n; i++) {
        expected_x[i*incx] = param[1] * x[i*incx] + param[2] * y[i*incy];
        expected_y[i*incy] = param[3] * x[i*incx] + param[4] * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasSrotm(handle, n, x.data(), incx, y.data(), incy, param);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++)
    {
        EXPECT_FLOAT_EQ(expected_x[i], x[i]);
    }
    for (int i = 0; i < n*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected_y[i], y[i]);
    }
}

TEST(Srotm, 11_incy1) {
    const int n = 11;
    const int incx = 2;
    const int incy = 1;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = 1.f * i;
    }
    std::vector<float> y(n*incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = 1.f * n - i;
    }
    float param[5] = { -1.f, -1.f, 3.f, 4.f, 1.f };

    auto expected_x = x;
    auto expected_y = y;
    for (int i = 0; i < n; i++) {
        expected_x[i*incx] = param[1] * x[i*incx] + param[2] * y[i*incy];
        expected_y[i*incy] = param[3] * x[i*incx] + param[4] * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasSrotm(handle, n, x.data(), incx, y.data(), incy, param);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++)
    {
        EXPECT_FLOAT_EQ(expected_x[i], x[i]);
    }
    for (int i = 0; i < n*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected_y[i], y[i]);
    }
}

TEST(Srotm, 11_incs1) {
    const int n = 11;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = 1.f * i;
    }
    std::vector<float> y(n*incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = 1.f * n - i;
    }
    float param[5] = { -1.f, -1.f, 3.f, 4.f, 1.f };

    auto expected_x = x;
    auto expected_y = y;
    for (int i = 0; i < n; i++) {
        expected_x[i*incx] = param[1] * x[i*incx] + param[2] * y[i*incy];
        expected_y[i*incy] = param[3] * x[i*incx] + param[4] * y[i*incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasSrotm(handle, n, x.data(), incx, y.data(), incy, param);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++)
    {
        EXPECT_FLOAT_EQ(expected_x[i], x[i]);
    }
    for (int i = 0; i < n*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected_y[i], y[i]);
    }
}
