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

TEST(Sspmv, Sspmv_up_3x3) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int incx = 1;
    const int incy = 1;

    const int n = 3;
    float alpha = 1.f;
    float beta = 1.f;

    float tmp_AP[n * n] = { 1.f, 2.f, 3.f,
                            2.f, 1.f, 4.f,
                            3.f, 4.f, 1.f };

    float AP[6] = { 1.f, 2.f, 1.f, 3.f, 4.f, 1.f };

    float x[n] = { 1.f, 1.f, 1.f };
    float y[n] = { 1.f, 1.f, 1.f };

    float ref_y[n];

    for (int i = 0; i < n; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += tmp_AP[i * n + k] * x[k];
        }

        ref_y[i] = alpha * value + beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspmv(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_FLOAT_EQ(ref_y[i], y[i]);
    }
}

TEST(Sspmv, Sspmv_up_nxn) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int incx = 1;
    const int incy = 1;

    const int n = 8;
    float alpha = 1.15f;
    float beta = 4.89f;

    float tmp_AP[n * n];
    float AP[(n * (n + 1)) / 2];

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                tmp_AP[i * n + k] = (std::rand() % 10) / 1.45f;
                tmp_AP[k * n + i] = tmp_AP[i * n + k];
            }

    int idx = 0;

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if(k >= i)
                AP[idx++] = tmp_AP[i * n + k];

    float x[n];
    float y[n];

    for (int i = 0; i < n; i++)
    {
        x[i] = i + i / 3.f;
        y[i] = 1.f;
    }

    float ref_y[n];

    for (int i = 0; i < n; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += tmp_AP[i * n + k] * x[k];
        }

        ref_y[i] = alpha * value + beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspmv(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_FLOAT_EQ(ref_y[i], y[i]);
    }
}

TEST(Sspmv, Sspmv_up_nxn_2) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int incx = 1;
    const int incy = 1;

    const int n = 9;
    float alpha = 1.45f;
    float beta = 2.43f;

    float tmp_AP[n * n];
    float AP[(n * (n + 1)) / 2];

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                tmp_AP[i * n + k] = (std::rand() % 10) / 2.95f;
                tmp_AP[k * n + i] = tmp_AP[i * n + k];
            }

    int idx = 0;

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if (k >= i)
                AP[idx++] = tmp_AP[i * n + k];

    float x[n];
    float y[n];

    for (int i = 0; i < n; i++)
    {
        x[i] = i + i / 3.1f;
        y[i] = 1.7f;
    }

    float ref_y[n];

    for (int i = 0; i < n; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += tmp_AP[i * n + k] * x[k];
        }

        ref_y[i] = alpha * value + beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspmv(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_FLOAT_EQ(ref_y[i], y[i]);
    }
}

TEST(Sspmv, Sspmv_low_3x3) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int incx = 2;
    const int incy = 1;

    const int n = 3;
    float alpha = 1.f;
    float beta = 1.f;

    float tmp_AP[n * n] = { 1.f, 2.f, 3.f,
                            2.f, 1.f, 4.f,
                            3.f, 4.f, 1.f };

    float AP[(n * (n + 1)) / 2] = { 1.f, 2.f, 3.f, 1.f, 4.f, 1.f };

    float x[n * incx] = { 1.f, 1.f, .5f, 1.f, 1.f, 1.f };
    float y[n * incy] = { 1.f, 1.f, 1.f };

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += tmp_AP[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspmv(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
    }
}

TEST(Sspmv, Sspmv_low_nxn) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int incx = 1;
    const int incy = 1;

    const int n = 5;
    float alpha = 1.f;
    float beta = 1.f;

    float tmp_AP[n * n];
    float AP[(n * (n + 1)) / 2];

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                tmp_AP[i * n + k] = static_cast<float>(std::rand() % 10);
                tmp_AP[k * n + i] = tmp_AP[i * n + k];
            }

    int idx = 0;

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if (k <= i)
                AP[idx++] = tmp_AP[i * n + k];

    float x[n * incx];
    float y[n * incy];

    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f;
        y[i] = 1.f;
    }

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += tmp_AP[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspmv(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
    }
}

TEST(Sspmv, Sspmv_low_nxn_2) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int incx = 1;
    const int incy = 1;

    const int n = 8;
    float alpha = 1.f;
    float beta = 1.f;

    float tmp_AP[n * n];
    float AP[(n * (n + 1)) / 2];

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (k <= i)
            {
                tmp_AP[i * n + k] = static_cast<float>(std::rand() % 15);
                tmp_AP[k * n + i] = tmp_AP[i * n + k];
            }

    int idx = 0;

    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if (k <= i)
                AP[idx++] = tmp_AP[i * n + k];

    float x[n * incx];
    float y[n * incy];

    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.35f;
        y[i] = 2.90f;
    }

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += tmp_AP[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspmv(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
    }
}
