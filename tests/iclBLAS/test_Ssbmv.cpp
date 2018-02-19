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

TEST(Ssbmv, Ssbmv_up_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;
    const int incy = 1;
    float alpha = 1.f;
    float beta = 1.f;

    float x[n * incx] = { 1.f, 1.f, 1.f };
    float y[n * incy] = { 1.f, 1.f, 1.f };
    float ref_A[n * n] = { 1.f, 2.f, 0.f,
                            2.f, 1.f, 2.f,
                            0.f, 2.f, 1.f };
    float A[n * lda] = { 0.f, 1.f, 2.f, 1.f, 2.f, 1.f };

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += ref_A[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsbmv(handle, uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
}

TEST(Ssbmv, Ssbmv_low_3x3)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 3;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;
    const int incy = 2;
    float alpha = 1.f;
    float beta = 1.f;

    float x[n * incx] = { 1.f, 1.f, 1.f };
    float y[n * incy] = { 1.f, 0.f, 1.f, 0.f, 1.f, 0.f };
    float ref_A[n * n] = { 1.f, 2.f, 0.f,
                             2.f, 1.f, 2.f,
                             0.f, 2.f, 1.f };
    float A[n * lda] = { 1.f, 2.f, 1.f, 2.f, 1.f, 0.f };

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += ref_A[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsbmv(handle, uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
}

TEST(Ssbmv, Ssbmv_up_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 4;
    const int k = 1;
    const int lda = 3;

    const int incx = 1;
    const int incy = 1;
    float alpha = 1.15f;
    float beta = .5f;

    float x[n * incx] = { 1.45f, 2.9f, 0.f };
    float y[n * incy] = { 1.5f, 0.f, 1.5f };
    float ref_A[n * n] = { 4.f, 1.4f, 0.f, 0.f,
                             1.4f, 1.f, .7f, 0.f,
                             0.f, .7f, 1.85f, 1.f,
                             0.f, 0.f, 1.f, 3.5f };
    float A[n * lda] = { 0.f, 4.f, 1.f, 1.4f, 1.f, 1.f, .7f, 1.85f, 1.f, 1.f, 3.5f, 1.f };

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += ref_A[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsbmv(handle, uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
}

TEST(Ssbmv, Ssbmv_low_4x4)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 4;
    const int k = 1;
    const int lda = 2;

    const int incx = 1;
    const int incy = 1;
    float alpha = 1.15f;
    float beta = .5f;

    float x[n] = { 1.35f, 2.f, 1.35f };
    float y[n] = { 1.5f, 0.f, 1.5f };
    float ref_A[n * n] = { 1.2f, 1.4f, 0.f, 0.f,
                             1.4f, 1.f, .7f, 0.f,
                             0.f, .7f, 1.85f, 1.f,
                             0.f, 0.f, 1.f, 3.5f };
    float A[n * lda] = { 1.2f, 1.4f, 1.f, .7f, 1.85f, 1.f, 3.5f, 0.f };

    float ref_y[n * incy];

    for (int i = 0; i < n * incy; i++)
        ref_y[i] = 0.f;

    for (int i = 0; i < n; i++)
    {
        float value = 0.f;

        for (int k = 0; k < n; k++)
        {
            value += ref_A[i * n + k] * x[k * incx];
        }

        ref_y[i * incy] = alpha * value + beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsbmv(handle, uplo, n, k, &alpha, A, lda, x, incx, &beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
        ASSERT_FLOAT_EQ(ref_y[i * incy], y[i * incy]);
}
