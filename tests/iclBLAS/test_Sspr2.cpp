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

/* Access for Upper symmetric matrix (packed) coordinates from normal matrix coordinates */
#define US_ACCESS(A, i, j, N) A[i+(j*(j+1))/2]

/* Access for Lower symmetric matrix (packed) coordinates from normal matrix coordinates */
#define LS_ACCESS(A, i, j, N) A[(i+((2*N-j+1)*j)/2) - (1*j)]

TEST(Sspr2, naive_test_upper) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;

    int n = 4;
    int incx = 1;
    int incy = 1;
    float alpha = 1.f;

    std::vector<float> AP(n * (n + 1) / 2);
    for (int i = 0; i < n * (n + 1) / 2; i++)
    {
        AP[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 1.f * i / n / n;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        x[i*incy] = 1.f * i / n / n;
    }

    /* CPU Sspr2 [Upper] */
    std::vector<float> ex_result(n * (n + 1) / 2);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float res_1 = alpha * x[i] * y[j];
            float res_2 = alpha * y[i] * x[j];

            if (j >= i)
                US_ACCESS(ex_result, i, j, n) = res_1 + res_2 + US_ACCESS(AP, i, j, n);
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspr2(handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, AP.data());
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * (n + 1) / 2; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i], AP[i]);
    }
}

TEST(Sspr2, naive_test_lower) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;

    int n = 4;
    int incx = 1;
    int incy = 1;
    float alpha = 1.f;

    std::vector<float> AP(n * (n + 1) / 2);
    for (int i = 0; i < n * (n + 1) / 2; i++)
    {
        AP[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 1.f * i / n / n;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        x[i*incy] = 1.f * i / n / n;
    }

    /* CPU Sspr2 [Lower] */
    std::vector<float> ex_result(n * (n + 1) / 2);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float res_1 = alpha * x[i] * y[j];
            float res_2 = alpha * y[i] * x[j];

            if (i >= j)
                LS_ACCESS(ex_result, i, j, n) = res_1 + res_2 + LS_ACCESS(AP, i, j, n);
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSspr2(handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, AP.data());
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * (n + 1) / 2; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i], AP[i]);
    }
}

#undef US_ACCESS
#undef LS_ACCESS
