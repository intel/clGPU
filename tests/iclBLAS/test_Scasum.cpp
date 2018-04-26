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

TEST(Scasum, big_test) {
    const int n = 100000;
    const int incx = 1;

    oclComplex_t x[n * incx];
    float ex_result = 0.f;
    float result = 0;

    for (int i = 0; i < n; ++i)
    {
        x[i] = { 1.f * i / n / n, 1.f * i / n / n };

        ex_result += fabs(x[i].real()) + fabs(x[i].imag());
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScasum(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(ex_result, result, 1.e-5f);
}

TEST(Scasum, opt_small_work) {
    const int n = 9;
    const int incx = 2;

    oclComplex_t x[n * incx];
    float ex_result = 0.f;
    float result = 0;

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = { 1.f * i / n / n, 1.f * i / n / n };
    }

    for (int i = 0; i < n; ++i)
    {
        ex_result += fabs(x[i * incx].real()) + fabs(x[i * incx].imag());
    }


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScasum(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(ex_result, result, 1.e-5f);
}

TEST(Scasum, opt_small_work_v2) {
    const int n = 288;
    const int incx = 1;

    oclComplex_t x[n * incx];
    float ex_result = 0.f;
    float result = 0;

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = { 1.f * i / n / n, 1.f * i / n / n };
    }

    for (int i = 0; i < n; ++i)
    {
        ex_result += fabs(x[i * incx].real()) + fabs(x[i * incx].imag());
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScasum(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(ex_result, result, 1.e-5f);
}

TEST(Scasum, 2stage_v2) {
    const int n = 2048;
    const int incx = 1;

    oclComplex_t x[n * incx];
    float ex_result = 0.f;
    float result = 0;

    for (int i = 0; i < n * incx; ++i)
    {
        x[i] = { 1.f * i / n / n, 1.f * i / n / n };
    }

    for (int i = 0; i < n; ++i)
    {
        ex_result += fabs(x[i * incx].real()) + fabs(x[i * incx].imag());
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScasum(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(ex_result, result, 1.e-5f);
}
