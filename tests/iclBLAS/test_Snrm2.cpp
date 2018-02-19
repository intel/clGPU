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
#include <cmath>

TEST(Snrm2, naive_11) {
    const int n = 11;
    const int incx = 1;
    float result = -1.f;
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f,
                        6.f, 7.f, 8.f, 9.f, 10.f, 11.f };

    float expected = std::sqrt(1.f*1.f + 2.f*2.f + 3.f*3.f + 4.f*4.f + 5.f*5.f
                          + 6.f*6.f + 7.f*7.f + 8.f*8.f + 9.f*9.f
                          + 10.f*10.f + 11.f*11.f);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Snrm2, naive_11_0s) {
    const int n = 11;
    const int incx = 1;
    float result = -1.f;
    float x[n*incx] = { 0.f, 0.f, 0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    float expected = 0.f;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Snrm2, naive_257n) {
    const int n = 257;
    const int incx = 1;
    float result = -1.f;
    float x[n*incx];
    for (int i = 0; i<n; i++) {
        x[i] = (float)i;
    }

    const float expected = 2371.7538f;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x, incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Snrm2, 1025_inc2) {
    const int n = 1025;
    const int incx = 2;
    float result = -1.f;
    std::vector<float> x(n*incx);
    for (int i = 0; i<n; i++) {
        x[i*incx] = (float)i / n / n;
    }

    float expected = 0.f;
    for (int i = 0; i < n; i++) {
        expected += x[i*incx] * x[i*incx];
    }
    expected = sqrtf(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_NEAR(expected, result, 1.e-5f);
}

TEST(Snrm2, 1025_inc1) {
    const int n = 1025;
    const int incx = 1;
    float result = -1.f;
    std::vector<float> x(n*incx);
    for (int i = 0; i<n; i++) {
        x[i*incx] = (float)i / n / n;
    }

    float expected = 0.f;
    for (int i = 0; i < n; i++) {
        expected += x[i*incx] * x[i*incx];
    }
    expected = sqrtf(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_NEAR(expected, result, 1.e-5f);
}

TEST(Snrm2, 65537_inc2) {
    const int n = 65537;
    const int incx = 2;
    float result = -1.f;
    std::vector<float> x(n*incx);
    for (int i = 0; i<n; i++) {
        x[i*incx] = (float)i / n / n;
    }

    float expected = 0.f;
    for (int i = 0; i < n; i++) {
        expected += x[i*incx] * x[i*incx];
    }
    expected = sqrtf(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_NEAR(expected, result, 1.e-5f);
}

TEST(Snrm2, 65537_inc1) {
    const int n = 65537;
    const int incx = 1;
    float result = -1.f;
    std::vector<float> x(n*incx);
    for (int i = 0; i<n; i++) {
        x[i*incx] = (float)i / n / n;
    }

    float expected = 0.f;
    for (int i = 0; i < n; i++) {
        expected += x[i*incx] * x[i*incx];
    }
    expected = sqrtf(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSnrm2(handle, n, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    EXPECT_NEAR(expected, result, 1.e-5f);
}
