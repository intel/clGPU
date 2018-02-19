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

TEST(Sasum, naive_5x1) {
    const int num = 5;
    const int incx = 1;
    float x[num*incx] = { 1.f, 1.3f, 1.6f, 2.5f, 8.2f };
    float result[] = { 0.0f };
    
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);


    EXPECT_FLOAT_EQ(14.6f, result[0]);
}

TEST(Sasum, naive_5x1_v2) {
    const int num = 5;
    const int incx = 1;
    float x[num*incx] = { -1.f, 1.3f, -1.6f, -2.5f, 8.2f };
    float result[] = { 0.0f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);


    EXPECT_FLOAT_EQ(14.6f, result[0]);
}

TEST(Sasum, naive_6x2) {
    const int num = 3;
    const int incx = 2;
    float x[num*incx] = { -1.f, 1.3f, -1.6f, -2.5f, 8.2f, 4.5f };
    float result[] = { 0.0f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x, incx, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);


    EXPECT_FLOAT_EQ(10.8f, result[0]);
}

TEST(Sasum, 256_inc2) {
    const int num = 256;
    const int incx = 2;

    std::vector<float> x(num*incx);
    for (int i = 0; i < num; i++) {
        float this_a = (float)i;
        if (i % 3 == 0) this_a = -this_a;
        x[i*incx] = this_a;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto a : x) {
        expected += std::abs(a);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(result, expected);
}

TEST(Sasum, 256_noinc) {
    const int num = 256;
    const int incx = 1;

    std::vector<float> x(num*incx);
    for (int i = 0; i < num; i++) {
        float this_a = (float)i;
        if (i % 3 == 0) this_a = -this_a;
        x[i*incx] = this_a;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto a : x) {
        expected += std::abs(a);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(result, expected);
}

TEST(Sasum, 1025_inc2) {
    const int num = 1025;
    const int incx = 2;

    std::vector<float> x(num*incx);
    for (int i = 0; i < num; i++) {
        float this_a = (float)i;
        if (i % 3 == 0) this_a = -this_a;
        x[i*incx] = this_a;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto a : x) {
        expected += std::abs(a);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(result, expected);
}

TEST(Sasum, 1025_noinc) {
    const int num = 1025;
    const int incx = 1;

    std::vector<float> x(num*incx);
    for (int i = 0; i < num; i++) {
        float this_a = (float)i;
        if (i % 3 == 0) this_a = -this_a;
        x[i*incx] = this_a;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto a : x) {
        expected += std::abs(a);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(result, expected);
}

TEST(Sasum, 65537_inc2) {
    const int num = 256 * 256 + 1;
    const int incx = 2;

    std::vector<float> x(num*incx);
    for (int i = 0; i < num; i++) {
        float this_a = (float)i / num / num;
        if (i % 3 == 0) this_a = -this_a;
        x[i*incx] = this_a;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto a : x) {
        expected += std::abs(a);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    // Because of large size of input vector and possibly different algorithm
    // used to calculate expected value, result is scaled to be in range [0, 1]
    // and appropriate absolute error is used to check correctness
    EXPECT_NEAR(result, expected, 1.e-5f);
}

TEST(Sasum, 65537_inc1) {
    const int num = 256*256 + 1;
    const int incx = 1;

    std::vector<float> x(num*incx);
    for (int i = 0; i < num; i++) {
        float this_a = ((float)i) / num / num;
        if (i % 3 == 0) this_a = -this_a;
        x[i*incx] = this_a;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto a : x) {
        expected += std::abs(a);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSasum(handle, num, x.data(), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    // Because of large size of input vector and possibly different algorithm
    // used to calculate expected value, result is scaled to be in range [0, 1]
    // and appropriate absolute error is used to check correctness
    EXPECT_NEAR(result, expected, 1.e-5f);
}
