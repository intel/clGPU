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

TEST(Srotmg, q1_bigger) {
    float d1 = 1.2f;
    float d2 = .5f;
    float b1 = 3.2f;
    float b2 = 3.7f;
    float result[5] = { 3.f, 5.f, 6.f, 7.f, 8.f };

    const float expected_result[5] = {0.f, 1.f, -b2/b1, d2*b2/d1/b1, 1.f};
    const float u = 1 + d2*b2/(d1*b1)*b2/b1;
    const float expected_d1 = d1/u;
    const float expected_d2 = d2/u;
    const float expected_b1 = b1*u;


    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotmg(handle, &d1, &d2, &b1, &b2, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected_result[0], result[0]);
    EXPECT_FLOAT_EQ(expected_result[2], result[2]);
    EXPECT_FLOAT_EQ(expected_result[3], result[3]);

    EXPECT_FLOAT_EQ(expected_d1, d1);
    EXPECT_FLOAT_EQ(expected_d2, d2);
    EXPECT_FLOAT_EQ(expected_b1, b1);
}

TEST(Srotmg, q2_bigger) {
    float d1 = .7f;
    float d2 = 1.1f;
    float b1 = 3.2f;
    float b2 = 2.7f;
    float result[5] = { 3.f, 5.f, 6.f, 7.f, 8.f };

    const float expected_result[5] = {1.f, b1*d1/b2/d2 , -1.f, 1.f, b1/b2};
    const float u = 1 + d1*b1*b1/d2/b2/b2;
    const float expected_d1 = d2/u;
    const float expected_d2 = d1/u;
    const float expected_b1 = b2*u;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotmg(handle, &d1, &d2, &b1, &b2, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected_result[0], result[0]);
    EXPECT_FLOAT_EQ(expected_result[1], result[1]);
    EXPECT_FLOAT_EQ(expected_result[4], result[4]);

    EXPECT_FLOAT_EQ(expected_d1, d1);
    EXPECT_FLOAT_EQ(expected_d2, d2);
    EXPECT_FLOAT_EQ(expected_b1, b1);
}

TEST(Srotmg, scaling) {
    float d1 = 1.f/4096.f/4096.f;
    float d2 = 1.f/4096.f/4096.f;
    float b1 = 3.3f;
    float b2 = 3.2f;
    float result[5] = { 3.f, 5.f, 6.f, 7.f, 8.f };
    
    const float gamma = 4096.f;

    const float expected_result[5] = {-1.f, 1.f/gamma, -b2/b1/gamma, d2*b2/d1/b1/gamma, 1.f/gamma};
    const float u = 1 + d2*b2*b2/d1/b1/b1;
    const float expected_d1 = d1/u*gamma*gamma;
    const float expected_d2 = d2/u*gamma*gamma;
    const float expected_b1 = b1*u/gamma;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotmg(handle, &d1, &d2, &b1, &b2, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected_result[0], result[0]);
    EXPECT_FLOAT_EQ(expected_result[1], result[1]);
    EXPECT_FLOAT_EQ(expected_result[2], result[2]);
    EXPECT_FLOAT_EQ(expected_result[3], result[3]);
    EXPECT_FLOAT_EQ(expected_result[4], result[4]);

    EXPECT_FLOAT_EQ(expected_d1, d1);
    EXPECT_FLOAT_EQ(expected_d2, d2);
    EXPECT_FLOAT_EQ(expected_b1, b1);
}

TEST(Srotmg, negative_d2) {
    float d1 = 1.1f;
    float d2 = -1.f;
    float b1 = 3.3f;
    float b2 = 3.2f;
    float result[5] = { 3.f, 5.f, 6.f, 7.f, 8.f };
    
    const float expected_result[5] = {0.f, 1.f, -b2/b1, d2*b2/d1/b1, 1.f};
    const float u = 1 + d2*b2/(d1*b1)*b2/b1;
    const float expected_d1 = d1/u;
    const float expected_d2 = d2/u;
    const float expected_b1 = b1*u;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotmg(handle, &d1, &d2, &b1, &b2, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected_result[0], result[0]);
    EXPECT_FLOAT_EQ(expected_result[2], result[2]);
    EXPECT_FLOAT_EQ(expected_result[3], result[3]);

    EXPECT_FLOAT_EQ(expected_d1, d1);
    EXPECT_FLOAT_EQ(expected_d2, d2);
    EXPECT_FLOAT_EQ(expected_b1, b1);
}

TEST(Srotmg, zero_d1) {
    float d1 = 0.f;
    float d2 = 1.2f;
    float b1 = 3.3f;
    float b2 = 3.2f;
    float result[5] = { 3.f, 5.f, 6.f, 7.f, 8.f };

    const float expected_result[5] = { 1.f, 0.f, -1.f, 1.f, b1/b2 };
    const float u = 1 + d1*b1*b1 / d2 / b2 / b2;
    const float expected_d1 = d2 / u;
    const float expected_d2 = d1 / u;
    const float expected_b1 = b2*u;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotmg(handle, &d1, &d2, &b1, &b2, result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected_result[0], result[0]);
    EXPECT_FLOAT_EQ(expected_result[1], result[1]);
    EXPECT_FLOAT_EQ(expected_result[4], result[4]);

    EXPECT_FLOAT_EQ(expected_d1, d1);
    EXPECT_FLOAT_EQ(expected_d2, d2);
    EXPECT_FLOAT_EQ(expected_b1, b1);
}
