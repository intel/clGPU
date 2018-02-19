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
#include <complex>

TEST(Scnrm2, oneComplex)
{
    const int n = 1;
    const int incx = 1;

    oclComplex_t x[n * incx] = { { 5.f, -2.f } };

    float res = 0.f;
    float ref = 5.3851648f;

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, x, incx, &res);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(ref, res);
}

TEST(Scnrm2, Complex_4)
{
    const int n = 4;
    const int incx = 2;

    oclComplex_t x[n * incx] = { { -1.f, 7.f },{ 23.f, 2.f },{ 3.f, -4.f },{ 14.f, 4.f },{ 4.f, 12.f },{ 8.f, 1.f },{ 7.f, -1.f },{ 2.f, -1.f } };

    float res = 0.f;
    float ref = 0.f;

    for (int i = 0; i < n; i++)
    {
        oclComplex_t complex = x[i * incx];

        float real = complex.val[0];
        float imag = complex.val[1];

        ref += real * real + imag * imag;
    }

    ref = std::sqrt(ref);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, x, incx, &res);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(ref, res);
}

TEST(Scnrm2, 16_inc2) {
    const int n = 16;
    const int incx = 2;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 2.f*i, 1.f*i + 1.f };
        x[i*incx] /= 2.f * n * n;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto this_x : x) {
        expected += std::norm(this_x);
    }
    expected = std::sqrt(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Scnrm2, 16_inc1) {
    const int n = 16;
    const int incx = 1;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 2.f*i, 1.f*i + 1.f };
        x[i*incx] /= 2.f * n * n;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto this_x : x) {
        expected += std::norm(this_x);
    }
    expected = std::sqrt(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Scnrm2, 1025_inc2) {
    const int n = 1025;
    const int incx = 2;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 2.f*i, 1.f*i + 1.f };
        x[i*incx] /= 2.f * n * n;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto this_x : x) {
        expected += std::norm(this_x);
    }
    expected = std::sqrt(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Scnrm2, 1025_inc1)
{
    const int n = 1025;
    const int incx = 1;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 2.f*i, 1.f*i + 1.f };
        x[i*incx] /= 2.f * n * n;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto this_x : x) {
        expected += std::norm(this_x);
    }
    expected = std::sqrt(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(expected, result);
}

TEST(Scnrm2, 65537_inc2) {
    const int n = 65537;
    const int incx = 2;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 2.f*i, 1.f*i + 1.f };
        x[i*incx] /= 2.f * n * n;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto this_x : x) {
        expected += std::norm(this_x);
    }
    expected = std::sqrt(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasScnrm2(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(expected, result, 1.e-5f);
}

TEST(Scnrm2, 65537_inc1) {
    const int n = 65537;
    const int incx = 1;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 2.f*i, 1.f*i + 1.f };
        x[i*incx] /= 2.f * n * n;
    }
    float result = -1.f;
    float expected = 0.f;
    for (auto this_x : x) {
        expected += std::norm(this_x);
    }
    expected = std::sqrt(expected);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasScnrm2(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, &result);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(expected, result, 1.e-5f);
}
