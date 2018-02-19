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

TEST(Saxpy, 5x3_plus_5x2) {
    const int num = 5;
    const int incy = 3;
    const int incx = 2;
    float alpha = 1.3f;
    float y[num*incy + 2] = { 1.f, 1.3f, 1.6f,
                          2.f, 2.3f, 2.6f,
                          3.f, 3.3f, 3.6f,
                          4.f, 4.3f, 4.6f,
                          5.f, 5.3f, 5.6f, 6.f, 7.f };
    float x[num*incx + 1] = { 1.f, 2.f,
                          3.f, 4.f,
                          5.f, 6.f,
                          7.f, 8.f,
                          9.f, 10.f, 11.f };
    float expected_result[num*incy + 2] = { 1.f + 1.3f * 1.f, 1.3f, 1.6f,
                                        2.f + 1.3f * 3.f, 2.3f, 2.6f,
                                        3.f + 1.3f * 5.f, 3.3f, 3.6f,
                                        4.f + 1.3f * 7.f, 4.3f, 4.6f,
                                        5.f + 1.3f * 9.f, 5.3f, 5.6f, 6.f, 7.f };
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSaxpy(handle, num, &alpha, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected_result[i], y[i]);
    }
}

TEST(Saxpy, alpha0) {
    const int num = 7;
    const int incy = 1;
    const int incx = 1;
    float alpha = .0f;
    float y[num*incy] = { 1.f, 2.f, 3.f,
                          4.f, 5.f, 6.f,
                          7.f };
    float x[num*incx + 1] = { 8.f, 9.f, 10.f,
                              11.f, 12.f, 13.f,
                              14.f, 15.f };
    float expected_result[num*incy] = { 1.f, 2.f, 3.f,
                                        4.f, 5.f, 6.f,
                                        7.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSaxpy(handle, num, &alpha, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected_result[i], y[i]);
    }
}

TEST(Saxpy, num0) {
    const int num = 0;
    const int incy = 1;
    const int incx = 1;
    const int x_size = 2;
    const int y_size = 3;
    float alpha = 12.8f;
    float y[y_size] = { 1.f, 2.f, 3.f };
    float x[x_size] = { 4.f, 5.f };
    float expected_result[y_size] = { 1.f, 2.f, 3.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSaxpy(handle, num, &alpha, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < y_size; i++)
    {
        EXPECT_FLOAT_EQ(expected_result[i], y[i]);
    }
}

TEST(Saxpy, 11_incx1) {
    const int num = 11;
    const int incy = 2;
    const int incx = 1;
    float alpha = 1.1f;
    std::vector<float> x(num * incx);
    for (int i = 0; i < num; i++) {
        x[i*incx] = 1.f * i / num;
    }
    std::vector<float> y(num * incy);
    for (int i = 0; i < num; i++) {
        y[i*incy] = 1.f * (num - i - 1) / num;
    }
    auto expected = y;
    for (int i = 0; i < num; i++) {
        expected[i*incy] += alpha * x[i*incx];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSaxpy(handle, num, &alpha, x.data(), incx, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}

TEST(Saxpy, 11_incy1) {
    const int num = 11;
    const int incy = 1;
    const int incx = 2;
    float alpha = 1.1f;
    std::vector<float> x(num * incx);
    for (int i = 0; i < num; i++) {
        x[i*incx] = 1.f * i / num;
    }
    std::vector<float> y(num * incy);
    for (int i = 0; i < num; i++) {
        y[i*incy] = 1.f * (num - i - 1) / num;
    }
    auto expected = y;
    for (int i = 0; i < num; i++) {
        expected[i*incy] += alpha * x[i*incx];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSaxpy(handle, num, &alpha, x.data(), incx, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}

TEST(Saxpy, 11_incs1) {
    const int num = 1;
    const int incy = 1;
    const int incx = 1;
    float alpha = 1.1f;
    std::vector<float> x(num * incx);
    for (int i = 0; i < num; i++) {
        x[i*incx] = 1.f * i / num;
    }
    std::vector<float> y(num * incy);
    for (int i = 0; i < num; i++) {
        y[i*incy] = 1.f * (num - i - 1) / num;
    }
    auto expected = y;
    for (int i = 0; i < num; i++) {
        expected[i*incy] += alpha * x[i*incx];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSaxpy(handle, num, &alpha, x.data(), incx, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incy; i++)
    {
        EXPECT_FLOAT_EQ(expected[i], y[i]);
    }
}
