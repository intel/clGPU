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
#include <complex>
#include "blas_reference.hpp"
#include "iclblas_test_base.hpp"
#include "gtest_utils.hpp"

using Caxpy = iclblas_test_base;

TEST_F(Caxpy, 5x3_plus_5x2) {
    const int num = 5;
    const int incy = 3;
    const int incx = 2;
    oclComplex_t alpha = { 1.3f, 1.1f };
    oclComplex_t y[num*incy + 2] = { { 1.f, 0.f }, { 1.3f, 0.3f }, { 1.6f, 0.6f },
                                     { 2.f, 1.f}, { 2.3f, 1.3f }, { 2.6f, 1.6f},
                                     { 3.f, 2.f },{ 3.3f, 2.3f },{ 3.6f, 2.6f },
                                     { 4.f, 3.f },{ 4.3f, 3.3f },{ 4.6f, 3.6f },
                                     { 5.f, 4.f },{ 5.3f, 4.3f },{ 5.6f, 4.6f },
                                     { 6.f, 5.f}, { 7.f, 6.f } };
    oclComplex_t x[num*incx + 1] = { { 1.f, 11.f }, { 2.f, 10.f },
                                     { 3.f, 9.f },{ 4.f, 8.f },
                                     { 5.f, 7.f },{ 6.f, 6.f },
                                     { 7.f, 5.f },{ 8.f, 4.f },
                                     { 9.f, 3.f },{ 10.f, 2.f }, {11.f, 1.f} };

    oclComplex_t expected_result[num*incy + 2];
    for(size_t i = 0; i < num*incy + 2; i++)
    {
        expected_result[i] = y[i];
    }


    Caxpy_reference(num, alpha, x, incx, expected_result, incy);

    auto status = iclblasCaxpy(_handle, num, &alpha, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<oclComplex_t>, expected_result, y);
}

TEST_F(Caxpy, alpha0) {
    const int num = 3;
    const int incy = 1;
    const int incx = 1;
    oclComplex_t alpha = { .0f, .0f };
    oclComplex_t y[num*incy] = { { 1.f, 1.f }, { 2.f, 2.f }, { 3.f, 3.f } };
    oclComplex_t x[num*incx + 1] = { { 4.f, 4.f }, { 5.f, 5.f }, { 6.f, 6.f}, { 7.f, 7.f } };
    oclComplex_t expected_result[num*incy];
    for(size_t i = 0; i < num*incy; i++)
    {
        expected_result[i] = y[i];
    }

    Caxpy_reference(num, alpha, x, incx, expected_result, incy);

    auto status = iclblasCaxpy(_handle, num, &alpha, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<oclComplex_t>, expected_result, y);
}

TEST_F(Caxpy, num0) {
    const int num = 0;
    const int incy = 1;
    const int incx = 1;
    const int x_size = 2;
    const int y_size = 3;
    oclComplex_t alpha = { 12.8f, 13.8f };
    oclComplex_t y[y_size] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };
    oclComplex_t x[x_size] = { { 4.f, 4.f },{ 5.f, 5.f } };
    oclComplex_t expected_result[y_size];

    for(size_t i = 0; i < y_size; i++)
    {
        expected_result[i] = y[i];
    }

    Caxpy_reference(num, alpha, x, incx, expected_result, incy);

    auto status = iclblasCaxpy(_handle, num, &alpha, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<oclComplex_t>, expected_result, y);
}

TEST_F(Caxpy, 11_incs1) {
    using complex = std::complex<float>;
    const int num = 11;
    const int incx = 1;
    const int incy = 1;

    complex alpha = { 1.1f, 0.8f };
    std::vector<complex> x(num*incx);
    for (int i = 0; i < num; i++) {
        x[i*incx] = { 1.f * i,1.f * i + 1 };
        x[i*incx] /= 2.f * num;
    }
    std::vector<complex> y(num*incy);
    for (int i = 0; i < num; i++) {
        y[i*incy] = {1.f * num - i - 1, 1.f * num - i};
        y[i*incy] /= 2.f * num;
    }

    auto expected = y;

    Caxpy_reference(num, alpha, x, incx, expected, incy);

    auto status = iclblasCaxpy(_handle, num, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x.data()), incx,
        reinterpret_cast<oclComplex_t*>(y.data()), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<std::complex<float>>, expected, y);
}

TEST_F(Caxpy, 11_incx1) {
    using complex = std::complex<float>;
    const int num = 11;
    const int incx = 1;
    const int incy = 2;

    complex alpha = { 1.1f, 0.8f };
    std::vector<complex> x(num*incx);
    for (int i = 0; i < num; i++) {
        x[i*incx] = { 1.f * i,1.f * i + 1 };
        x[i*incx] /= 2.f * num;
    }
    std::vector<complex> y(num*incy);
    for (int i = 0; i < num; i++) {
        y[i*incy] = { 1.f * num - i - 1, 1.f * num - i };
        y[i*incy] /= 2.f * num;
    }

    auto expected = y;

    Caxpy_reference(num, alpha, x, incx, expected, incy);

    auto status = iclblasCaxpy(_handle, num, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x.data()), incx,
        reinterpret_cast<oclComplex_t*>(y.data()), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<std::complex<float>>, expected, y);
}

TEST_F(Caxpy, 11_incy1) {
    using complex = std::complex<float>;
    const int num = 11;
    const int incx = 2;
    const int incy = 1;

    complex alpha = { 1.1f, 0.8f };
    std::vector<complex> x(num*incx);
    for (int i = 0; i < num; i++) {
        x[i*incx] = { 1.f * i,1.f * i + 1 };
        x[i*incx] /= 2.f * num;
    }
    std::vector<complex> y(num*incy);
    for (int i = 0; i < num; i++) {
        y[i*incy] = { 1.f * num - i - 1, 1.f * num - i };
        y[i*incy] /= 2.f * num;
    }

    auto expected = y;

    Caxpy_reference(num, alpha, x, incx, expected, incy);

    auto status = iclblasCaxpy(_handle, num, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x.data()), incx,
        reinterpret_cast<oclComplex_t*>(y.data()), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<std::complex<float>>, expected, y);
}
