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

#define EXPECT_OCLCOMPLEX_EQ(expected, result) \
    EXPECT_FLOAT_EQ(expected.val[0], result.val[0]); \
    EXPECT_FLOAT_EQ(expected.val[1], result.val[1])

TEST(Cswap, n0) {
    const int num = 0;
    const int incx = 1;
    const int incy = 1;

    oclComplex_t x[] = { { 1.f, 1.f }, { 2.f, 2.f }, { 3.f, 3.f } };

    const oclComplex_t expected_x[] = { { 1.f, 1.f },{ 2.f, 2.f },{ 3.f, 3.f } };

    oclComplex_t y[] = { { 4.f, 4.f }, { 5.f, 5.f } };

    const oclComplex_t expected_y[] = { { 4.f, 4.f },{ 5.f, 5.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCswap(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_OCLCOMPLEX_EQ(expected_x[i], x[i]);
    }
    for (int i = 0; i < num*incy; ++i)
    {
        EXPECT_OCLCOMPLEX_EQ(expected_y[i], y[i]);
    }
}

TEST(Cswap, n5_incx1) {
    const int num = 5;
    const int incx = 1;
    const int incy = 2;

    oclComplex_t x[num*incx] = { { 1.f, 1.f }, { 2.f, 2.f }, { 3.f, 3.f }, { 4.f, 4.f }, { 5.f, 5.f } };
    
    oclComplex_t y[num*incy+1] = { { 6.f, 1.f }, { 7.f, 2.f }, { 8.f, 3.f }, { 9.f, 4.f }, { 10.f, 5.f },
                                   { 11.f, 6.f }, { 12.f, 7.f }, { 13.f, 8.f }, { 14.f, 9.f }, { 15.f, 10.f }, { 16.f, 11.f } };

    const oclComplex_t expected_x[num*incx] = { { 6.f, 1.f }, { 8.f, 3.f }, { 10.f, 5.f }, { 12.f, 7.f }, { 14.f, 9.f } };

    const oclComplex_t expected_y[num*incy + 1] = { { 1.f, 1.f }, { 7.f, 2.f }, { 2.f, 2.f }, { 9.f, 4.f }, { 3.f, 3.f },
                                              { 11.f, 6.f }, { 4.f, 4.f }, { 13.f, 8.f }, { 5.f, 5.f }, { 15.f, 10.f }, { 16.f, 11.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCswap(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_OCLCOMPLEX_EQ(expected_x[i], x[i]);
    }
    for (int i = 0; i < num*incy; ++i)
    {
        EXPECT_OCLCOMPLEX_EQ(expected_y[i], y[i]);
    }
}

TEST(Cswap, n11_incy1) {
    const int n = 11;
    const int incx = 2;
    const int incy = 1;

    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = {1.f * i, 1.f * i + 1.f};
    }
    std::vector<std::complex<float>> y(n*incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = { 1.f * n - i - 1.f, 1.f * n - i };
    }

    auto expected_x = x;
    auto expected_y = y;
    for (int i = 0; i < n; i++) {
        expected_x[i*incx] = y[i*incy];
        expected_y[i*incy] = x[i*incx];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCswap(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(y.data()), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*incx; i++) {
        EXPECT_FLOAT_EQ(expected_x[i].real(), x[i].real());
        EXPECT_FLOAT_EQ(expected_x[i].imag(), x[i].imag());
    }
    for (int i = 0; i < n*incy; i++) {
        EXPECT_FLOAT_EQ(expected_y[i].real(), y[i].real());
        EXPECT_FLOAT_EQ(expected_y[i].imag(), y[i].imag());
    }
}

TEST(Cswap, n11_incs1) {
    const int num = 11;
    const int incx = 1;
    const int incy = 1;

    oclComplex_t x[num*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f },
                                 { 13.f, 14.f }, { 15.f, 16.f }, { 17.f, 18.f }, { 19.f, 20.f }, { 21.f, 22.f} };

    oclComplex_t y[num*incy] = { { 1.5f, 2.5f },{ 3.5f, 4.5f },{ 5.5f, 6.5f },{ 7.5f, 8.5f },{ 9.5f, 10.5f },{ 11.5f, 12.5f },
                                 { 13.5f, 14.5f },{ 15.5f, 16.5f },{ 17.5f, 18.5f },{ 19.5f, 20.5f },{ 21.5f, 22.5f } };

    const oclComplex_t expected_x[num*incx] = { { 1.5f, 2.5f },{ 3.5f, 4.5f },{ 5.5f, 6.5f },{ 7.5f, 8.5f },{ 9.5f, 10.5f },{ 11.5f, 12.5f },
                                                { 13.5f, 14.5f },{ 15.5f, 16.5f },{ 17.5f, 18.5f },{ 19.5f, 20.5f },{ 21.5f, 22.5f } };

    const oclComplex_t expected_y[num*incy] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },
                                                { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f },{ 21.f, 22.f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCswap(handle, num, x, incx, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < num*incx; ++i)
    {
        EXPECT_OCLCOMPLEX_EQ(expected_x[i], x[i]);
    }
    for (int i = 0; i < num*incy; ++i)
    {
        EXPECT_OCLCOMPLEX_EQ(expected_y[i], y[i]);
    }
}
