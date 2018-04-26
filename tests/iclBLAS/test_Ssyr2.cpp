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

#include <stdexcept>

// TODO Move to more suitable place
inline char iclblasFillModeCast(iclblasFillMode_t uplo)
{
    switch (uplo)
    {
    case ICLBLAS_FILL_MODE_UPPER:
        return 'U';
    case ICLBLAS_FILL_MODE_LOWER:
        return 'L';
    default:
        throw std::invalid_argument("uplo");
    }
}

int Ssyr2_reference(iclblasFillMode_t uplo, int n, float alpha, array_ref<float> x, int incx, array_ref<float> y, int incy, array_ref<float> A, int lda)
{
    return Ssyr2_reference(iclblasFillModeCast(uplo), n, alpha, x, incx, y, incy, A, lda);
}

using Ssyr2 = iclblas_test_base;

TEST_F(Ssyr2, upper)
{
    const iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 64;
    const int lda = n;
    const int incx = 1;
    const int incy = 1;
    const float alpha = 1.f;

    std::vector<float> A(n * lda);
    for (int i = 0; i < n*lda; i++)
    {
        A[i] = 0.75f * i;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 0.25f * i;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i;
    }

    auto expected_result = A;

    Ssyr2_reference(uplo, n, alpha, x, incx, y, incy, expected_result, lda);

    auto status = iclblasSsyr2(_handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<float>, expected_result, A);
}

TEST_F(Ssyr2, upper_inc)
{
    const iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 64;
    const int lda = n;
    const int incx = 2;
    const int incy = 4;
    const float alpha = 1.f;

    std::vector<float> A(n * lda);
    for (int i = 0; i < n*lda; i++)
    {
        A[i] = 0.75f * i;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 0.25f * i;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i;
    }

    auto expected_result = A;

    Ssyr2_reference(uplo, n, alpha, x, incx, y, incy, expected_result, lda);

    auto status = iclblasSsyr2(_handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<float>, expected_result, A);
}

TEST_F(Ssyr2, upper_lda)
{
    const iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 512;
    const int lda = 542;
    const int incx = 1;
    const int incy = 1;
    const float alpha = 1.f;

    std::vector<float> A(n * lda);
    for (int i = 0; i < n*lda; i++)
    {
        A[i] = 0.75f * i;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 0.25f * i;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i;
    }

    auto expected_result = A;

    Ssyr2_reference(uplo, n, alpha, x, incx, y, incy, expected_result, lda);

    auto status = iclblasSsyr2(_handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<float>, expected_result, A);
}

TEST_F(Ssyr2, lower)
{
    const iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 64;
    const int lda = n;
    const int incx = 1;
    const int incy = 1;
    const float alpha = 1.f;

    std::vector<float> A(n * lda);
    for (int i = 0; i < n*lda; i++)
    {
        A[i] = 0.75f * i;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 0.25f * i;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i;
    }

    auto expected_result = A;

    Ssyr2_reference(uplo, n, alpha, x, incx, y, incy, expected_result, lda);

    auto status = iclblasSsyr2(_handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<float>, expected_result, A);
}

TEST_F(Ssyr2, lower_inc)
{
    const iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 64;
    const int lda = n;
    const int incx = 2;
    const int incy = 4;
    const float alpha = 1.f;

    std::vector<float> A(n * lda);
    for (int i = 0; i < n*lda; i++)
    {
        A[i] = 0.75f * i;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 0.25f * i;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i;
    }

    auto expected_result = A;

    Ssyr2_reference(uplo, n, alpha, x, incx, y, incy, expected_result, lda);

    auto status = iclblasSsyr2(_handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<float>, expected_result, A);
}

TEST_F(Ssyr2, lower_lda)
{
    const iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 512;
    const int lda = 542;
    const int incx = 1;
    const int incy = 1;
    const float alpha = 1.f;

    std::vector<float> A(n * lda);
    for (int i = 0; i < n*lda; i++)
    {
        A[i] = 0.75f * i;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 0.25f * i;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i;
    }

    auto expected_result = A;

    Ssyr2_reference(uplo, n, alpha, x, incx, y, incy, expected_result, lda);

    auto status = iclblasSsyr2(_handle, uplo, n, &alpha, x.data(), incx, y.data(), incy, A.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_PRED_FORMAT2(AssertArraysEqual<float>, expected_result, A);
}
