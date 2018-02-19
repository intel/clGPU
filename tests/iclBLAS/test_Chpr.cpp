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

#define EXPECT_COMPLEX_EQ(expected, result) \
    EXPECT_FLOAT_EQ(expected.real(), result.real()); \
    EXPECT_FLOAT_EQ(expected.imag(), result.imag())

TEST(Chpr, naive_up_5x5) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 5;
    const int incx = 1;

    float alpha = 1.1f;

    std::complex<float> x[n*incx] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ 9.f, 10.f } };
    std::complex<float> a[n*(n + 1) / 2] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f },
                                             { 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },{ 19.f, 20.f }, 
                                             { 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };

    std::complex<float> expected[n*(n + 1) / 2];
    for (int i = 0, k=0; i < n; i++) {
        for (int j = 0; j < i; j++, k++) {
            expected[k] = a[k] + alpha*(x[j*incx] * std::conj(x[i*incx]));
        }
        expected[k] = a[k] + alpha * (x[i*incx] * std::conj(x[i*incx]));
        expected[k].imag(0.f);
        k++;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChpr(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*(n + 1) / 2; i++) {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Chpr, naive_up_3x3_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 3;
    const int incx = 2;

    float alpha = 1.1f;

    std::complex<float> x[n*incx] = { { -1.f, 2.f },{ -3.f, 4.f },{ -5.f, 6.f },{ -7.f, 8.f },{ -9.f, 10.f },{ -11.f, 12.f } };
    std::complex<float> a[n*(n + 1) / 2] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ 9.f, 10.f },{ 11.f, 12.f } };

    std::complex<float> expected[n*(n + 1) / 2];
    for (int i = 0, k = 0; i < n; i++) {
        for (int j = 0; j < i; j++, k++) {
            expected[k] = a[k] + alpha*(x[j*incx] * std::conj(x[i*incx]));
        }
        expected[k] = a[k] + alpha * (x[i*incx] * std::conj(x[i*incx]));
        expected[k].imag(0.f);
        k++;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChpr(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*(n + 1) / 2; i++) {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Chpr, naive_low_5x5) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 5;
    const int incx = 1;

    float alpha = -1.1f;

    std::complex<float> x[n*incx] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ 9.f, 10.f } };
    std::complex<float> a[n*(n + 1) / 2] = { { 1.f, 2.f },{ 3.f, 4.f },{ 5.f, 6.f },{ 7.f, 8.f },{ 9.f, 10.f },
                                             { 11.f, 12.f },{ 13.f, 14.f },{ 15.f, 16.f },{ 17.f, 18.f },
                                             { 19.f, 20.f },{ 21.f, 22.f },{ 23.f, 24.f },{ 25.f, 26.f },{ 27.f, 28.f },{ 29.f, 30.f } };

    std::complex<float> expected[n*(n + 1) / 2];
    for (int i = 0, k = 0; i < n; i++) {
        expected[k] = a[k] + alpha * (x[i*incx] * std::conj(x[i*incx]));
        expected[k].imag(0.f);
        k++;
        for (int j = i+1; j < n; j++, k++) {
            expected[k] = a[k] + alpha*(x[j*incx] * std::conj(x[i*incx]));
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChpr(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*(n + 1) / 2; i++) {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Chpr, naive_low_3x3_inc2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 3;
    const int incx = 2;

    float alpha = -1.1f;

    std::complex<float> x[n*incx] = { { 1.f, -2.f },{ 3.f, -4.f },{ 5.f, -6.f },{ 7.f, -8.f },{ 9.f, -10.f },{ 11.f, -12.f } };
    std::complex<float> a[n*(n + 1) / 2] = { { -1.f, 2.f },{ -3.f, 4.f },{ -5.f, 6.f },{ -7.f, 8.f },{ -9.f, 10.f },{ -11.f, 12.f } };

    std::complex<float> expected[n*(n + 1) / 2];
    for (int i = 0, k = 0; i < n; i++) {
        expected[k] = a[k] + alpha * (x[i*incx] * std::conj(x[i*incx]));
        expected[k].imag(0.f);
        k++;
        for (int j = i + 1; j < n; j++, k++) {
            expected[k] = a[k] + alpha*(x[j*incx] * std::conj(x[i*incx]));
        }
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasChpr(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*(n + 1) / 2; i++) {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}
