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

#define IDX(m, n, ld) (n)*(ld) + (m)

#define EXPECT_COMPLEX_EQ(expected, result) \
    EXPECT_FLOAT_EQ(expected.real(), result.real()); \
    EXPECT_FLOAT_EQ(expected.imag(), result.imag())

std::vector<std::complex<float>> cpuCher_upper(const int n, const float alpha, const std::vector<std::complex<float>> x, const int incx, const std::vector<std::complex<float>> a, const int lda) {
    auto result = a;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            result[IDX(j, i, lda)] += alpha * x[j*incx] * std::conj(x[i*incx]);
        }
        result[IDX(i, i, lda)].imag(0.f);
    }
    return result;
}

std::vector<std::complex<float>> cpuCher_lower(const int n, const float alpha, const std::vector<std::complex<float>> x, const int incx, const std::vector<std::complex<float>> a, const int lda) {
    auto result = a;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            result[IDX(j, i, lda)] += alpha * x[j*incx] * std::conj(x[i*incx]);
        }
        result[IDX(i, i, lda)].imag(0.f);
    }
    return result;
}

TEST(Cher, 5x5_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 5;
    const int incx = 1;
    const int lda = n;

    float alpha = 1.1f;
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f }, { 9.f, 10.f } };
    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { -7.f, -8.f },
                                     { 3.f, 4.f }, { 5.f, 6.f }, { -9.f, -10.f }, { -11.f, -12.f }, { -13.f, -14.f },
                                     { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { -15.f, -16.f }, { -17.f, -18.f },
                                     { 13.f, 14.f }, { 15.f, 16.f }, { 17.f, 18.f }, { 19.f, 20.f }, { -19.f, -20.f },
                                     { 21.f, 22.f }, { 23.f, 24.f }, { 25.f, 26.f }, { 27.f, 28.f }, { 29.f, 30.f } };

    std::complex<float> expected[n*lda];
    for (int i = 0; i < n*lda; i++) expected[i] = a[i];
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            expected[IDX(i, j, lda)] += alpha * x[i*incx] * std::conj(x[j*incx]);
        }
        expected[IDX(i, i, lda)].imag(0.f);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 5x4_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 4;
    const int incx = 1;
    const int lda = 5;

    float alpha = 1.1f;
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f } };
    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { -7.f, -8.f },
                                     { 3.f, 4.f }, { 5.f, 6.f }, { -9.f, -10.f }, { -11.f, -12.f }, { -13.f, -14.f },
                                     { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f }, { -15.f, -16.f }, { -17.f, -18.f },
                                     { 13.f, 14.f }, { 15.f, 16.f }, { 17.f, 18.f }, { 19.f, 20.f }, { -19.f, -20.f } };

    std::complex<float> expected[n*lda];
    for (int i = 0; i < n*lda; i++) expected[i] = a[i];
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            expected[IDX(i, j, lda)] += alpha * x[i*incx] * std::conj(x[j*incx]);
        }
        expected[IDX(i, i, lda)].imag(0.f);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 5x5_low_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 5;
    const int incx = 1;
    const int lda = n;

    float alpha = -1.3f;
    std::complex<float> x[n*incx] = { { -1.f, -2.f }, { -3.f, -4.f }, { -5.f, -6.f }, { -7.f, -8.f }, { -9.f, -10.f } };
    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f }, { 7.f, 8.f }, { 9.f, 10.f },
                                     { -1.f, -2.f }, { 11.f, 12.f }, { 13.f, 14.f }, { 15.f, 16.f }, { 17.f, 18.f },
                                     { -3.f, -4.f }, { -5.f, -6.f }, { 19.f, 20.f }, { 21.f, 22.f }, { 23.f, 24.f },
                                     { -7.f, -8.f }, { -9.f, -10.f }, { -11.f, -12.f }, { 25.f, 26.f }, { 27.f, 28.f },
                                     { -13.f, -14.f }, { -15.f, -16.f }, { -17.f, -18.f }, { -19.f, -20.f }, { 29.f, 30.f } };

    std::complex<float> expected[n*lda];
    for (int i = 0; i < n*lda; i++) expected[i] = a[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i+1; j++) {
            expected[IDX(i, j, lda)] += alpha * x[i*incx] * std::conj(x[j*incx]);
        }
        expected[IDX(i, i, lda)].imag(0.f);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 3x3_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 3;
    const int incx = 2;
    const int lda = n;

    float alpha = -0.1f;
    std::complex<float> x[n*incx] = { { 1.f, 2.f }, { -1.f, -2.f }, { 3.f, 4.f }, { -3.f, -4.f }, { 5.f, 6.f }, { -5.f, -6.f } };
    std::complex<float> a[n*lda] = { { 1.f, 2.f }, { 3.f, 4.f }, { 5.f, 6.f },
                                     { -1.f, -2.f }, { 7.f, 8.f }, { 9.f, 10.f },
                                     { -3.f, -4.f }, { -5.f, -6.f }, { 11.f, 12.f } };

    std::complex<float> expected[n*lda];
    for (int i = 0; i < n*lda; i++) expected[i] = a[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i + 1; j++) {
            expected[IDX(i, j, lda)] += alpha * x[i*incx] * std::conj(x[j*incx]);
        }
        expected[IDX(i, i, lda)].imag(0.f);
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(a), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 16x16_up_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 16;
    const int incx = 2;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 16x16_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 16;
    const int incx = 1;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 16x16_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 7;
    const int incx = 2;
    const int lda = 8;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 16x16_low_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 16;
    const int incx = 1;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 35x35_up_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 35;
    const int incx = 2;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 35x35_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 35;
    const int incx = 1;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 35x35_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 35;
    const int incx = 2;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}

TEST(Cher, 35x35_low_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 35;
    const int incx = 1;
    const int lda = n;

    float alpha = -0.1f;
    std::vector<std::complex<float>> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i*incx] = { 1.f * i, 1.f * i + 1.f };
    }
    std::vector<std::complex<float>> a(n*lda);
    for (int i = 0; i < n*lda; i++) {
        a[i] = { 1.f * i, 1.f * i + 1.f };
    }

    auto expected = cpuCher_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCher(handle, uplo, n, &alpha, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(a.data()), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++)
    {
        EXPECT_COMPLEX_EQ(expected[i], a[i]);
    }
}
