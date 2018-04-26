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

#define IDX(m, n, ld) (n)*(ld) + (m)

std::vector<float> cpuSsyr_upper(const int n, const float alpha, const std::vector<float> x, const int incx,const std::vector<float> a, const int lda) {
    std::vector<float> result = a;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            result[IDX(j, i, lda)] += alpha * x[j * incx] * x[i * incx];
        }
    }
    return result;
}

std::vector<float> cpuSsyr_lower(const int n, const float alpha, const std::vector<float> x, const int incx, const std::vector<float> a, const int lda) {
    std::vector<float> result = a;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            result[IDX(j, i, lda)] += alpha * x[j * incx] * x[i * incx];
        }
    }
    return result;
}

TEST(Ssyr, 6x5_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 5;
    const int lda = 6;
    const int incx = 1;
    float alpha = 1.1f;
    float a[lda*n] = { 1.f, -1.f, -2.f, -3.f, -4.f, -5.f,
                       2.f, 3.f, -6.f, -7.f, -8.f, -9.f,
                       4.f, 5.f, 6.f, -10.f, -11.f, -12.f,
                       7.f, 8.f, 9.f, 10.f, -13.f, -14.f,
                       11.f, 12.f, 13.f, 14.f, 15.f, -15.f };
    float x[n*incx] = {1.f, 2.f, 3.f, 4.f, 5.f};
    const float expected[lda*n] = { 1.f + 1.1f, -1.f, -2.f, -3.f, -4.f, -5.f,
                                    2.f+1.1f*2.f, 3.f+1.1f*4.f, -6.f, -7.f, -8.f, -9.f,
                                    4.f+1.1f*3.f, 5.f+1.1f*6.f, 6.f+1.1f*9.f, -10.f, -11.f, -12.f,
                                    7.f+1.1f*4.f, 8.f+1.1f*8.f, 9.f+1.1f*12.f, 10.f+1.1f*16.f, -13.f, -14.f,
                                    11.f+1.1f*5.f, 12.f+1.1f*10.f, 13.f+1.1f*15.f, 14.f+1.1f*20.f, 15.f+1.1f*25.f, -15.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x, incx, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 5x5_low_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 5;
    const int lda = 5;
    const int incx = 1;
    float alpha = 1.1f;
    float a[lda*n] = { 1.f, 2.f, 3.f, 4.f, 5.f,
                      -1.f, 6.f, 7.f, 8.f, 9.f,
                      -2.f, -3.f, 10.f, 11.f, 12.f,
                      -4.f, -5.f, -6.f, 13.f, 14.f,
                      -7.f, -8.f, -9.f, -10.f, 15.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    const float expected[lda*n] = { 1.f+1.1f*1.f, 2.f+1.1f*2.f, 3.f+1.1f*3.f, 4.f+1.1f*4.f, 5.f+1.1f*5.f,
                                   -1.f, 6.f+1.1f*4.f, 7.f+1.1f*6.f, 8.f+1.1f*8.f, 9.f+1.1f*10.f,
                                   -2.f, -3.f, 10.f+1.1f*9.f, 11.f+1.1f*12.f, 12.f+1.1f*15.f,
                                   -4.f, -5.f, -6.f, 13.f+1.1f*16.f, 14.f+1.1f*20.f,
                                   -7.f, -8.f, -9.f, -10.f, 15.f+1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x, incx, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 3x3_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 3;
    const int lda = 3;
    const int incx = 2;
    float alpha = 1.1f;
    float a[lda*n] = { 1.f, 2.f, 3.f,
                      -1.f, 4.f, 5.f,
                      -2.f, -3.f, 6.f };
    float x[n*incx] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    const float expected[lda*n] = { 1.f+1.1f*1.f, 2.f+1.1f*3.f, 3.f+1.1f*5.f,
                                   -1.f, 4.f+1.1f*9.f, 5.f+1.1f*15.f,
                                   -2.f, -3.f, 6.f+1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x, incx, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 17x17_up_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 17;
    const int lda = n;
    const int incx = 2;
    float alpha = 1.1f;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = 1.f * i;
    }
    std::vector<float> a(n*lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            a[i*lda + j] = 1.f * i*lda + j;
        }
    }

    std::vector<float> expected = cpuSsyr_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x.data(), incx, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 17x17_up_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 17;
    const int lda = n;
    const int incx = 1;
    float alpha = 1.1f;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = 1.f * i;
    }
    std::vector<float> a(n*lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            a[i*lda + j] = 1.f * i*lda + j;
        }
    }

    std::vector<float> expected = cpuSsyr_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x.data(), incx, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda * n; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 17x17_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 17;
    const int lda = n;
    const int incx = 2;
    float alpha = 1.1f;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = 1.f * i;
    }
    std::vector<float> a(n*lda);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            a[i*lda + j] = 1.f * i*lda + j;
        }
    }

    std::vector<float> expected = cpuSsyr_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x.data(), incx, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 17x17_low_incx1) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 17;
    const int lda = n;
    const int incx = 1;
    float alpha = 1.1f;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = 1.f * i;
    }
    std::vector<float> a(n*lda);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            a[i*lda + j] = 1.f * i*lda + j;
        }
    }

    std::vector<float> expected = cpuSsyr_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x.data(), incx, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 72x71_up_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 71;
    const int lda = 72;
    const int incx = 2;
    float alpha = 1.1f;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = 1.f * i;
    }
    std::vector<float> a(n*lda);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            a[i*lda + j] = 1.f * i*lda + j;
        }
    }

    std::vector<float> expected = cpuSsyr_upper(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x.data(), incx, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda * n; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Ssyr, 72x71_low_incx2) {
    const auto uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 71;
    const int lda = 72;
    const int incx = 2;
    float alpha = 1.1f;

    std::vector<float> x(n*incx);
    for (int i = 0; i < n; i++) {
        x[i * incx] = 1.f * i;
    }
    std::vector<float> a(n*lda);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            a[i*lda + j] = 1.f * i*lda + j;
        }
    }

    std::vector<float> expected = cpuSsyr_lower(n, alpha, x, incx, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsyr(handle, uplo, n, &alpha, x.data(), incx, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda * n; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}
