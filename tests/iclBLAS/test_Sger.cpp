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

#define IDX(row, col, ld) ((col) * (ld) + (row))

std::vector<float> Sger_cpu(
    const int& m,
    const int& n,
    const float& alpha,
    const std::vector<float>& x,
    const int& incx,
    const std::vector<float>& y,
    const int& incy,
    const std::vector<float>& a,
    const int& lda)
{
    std::vector<float> result = a;
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            result[IDX(row, col,lda)] += alpha * (x[row * incx] * y[col * incy]);
        }
    }
    return result;
}

TEST(Sger, naive_3x5) {
    const int m = 3;
    const int n = 5;
    const int lda = 3;
    const int incx = 2;
    const int incy = 1;
    float alpha = 1.1f;
    float x[m*incx] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float y[n*incy] = {1.f, 2.f, 3.f, 4.f, 5.f};
    float a[lda*n] = { -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f };

    const float expected[lda*n] = { -1.f+1.1f*1.f, -2.f+1.1f*3.f, -3.f+1.1f*5.f,
                                    -1.f+1.1f*2.f, -2.f+1.1f*6.f, -3.f+1.1f*10.f,
                                    -1.f+1.1f*3.f, -2.f+1.1f*9.f, -3.f+1.1f*15.f,
                                    -1.f+1.1f*4.f, -2.f+1.1f*12.f, -3.f+1.1f*20.f,
                                    -1.f+1.1f*5.f, -2.f+1.1f*15.f, -3.f+1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x, incx, y, incy, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Sger, no_inc_3x5) {
    const int m = 3;
    const int n = 5;
    const int lda = 3;
    const int incx = 1;
    const int incy = 1;
    float alpha = 1.1f;
    float x[m*incx] = { 1.f, 3.f, 5.f };
    float y[n*incy] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    float a[lda*n] = { -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f,
                       -1.f, -2.f, -3.f };

    const float expected[lda*n] = { -1.f + 1.1f*1.f, -2.f + 1.1f*3.f, -3.f + 1.1f*5.f,
        -1.f + 1.1f*2.f, -2.f + 1.1f*6.f, -3.f + 1.1f*10.f,
        -1.f + 1.1f*3.f, -2.f + 1.1f*9.f, -3.f + 1.1f*15.f,
        -1.f + 1.1f*4.f, -2.f + 1.1f*12.f, -3.f + 1.1f*20.f,
        -1.f + 1.1f*5.f, -2.f + 1.1f*15.f, -3.f + 1.1f*25.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x, incx, y, incy, a, lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < lda*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Sger, 17x17_inc2) {
    const int m = 17;
    const int n = 17;
    const int lda = m;
    float alpha = 1.1f;
    const int incx = 2;
    const int incy = 2;

    std::vector<float> x(m * incx);
    for (int i = 0; i < m; i++) {
        x[i*incx] = 1.f * i / m;
    }
    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = (1.f - i) / n;
    }
    std::vector<float> a(n * lda);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            a[IDX(row, col, lda)] = (1.f * IDX(row, col, lda)) / m / n;
        }
    }
    std::vector<float> expected = Sger_cpu(m, n, alpha, x, incx, y, incy, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x.data(), incx, y.data(), incy, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++) {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Sger, 17x17_inc1) {
    const int m = 17;
    const int n = 17;
    const int lda = m;
    float alpha = 1.1f;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> x(m * incx);
    for (int i = 0; i < m; i++) {
        x[i*incx] = 1.f * i / m;
    }
    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = (1.f - i) / n;
    }
    std::vector<float> a(n * lda);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            a[IDX(row, col, lda)] = (1.f * IDX(row, col, lda));
        }
    }
    std::vector<float> expected = Sger_cpu(m, n, alpha, x, incx, y, incy, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x.data(), incx, y.data(), incy, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++) {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Sger, 68x68_inc2) {
    const int m = 68;
    const int n = 68;
    const int lda = m;
    float alpha = 1.1f;
    const int incx = 2;
    const int incy = 2;

    std::vector<float> x(m * incx);
    for (int i = 0; i < m; i++) {
        x[i*incx] = 1.5f * i;
    }
    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = 1.25f * i;
    }
    std::vector<float> a(n * lda);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            a[IDX(row, col, lda)] = (1.f * IDX(row, col, lda)) / m / n;
        }
    }
    std::vector<float> expected = Sger_cpu(m, n, alpha, x, incx, y, incy, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x.data(), incx, y.data(), incy, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++) {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}

TEST(Sger, 68x68_inc1) {
    const int m = 68;
    const int n = 68;
    const int lda = m;
    float alpha = 1.1f;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> x(m * incx);
    for (int i = 0; i < m; i++) {
        x[i*incx] = 1.5f * i;
    }
    std::vector<float> y(n * incy);
    for (int i = 0; i < n; i++) {
        y[i*incy] = 1.25f * i;
    }
    std::vector<float> a(n * lda);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            a[IDX(row, col, lda)] = (1.f * IDX(row, col, lda)) / m / n;
        }
    }
    std::vector<float> expected = Sger_cpu(m, n, alpha, x, incx, y, incy, a, lda);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSger(handle, m, n, &alpha, x.data(), incx, y.data(), incy, a.data(), lda);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n*lda; i++) {
        EXPECT_FLOAT_EQ(expected[i], a[i]);
    }
}
