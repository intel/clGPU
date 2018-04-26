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

#define MAT_ACCESS(A, row, col, N) A[col*N + row]

TEST(Sgemv, N_256x256) {
    iclblasOperation_t uplo = ICLBLAS_OP_N;
    const int n = 256;
    const int m = 256;
    int lda = 256;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n*incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(m*incy);
    for (int i = 0; i < m * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 1.f;

    std::vector<float> eq_result(m*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < m; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < n; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, row_id, col_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result + beta * y[row_id * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m * incy; ++i)
    {
        EXPECT_NEAR(eq_result[i], y[i], 0.00001f);
    }
}

TEST(Sgemv, N_15x25_inc) {
    iclblasOperation_t uplo = ICLBLAS_OP_N;
    const int n = 20;
    const int m = 25;
    int lda = 25;
    const int incx = 2;
    const int incy = 4;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n*incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(m*incy);
    for (int i = 0; i < m * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 1.f;

    std::vector<float> eq_result(m*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < m; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < n; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, row_id, col_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result + beta * y[row_id * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

TEST(Sgemv, N_35x25_beta0) {
    iclblasOperation_t uplo = ICLBLAS_OP_N;
    const int n = 35;
    const int m = 25;
    int lda = 25;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n*incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(m*incy);
    for (int i = 0; i < m * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 0;

    std::vector<float> eq_result(m*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < m; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < n; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, row_id, col_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

TEST(Sgemv, N_25x25_lda) {
    iclblasOperation_t uplo = ICLBLAS_OP_N;
    const int n = 25;
    const int m = 25;
    int lda = 35;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n*incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(m*incy);
    for (int i = 0; i < m * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 1.f;

    std::vector<float> eq_result(m*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < m; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < n; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, row_id, col_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result + beta * y[row_id * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < m * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

TEST(Sgemv, T_15x25)
{
    iclblasOperation_t uplo = ICLBLAS_OP_T;
    const int n = 15;
    const int m = 25;
    int lda = 25;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> A(lda * n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(m*incx);
    for (int i = 0; i < m * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n*incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 1.f;

    std::vector<float> eq_result(n*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < n; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < m; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, col_id, row_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result + beta * y[row_id * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

TEST(Sgemv, T_15x25_inc)
{
    iclblasOperation_t uplo = ICLBLAS_OP_T;
    const int n = 15;
    const int m = 25;
    int lda = 25;
    const int incx = 2;
    const int incy = 4;

    std::vector<float> A(lda * n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(m*incx);
    for (int i = 0; i < m * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n*incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 1.f;

    std::vector<float> eq_result(n*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < n; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < m; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, col_id, row_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result + beta * y[row_id * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

TEST(Sgemv, T_15x25_beta0)
{
    iclblasOperation_t uplo = ICLBLAS_OP_T;
    const int n = 15;
    const int m = 25;
    int lda = 25;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> A(lda * n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(m*incx);
    for (int i = 0; i < m * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n*incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 0;

    std::vector<float> eq_result(n*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < n; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < m; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, col_id, row_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

TEST(Sgemv, T_15x25_lda)
{
    iclblasOperation_t uplo = ICLBLAS_OP_T;
    const int n = 15;
    const int m = 25;
    int lda = 35;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> A(lda * n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(m*incx);
    for (int i = 0; i < m * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n*incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }

    float alpha = 1.f;
    float beta = 0;

    std::vector<float> eq_result(n*incy);
    eq_result = y;

    //CPU Sgemv
    for (int row_id = 0; row_id < n; ++row_id)
    {
        float l_result = 0;
        for (int col_id = 0; col_id < m; ++col_id)
        {
            l_result += alpha * MAT_ACCESS(A, col_id, row_id, lda) * x[col_id * incx];
        }
        eq_result[row_id * incy] = l_result + beta * y[row_id * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSgemv(handle, uplo, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_FLOAT_EQ(eq_result[i], y[i]);
    }
}

#undef MAT_ACCESS
