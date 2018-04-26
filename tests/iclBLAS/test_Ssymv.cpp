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

#define ENABLE_4K_TESTS 0

#define MAT_ACCESS(A, col, row, n) A[col * n + row]

TEST(Ssymv, Ssymv_opt_locgr_1_upper_252)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 252;
    const int lda = 252;
    const int incx = 1;
    const int incy = 1;

    float alpha = 8.f;
    float beta = 4.f;

    std::vector<float> A(lda*n);
    for (int i = 0; i < n*n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 1.f * i / n / n;
    }

    std::vector<float> y(n);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i / n / n;
    }

    /* CPU Ssymv */
    std::vector<float> ex_result(n);
    for (int i = 0; i < n; i++)
    {
        ex_result.data()[i] = 0.f;
        for (int j = 0; j < n; j++)
        {
            if (j >= i)
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, j, i, lda) * x[j]);
            else
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, i, j, lda) * x[j]);
        }
        ex_result[i] += beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

TEST(Ssymv, Ssymv_opt_locgr_1_lower_252)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 252;
    const int lda = n;
    const int incx = 1;
    const int incy = 1;

    float alpha = 8.f;
    float beta = 4.f;

    std::vector<float> A(n*n);
    for (int i = 0; i < n*n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 1.f * i / n / n;
    }

    std::vector<float> y(n);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i / n / n;
    }

    /* CPU Ssymv */
    std::vector<float> ex_result(n);
    for (int i = 0; i < n; i++)
    {
        ex_result.data()[i] = 0.f;
        for (int j = 0; j < n; j++)
        {
            if (i >= j)
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, j, i, n) * x[j]);
            else
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, i, j, n) * x[j]);
        }
        ex_result[i] += beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

TEST(Ssymv, Ssymv_opt_locgr_1_lower_512)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;

    const int n = 512;
    const int lda = n;
    const int incx = 1;
    const int incy = 1;

    float alpha = 8.f;
    float beta = 4.f;

    std::vector<float> A(n*n);
    for (int i = 0; i < n*n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 1.f * i / n / n;
    }

    std::vector<float> y(n);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i / n / n;
    }

    /* CPU Ssymv */
    std::vector<float> ex_result(n);
    for (int i = 0; i < n; i++)
    {
        ex_result.data()[i] = 0.f;
        for (int j = 0; j < n; j++)
        {
            if (i >= j)
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, j, i, n) * x[j]);
            else
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, i, j, n) * x[j]);
        }
        ex_result[i] += beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

TEST(Ssymv, Ssymv_opt_locgr_1_upper_512)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 512;
    const int lda = n;
    const int incx = 1;
    const int incy = 1;

    float alpha = 8.f;
    float beta = 4.f;

    std::vector<float> A(n*n);
    for (int i = 0; i < n*n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n);
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = 1.f * i / n / n;
    }

    std::vector<float> y(n);
    for (int i = 0; i < n; i++)
    {
        y[i*incy] = 1.f * i / n / n;
    }

    /* CPU Ssymv */
    std::vector<float> ex_result(n);
    for (int i = 0; i < n; i++)
    {
        ex_result.data()[i] = 0.f;
        for (int j = 0; j < n; j++)
        {
            if (j >= i)
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, j, i, n) * x[j]);
            else
                ex_result.data()[i] += (alpha * MAT_ACCESS(A, i, j, n) * x[j]);
        }
        ex_result[i] += beta * y[i];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

#if 1
TEST(Ssymv, Ssymv_opt_locgr_1_upper_252_incx)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 252;
    const int lda = 252;
    const int incx = 2;
    const int incy = 2;

    float alpha = 8.f;
    float beta = 4.f;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }


    std::vector<float> ex_result(n * incy);
    ex_result = y;

    /* CPU Ssymv */
    for (int i = 0; i < n; ++i)
    {
        ex_result.data()[i * incy] = 0.f;
        for (int j = 0; j < n; ++j)
        {
            if (j >= i)
                ex_result.data()[i * incy] += (alpha * MAT_ACCESS(A, j, i, lda) * x[j * incx]);
            else
                ex_result.data()[i * incy] += (alpha * MAT_ACCESS(A, i, j, lda) * x[j * incx]);
        }
        ex_result[i * incy] += beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

#endif

#if 1
TEST(Ssymv, Ssymv_opt_locgr_1_upper_512_incx)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 512;
    const int lda = 512;
    const int incx = 2;
    const int incy = 2;

    float alpha = 8.f;
    float beta = 4.f;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }


    std::vector<float> ex_result(n * incy);
    ex_result = y;

    /* CPU Ssymv */
    for (int i = 0; i < n; ++i)
    {
        ex_result.data()[i * incy] = 0.f;
        for (int j = 0; j < n; ++j)
        {
            if (j >= i)
                ex_result.data()[i * incy] += (alpha * MAT_ACCESS(A, j, i, lda) * x[j * incx]);
            else
                ex_result.data()[i * incy] += (alpha * MAT_ACCESS(A, i, j, lda) * x[j * incx]);
        }
        ex_result[i * incy] += beta * y[i * incy];
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

#endif

#if 1
TEST(Ssymv, Ssymv_opt_locgr_1_upper_252_incx_beta0)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;

    const int n = 252;
    const int lda = 252;
    const int incx = 2;
    const int incy = 2;

    float alpha = 8.f;
    float beta = 0;

    std::vector<float> A(lda*n);
    for (int i = 0; i < lda * n; i++)
    {
        A[i] = 1.f * i / n / n;
    }

    std::vector<float> x(n * incx);
    for (int i = 0; i < n * incx; i++)
    {
        x[i] = 1.f * i / n / n;
    }

    std::vector<float> y(n * incy);
    for (int i = 0; i < n * incy; i++)
    {
        y[i] = 1.f * i / n / n;
    }


    std::vector<float> ex_result(n * incy);
    ex_result = y;

    /* CPU Ssymv */
    for (int i = 0; i < n; ++i)
    {
        ex_result.data()[i * incy] = 0.f;
        for (int j = 0; j < n; ++j)
        {
            if (j >= i)
                ex_result.data()[i * incy] += (alpha * MAT_ACCESS(A, j, i, lda) * x[j * incx]);
            else
                ex_result.data()[i * incy] += (alpha * MAT_ACCESS(A, i, j, lda) * x[j * incx]);
        }
       
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n * incy; ++i)
    {
        EXPECT_NEAR(ex_result.data()[i], y.data()[i], 1.e-4f);
    }
}

#endif

#undef MAT_ACCESS
#undef ENABLE_4K_TESTS