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

TEST(Ssymv, naive_test_lower) {
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_LOWER;
    const int n = 2;
    int lda = 2;
    const int incx = 1;
    const int incy = 1;
    float A[n*n] = { 1.f, 
                     3.f, 4.f };
    float x[n] = { 1.f, 2.f };
    float y[n] = { 1.f, 2.f };
    float alpha[] = { 2.f };
    float beta[] = { 4.f };

    float ex_result[n] = { 6.f, 16.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i], y[i]);
    }

}

TEST(Ssymv, naive_test_upper)
{
    iclblasFillMode_t uplo = ICLBLAS_FILL_MODE_UPPER;
    const int n = 2;
    int lda = 2;
    const int incx = 1;
    const int incy = 1;
    float A[n*n] = { 1.f, 2.f,
                          4.f };
    float x[n] = { 1.f, 2.f };
    float y[n] = { 1.f, 2.f };
    float alpha[] = { 2.f };
    float beta[] = { 4.f };

    float ex_result[n] = { 14.f, 8.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(ex_result[i], y[i]);
    }

}
