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
#include <gtest_utils.hpp>

TEST(Cscal, naive_5x1)
{
    const int num = 5;
    const int incx = 1;
    oclComplex_t alpha = { 1.5f, 2.f };
    oclComplex_t x[num*incx] = { { 1.f, 1.f }, { 2.f, 2.f }, { 3.f, 3.f }, { 4.f, 4.f }, { 5.f, 5.f } };
    oclComplex_t ref_x[num*incx] = {
        { 1.f * 1.5f - 1.f * 2.f,  1.f *  2.f + 1.f * 1.5f },
        { 2.f * 1.5f - 2.f * 2.f,  2.f *  2.f + 2.f * 1.5f },
        { 3.f * 1.5f - 3.f * 2.f,  3.f *  2.f + 3.f * 1.5f },
        { 4.f * 1.5f - 4.f * 2.f,  4.f *  2.f + 4.f * 1.5f },
        { 5.f * 1.5f - 5.f * 2.f,  5.f *  2.f + 5.f * 1.5f } };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;
    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCscal(handle, num, &alpha, x, incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_ARRAYS_EQ(oclComplex_t, ref_x, x);
}

TEST(Cscal, incx_2)
{
    const int n = 175;
    const int incx = 2;

    std::complex<float> x[n * incx];
    std::complex<float> ref_x[n * incx];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    std::complex<float> alpha = { 1.25f, .75f };

    for (int i = 0; i < n; i++)
    {
        ref_x[i * incx] = x[i * incx] * alpha;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCscal(handle, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x), incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_COMPLEX_EQ(ref_x[i * incx], x[i * incx]);
    }
}

TEST(Cscal, noinc)
{
    const int n = 150;
    const int incx = 1;

    std::complex<float> x[n * incx];
    std::complex<float> ref_x[n * incx];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    std::complex<float> alpha = { 1.25f, .75f };

    for (int i = 0; i < n; i++)
    {
        ref_x[i * incx] = x[i * incx] * alpha;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCscal(handle, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x), incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_COMPLEX_EQ(ref_x[i * incx], x[i * incx]);
    }
}

TEST(Cscal, optim)
{
    const int n = 65536;
    const int incx = 1;

    std::complex<float> *x = new std::complex<float>[n * incx];
    std::complex<float> *ref_x = new std::complex<float>[n * incx];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    std::complex<float> alpha = { .25f, 1.75f };

    for (int i = 0; i < n; i++)
    {
        ref_x[i * incx] = x[i * incx] * alpha;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCscal(handle, n, reinterpret_cast<oclComplex_t*>(&alpha), reinterpret_cast<oclComplex_t*>(x), incx);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_COMPLEX_EQ(ref_x[i * incx], x[i * incx]);
    }
}
