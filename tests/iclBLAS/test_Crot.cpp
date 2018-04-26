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

TEST(Crot, n1_c2_s1)
{
    const int n = 1;
    const int incx = 1;
    const int incy = 1;

    std::complex<float> x[n] = { {1.f, 0.f} };
    std::complex<float> y[n] = { {2.f, 0.f} };

    float c = 2.f;
    std::complex<float> s = { 1.f, 0.f };

    std::complex<float> refx = { 4.f, 0.f };
    std::complex<float> refy = { 3.f, 0.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(y), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_FLOAT_EQ(refx.real(), x[0].real());
    EXPECT_FLOAT_EQ(refx.imag(), x[0].imag());
    EXPECT_FLOAT_EQ(refy.real(), y[0].real());
    EXPECT_FLOAT_EQ(refy.imag(), y[0].imag());
}

TEST(Crot, n11_c2_s1)
{
    const int n = 11;
    const int incx = 1;
    const int incy = 1;

    std::complex<float> x[n * incx] = { { -1.f, -1.f },{ 23.f, 1.f },{ 3.f, 0.f },{ 14.f, 0.f },{ 4.f, 0.f },{ 8.f, -15.25f },{ 7.f, 0.f },{ -11.f, 0.f },{ 9.f, 0.f },{ 10.f, -1.5f } };
    std::complex<float> y[n * incy] = { { -1.f, 0.f },{ 23.f, 12.f },{ 3.f, -1.f },{ 14.f, .5f },{ 4.f, 0.f },{ 8.f, 0.f },{ 7.f, 0.f },{ -11.f, 0.f },{ 9.f, 7.f },{ 10.f, 0.f } };

    float c = 2;
    std::complex<float> s = { 1.f, 0.f };

    std::complex<float> refx[n * incx];
    std::complex<float> refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        std::complex<float> _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1.f * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(y), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i].real(), x[i].real());
        EXPECT_FLOAT_EQ(refx[i].imag(), x[i].imag());
        EXPECT_FLOAT_EQ(refy[i].real(), y[i].real());
        EXPECT_FLOAT_EQ(refy[i].imag(), y[i].imag());
    }
}

TEST(Crot, n5x2_c2_s1)
{
    const int n = 5;
    const int incx = 2;
    const int incy = 2;

    std::complex<float> x[n * incx] = { { -1.f, -1.f },{ 23.f, 1.f },{ 3.f, 0.f },{ 14.f, 0.f },{ 4.f, 0.f },{ 8.f, -15.25f },{ 7.f, 0.f },{ -11.f, 0.f },{ 9.f, 0.f },{ 10.f, -1.5f } };
    std::complex<float> y[n * incy] = { { -1.f, 0.f },{ 23.f, 12.f },{ 3.f, -1.f },{ 14.f, .5f },{ 4.f, 0.f },{ 8.f, 0.f },{ 7.f, 0.f },{ -11.f, 0.f },{ 9.f, 7.f },{ 10.f, 0.f } };

    float c = 2.f;
    std::complex<float> s = { 1.f, 0.f };

    std::complex<float> refx[n * incx];
    std::complex<float> refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        std::complex<float> _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = -1.f * s * x[i * incx] + c * y[i * incy];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(y), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx].real(), x[i * incx].real());
        EXPECT_FLOAT_EQ(refx[i * incx].imag(), x[i * incx].imag());
        EXPECT_FLOAT_EQ(refy[i * incy].real(), y[i * incy].real());
        EXPECT_FLOAT_EQ(refy[i * incy].imag(), y[i * incy].imag());
    }
}

TEST(Crot, noinc)
{
    const int n = 100;
    const int incx = 1;
    const int incy = 1;

    std::complex<float> x[n * incx];
    std::complex<float> y[n * incy];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
        y[i * incy] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    float c = 1.25f;
    std::complex<float> s = { 1.f, -1.5f };

    std::complex<float> refx[n * incx];
    std::complex<float> refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        std::complex<float> _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = c * y[i * incy] - std::conj(s) * x[i * incx];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(y), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx].real(), x[i * incx].real());
        EXPECT_FLOAT_EQ(refx[i * incx].imag(), x[i * incx].imag());
        EXPECT_FLOAT_EQ(refy[i * incy].real(), y[i * incy].real());
        EXPECT_FLOAT_EQ(refy[i * incy].imag(), y[i * incy].imag());
    }
}

TEST(Crot, noincx)
{
    const int n = 100;
    const int incx = 1;
    const int incy = 3;

    std::complex<float> x[n * incx];
    std::complex<float> y[n * incy];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
        y[i * incy] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    float c = 1.25f;
    std::complex<float> s = { .75f, -1.25f };

    std::complex<float> refx[n * incx];
    std::complex<float> refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        std::complex<float> _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = c * y[i * incy] - std::conj(s) * x[i * incx];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(y), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx].real(), x[i * incx].real());
        EXPECT_FLOAT_EQ(refx[i * incx].imag(), x[i * incx].imag());
        EXPECT_FLOAT_EQ(refy[i * incy].real(), y[i * incy].real());
        EXPECT_FLOAT_EQ(refy[i * incy].imag(), y[i * incy].imag());
    }
}

TEST(Crot, noincy)
{
    const int n = 150;
    const int incx = 2;
    const int incy = 1;

    std::complex<float> x[n * incx];
    std::complex<float> y[n * incy];

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
        y[i * incy] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    float c = 1.75f;
    std::complex<float> s = { .5f, -1.25f };

    std::complex<float> refx[n * incx];
    std::complex<float> refy[n * incy];

    for (int i = 0; i < n; i++)
    {
        std::complex<float> _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = c * y[i * incy] - std::conj(s) * x[i * incx];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x), incx, reinterpret_cast<oclComplex_t*>(y), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx].real(), x[i * incx].real());
        EXPECT_FLOAT_EQ(refx[i * incx].imag(), x[i * incx].imag());
        EXPECT_FLOAT_EQ(refy[i * incy].real(), y[i * incy].real());
        EXPECT_FLOAT_EQ(refy[i * incy].imag(), y[i * incy].imag());
    }
}

TEST(Crot, optim)
{
    const int n = 65536;
    const int incx = 3;
    const int incy = 2;

    //use vector to avoid stack size limit problem
    std::vector<std::complex<float>> x(n * incx);
    std::vector<std::complex<float>> y(n * incy);

    for (int i = 0; i < n; i++)
    {
        x[i * incx] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
        y[i * incy] = { static_cast<float>(std::rand() % 15), static_cast<float>(std::rand() % 15) };
    }

    float c = 1.25f;
    std::complex<float> s = { 1.f, -1.5f };

    std::vector<std::complex<float>> refx(n * incx);
    std::vector<std::complex<float>> refy(n * incy);

    for (int i = 0; i < n; i++)
    {
        std::complex<float> _x = c * x[i * incx] + s * y[i * incy];
        refy[i * incy] = c * y[i * incy] - std::conj(s) * x[i * incx];
        refx[i * incx] = _x;
    }

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrot(handle, n, reinterpret_cast<oclComplex_t*>(x.data()), incx, reinterpret_cast<oclComplex_t*>(y.data()), incy, &c, reinterpret_cast<oclComplex_t*>(&s));
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(refx[i * incx].real(), x[i * incx].real());
        EXPECT_FLOAT_EQ(refx[i * incx].imag(), x[i * incx].imag());
        EXPECT_FLOAT_EQ(refy[i * incy].real(), y[i * incy].real());
        EXPECT_FLOAT_EQ(refy[i * incy].imag(), y[i * incy].imag());
    }
}
