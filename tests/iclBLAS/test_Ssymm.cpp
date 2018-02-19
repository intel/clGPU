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

TEST(Ssymm, 3x3_left_up) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto  uplo = ICLBLAS_FILL_MODE_UPPER;
    const int m = 3;
    const int n = 3;
    float alpha = 1.1f;
    float beta = 1.3f;

    const int lda = 4;
    float A[lda*m] = { 1.f, -1.f, -2.f, -3.f,
                       2.f, 3.f, -4.f, -5.f,
                       4.f, 5.f, 6.f, -6.f };
    const int ldb = 3;
    float B[ldb*n] = { 7.f, 8.f, 9.f,
                       10.f, 11.f, 12.f,
                       13.f, 14.f, 15.f };
    const int ldc = 3;
    float C[ldc*n] = { 16.f, 17.f, 18.f,
                       19.f, 20.f, 21.f,
                       22.f, 23.f, 24.f };

    const float expected[ldc*n] = { 1.3f * 16.f + 1.1f * 59.f, 1.3f * 17.f + 1.1f * 83.f, 1.3f * 18.f + 1.1f * 122.f,
                                    1.3f * 19.f + 1.1f * 80.f, 1.3f * 20.f + 1.1f * 113.f, 1.3f * 21.f + 1.1f * 167.f, 
                                    1.3f * 22.f + 1.1f * 101.f, 1.3f * 23.f + 1.1f * 143.f, 1.3f * 24.f + 1.1f * 212.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

TEST(Ssymm, 3x3_left_low) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto  uplo = ICLBLAS_FILL_MODE_LOWER;
    const int m = 3;
    const int n = 3;
    float alpha = 1.1f;
    float beta = 1.3f;

    const int lda = 3;
    float A[lda*m] = { 1.f, 2.f, 4.f,
                      -1.f, 3.f, 5.f,
                       -2.f, -3.f, 6.f };
    const int ldb = 4;
    float B[ldb*n] = { 7.f, 8.f, 9.f, -1.f,
        10.f, 11.f, 12.f, -2.f,
        13.f, 14.f, 15.f, -3.f };
    const int ldc = 3;
    float C[ldc*n] = { 16.f, 17.f, 18.f,
        19.f, 20.f, 21.f,
        22.f, 23.f, 24.f };

    const float expected[ldc*n] = { 1.3f * 16.f + 1.1f * 59.f, 1.3f * 17.f + 1.1f * 83.f, 1.3f * 18.f + 1.1f * 122.f,
        1.3f * 19.f + 1.1f * 80.f, 1.3f * 20.f + 1.1f * 113.f, 1.3f * 21.f + 1.1f * 167.f,
        1.3f * 22.f + 1.1f * 101.f, 1.3f * 23.f + 1.1f * 143.f, 1.3f * 24.f + 1.1f * 212.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

TEST(Ssymm, 3x3_right_up) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto  uplo = ICLBLAS_FILL_MODE_UPPER;
    const int m = 3;
    const int n = 3;
    float alpha = 1.1f;
    float beta = 1.3f;

    const int lda = 3;
    float A[lda*n] = { 1.f, -1.f, -2.f,
        2.f, 3.f, -4.f,
        4.f, 5.f, 6.f };
    const int ldb = 3;
    float B[ldb*n] = { 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f,
        13.f, 14.f, 15.f };
    const int ldc = 4;
    float C[ldc*n] = { 16.f, 17.f, 18.f, -1.f,
        19.f, 20.f, 21.f, -2.f,
        22.f, 23.f, 24.f, -3.f };

    const float expected[ldc*n] = { 1.3f * 16.f + 1.1f * 79.f, 1.3f * 17.f + 1.1f * 86.f, 1.3f * 18.f + 1.1f * 93.f, -1.f,
        1.3f * 19.f + 1.1f * 109.f, 1.3f * 20.f + 1.1f * 119.f, 1.3f * 21.f + 1.1f * 129.f, -2.f,
        1.3f * 22.f + 1.1f * 156.f, 1.3f * 23.f + 1.1f * 171.f, 1.3f * 24.f + 1.1f * 186.f, -3.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

TEST(Ssymm, 4x3_right_low) {
    const auto side = ICLBLAS_SIDE_RIGHT;
    const auto  uplo = ICLBLAS_FILL_MODE_LOWER;
    const int m = 4;
    const int n = 3;
    float alpha = -1.1f;
    float beta = 1.3f;

    const int lda = 3;
    float A[lda*n] = { 1.f, 2.f, 3.f,
        -1.f, 4.f, 5.f,
        -2.f, -3.f, 6.f };
    const int ldb = 4;
    float B[ldb*n] = { 7.f, 8.f, 9.f, 10.f,
                       11.f, 12.f, 13.f, 14.f,
                       15.f, 16.f, 17.f, 18.f };
    const int ldc = 4;
    float C[ldc*n] = { 19.f, 20.f, 21.f, 22.f,
                       23.f, 24.f, 25.f, 26.f,
                       27.f, 28.f, 29.f, 30.f};

    const float expected[ldc*n] = { 1.3f * 19.f - 1.1f * 74.f, 1.3f * 20.f - 1.1f * 80.f, 1.3f * 21.f - 1.1f * 86.f, 1.3f*22.f - 1.1f * 92.f,
        1.3f * 23.f - 1.1f * 133.f, 1.3f * 24.f - 1.1f * 144.f, 1.3f * 25.f - 1.1f * 155.f, 1.3f * 26.f - 1.1f*166.f,
        1.3f * 27.f - 1.1f * 166.f, 1.3f * 28.f - 1.1f * 180.f, 1.3f * 29.f - 1.1f * 194.f, 1.3f*30.f - 1.1f*208.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_FLOAT_EQ(expected[i], C[i]);
    }
}

#define ACCESS(A, m, n, N) A[(n)*(N) + (m)]

std::vector<float> Ssymm_left_up_reference(
    const int m,
    const int n,
    const float alpha_f,
    const std::vector<float>& Af,
    const int lda,
    const std::vector<float>& Bf,
    const int ldb,
    const float beta_f,
    const std::vector<float>& Cf,
    const int ldc)
{
    double alpha = alpha_f;
    double beta = beta_f;
    std::vector<double> A(Af.size());
    std::copy(std::begin(Af), std::end(Af), std::begin(A));

    std::vector<double> B(Bf.size());
    std::copy(std::begin(Bf), std::end(Bf), std::begin(B));

    std::vector<double> C(Cf.size());
    std::copy(std::begin(Cf), std::end(Cf), std::begin(C));

    std::vector<double> expected(C);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double value = 0.;
            for (int k = 0; k < m; k++) {
                if (k < i) {
                    value += ACCESS(A, k, i, lda)*ACCESS(B, k, j, ldb);
                }
                else {
                    value += ACCESS(A, i, k, lda)*ACCESS(B, k, j, ldb);
                }
            }
            value *= alpha;
            ACCESS(expected, i, j, ldc) *= beta;
            ACCESS(expected, i, j, ldc) += value;
        }
    }
    std::vector<float> res(expected.size());
    for(size_t i = 0; i < res.size(); i++)
    {
        res[i] = static_cast<float>(expected[i]);
    }
    return res;
}

TEST(Ssymm, 256x256_left_up) {
    const auto side = ICLBLAS_SIDE_LEFT;
    const auto  uplo = ICLBLAS_FILL_MODE_UPPER;
    const int m = 256;
    const int n = 256;
    float alpha = 0.9f;
    float beta = -1.2f;

    const int lda = 256;
    std::vector<float> A(lda*m, 1.2f);
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            A[j*lda + i] = (float)(i*j + i + j)/(m*m + 2*m);
        }
    }
    const int ldb = 256;
    std::vector<float> B(ldb*n, 0.9f);
    for (int i = 0; i < ldb*n; i++) {
        B[i] = 1.f * i / ldb / n;
    }

    const int ldc = 256;
    std::vector<float> C(ldc*n, 2.3f);

    auto expected = Ssymm_left_up_reference(m, n, alpha, A, lda, B, ldb, beta, C, ldc);

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSsymm(handle, side, uplo, m, n, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < ldc*n; i++) {
        EXPECT_NEAR(expected[i], C[i], 1.e-4f);
    }
}
