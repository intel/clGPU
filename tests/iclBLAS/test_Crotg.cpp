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
#include <gtest_utils.hpp>

TEST(Crotg, naive_5x2)
{
    oclComplex_t a = { 1.f, 0.f };
    oclComplex_t b = { 1.5f, 0.f };
    float c = 0.5f;
    oclComplex_t s = { 1.f, 0.f };

    oclComplex_t ref_a = { 1.802776f, 0.f };
    oclComplex_t ref_b = { 1.5f, 0.f };
    float ref_c = 0.554700f;
    oclComplex_t ref_s = { 0.8320503f, 0.f };

    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasCrotg(handle, &a, &b, &c, &s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    EXPECT_COMPLEX_EQ(ref_a, a);
    EXPECT_COMPLEX_EQ(ref_b, b);
    EXPECT_FLOAT_EQ(ref_c, c);
    EXPECT_COMPLEX_EQ(ref_s, s);
}
