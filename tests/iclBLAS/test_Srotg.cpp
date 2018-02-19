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

TEST(Srotg, naive_5x2)
{
    float a = 1.f;
    float b = 1.5f;

    float c = 0.5f;
    float s = 1.0f;
    
    float ref_a = 1.802776f;
    float ref_b = 1.802776f;
    float ref_c = 0.554700f;
    float ref_s = 0.8320503f;
    
    iclblasHandle_t handle;
    iclblasStatus_t status = ICLBLAS_STATUS_SUCCESS;

    status = iclblasCreate(&handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasSrotg(handle, &a, &b, &c, &s);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    status = iclblasDestroy(handle);
    ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);

    ASSERT_FLOAT_EQ(ref_a, a);
    ASSERT_FLOAT_EQ(ref_b, b);
    ASSERT_FLOAT_EQ(ref_c, c);
    ASSERT_FLOAT_EQ(ref_s, s);
}
