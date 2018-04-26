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

#include "test_helpers.hpp"
#include "test_gemm.hpp"

#include "functions/Sgemm.hpp"
#include "functions/Cgemm.hpp"

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Sgemm>
{
    using data_type = float;
    static void reference(iclgpu::functions::Sgemm::params& params)
    {
        cpu_gemm<data_type>(params.transa, params.transb, params.m, params.n, params.k, params.alpha,
                            params.A, params.lda, params.B, params.ldb, params.beta, params.C, params.ldc);
    }
};

using test_Sgemm = test_gemm<iclgpu::functions::Sgemm>;

TEST_P(test_Sgemm, basic)
{

    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Sgemm>(params, impl_name));
    EXPECT_EQ(C.size(), C_ref.size());
    size_t errors = 0;
    for (size_t i = 0; i < C.size() && errors < 100; i++)
    {
        EXPECT_EQ(C[i], C_ref[i]);

        if (C[i] != C_ref[i])
            ++errors;
    }
}

INSTANTIATE_TEST_CASE_P(
    s_m256_n256_k256,
    test_Sgemm,
    Combine(
        Values("n3_sg_ntransAB", "ntransAB"),               // impl_name
        Values(ICLBLAS_OP_N),                               // transa
        Values(ICLBLAS_OP_N),                               // transb
        Values(256),                                        // m
        Values(256),                                        // n
        Values(256),                                        // k
        Values(0, 29),                                      // lda_add
        Values(0, 29),                                      // ldb_add
        Values(false, true),                                // beta_zero
        Values(0, 29)                                       // ldc_add
    ),
    testing::internal::DefaultParamName<test_Sgemm::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Cgemm>
{
    using data_type = iclgpu::complex_t;
    static void reference(iclgpu::functions::Cgemm::params& params)
    {
        cpu_gemm<data_type>(params.transa, params.transb, params.m, params.n, params.k, params.alpha,
                            params.A, params.lda, params.B, params.ldb, params.beta, params.C, params.ldc);
    }
};

using test_Cgemm = test_gemm<iclgpu::functions::Cgemm>;

TEST_P(test_Cgemm, basic)
{

    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Cgemm>(params, impl_name));
    EXPECT_EQ(C.size(), C_ref.size());
    size_t errors = 0;
    for (size_t i = 0; i < C.size() && errors < 100; i++)
    {
        EXPECT_COMPLEX_EQ(C[i], C_ref[i]);

        if (C[i] != C_ref[i])
            ++errors;
    }
}

// The test case is currently to slow for naive kernel.
// It will be re-enabled once the optimized kernel for complex GEMM will be added.
/*
INSTANTIATE_TEST_CASE_P(
    c_m256_n256_k256,
    test_Cgemm,
    Combine(
        Values(""),                                         // impl_name
        Values(ICLBLAS_OP_N, ICLBLAS_OP_T, ICLBLAS_OP_C),   // transa
        Values(ICLBLAS_OP_N, ICLBLAS_OP_T, ICLBLAS_OP_C),   // transb
        Values(256),                                        // m
        Values(256),                                        // n
        Values(256),                                        // k
        Values(0, 29),                                      // lda_add
        Values(0, 29),                                      // ldb_add
        Values(false, true),                                // beta_zero
        Values(0, 29)                                       // ldc_add
    ),
    testing::internal::DefaultParamName<test_Cgemm::ParamType>
);
*/

}}
