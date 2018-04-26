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

#pragma once

#include <gtest/gtest.h>
#include "test_helpers.hpp"

namespace iclgpu { namespace tests {

template<typename data_type, typename alpha_data_type, bool is_hermitian = false, typename acc_type = accumulator_type_t<data_type>>
void cpu_syr(
    const iclblasFillMode_t uplo,
    const int n,
    const alpha_data_type alpha,
    const data_type* x,
    const int incx,
    data_type* A,
    const int lda)
{
    auto alpha_acc = static_cast<acc_type>(alpha);

    for (int col = 0; col < n; ++col)
    {
        int start_row = uplo == ICLBLAS_FILL_MODE_UPPER ? 0 : col;
        int end_row = uplo == ICLBLAS_FILL_MODE_UPPER ? col + 1 : n;

        auto col_x = static_cast<acc_type>(x[col * incx]);
        if (is_hermitian)
        {
            col_x = blas_conj(col_x);
        }
        for (int row = start_row; row < end_row; ++row)
        {
            auto row_x = static_cast<acc_type>(x[row * incx]);
            auto this_A = static_cast<acc_type>(A[IDX(row, col, lda)]);

            if (is_hermitian && row == col)
            {
                this_A = zeroe_imag(this_A);
            }

            auto prod = col_x * row_x;
            this_A += alpha_acc * prod;

            A[IDX(row, col, lda)] = static_cast<data_type>(this_A);
        }
    }
}

template<class Func>
struct test_syr : test_base_SMV<Func>
{
    using data_type = typename test_base_SMV<Func>::data_type;
    using alpha_data_type = typename func_traits<Func>::alpha_data_type;
    using data_arr_type = typename test_base_SMV<Func>::data_arr_type;

    alpha_data_type alpha;
    data_arr_type A_ref;

    void init_values() override
    {
        alpha = get_random_scalar<alpha_data_type>();
        A_ref = this->A;
    }

    typename Func::params get_params() override
    {
        return
        {
            this->uplo,
            this->n,
            this->alpha,
            this->x.data(),
            this->incx,
            this->A.data(),
            this->lda
        };
    }
    typename Func::params get_params_ref() override
    {
        return
        {
            this->uplo,
            this->n,
            this->alpha,
            this->x.data(),
            this->incx,
            this->A_ref.data(),
            this->lda
        };
    }
};

} } // namespace iclgpu::tests
