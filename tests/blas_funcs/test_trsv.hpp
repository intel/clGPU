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

// Helper trmv functions - triangular matrix by vector multiplication

template<typename data_type>
struct accumulator_type { using type = data_type; };

template<>
struct accumulator_type<float>
{
    using type = double;
};

template<>
struct accumulator_type<iclgpu::complex_t>
{
    using type = std::complex<double>;
};

template<typename data_type, typename acc_type>
std::vector<data_type> cpu_trmv_upper_ntrans(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
{
    auto result = x;

    for (int row = 0; row < n; row++)
    {
        acc_type prod = 0.0;
        if (diag == ICLBLAS_DIAG_UNIT)
        {
            prod += static_cast<acc_type>(x[row * incx]);
        }
        else
        {
            prod += static_cast<acc_type>(A[IDX(row, row, lda)]) * static_cast<acc_type>(x[row * incx]);
        }
        for (int col = row + 1; col < n; col++)
        {
            prod += static_cast<acc_type>(A[IDX(row, col, lda)]) * static_cast<acc_type>(x[col * incx]);
        }

        result[row * incx] = static_cast<data_type>(prod);
    }
    return result;
}

template<typename data_type, typename acc_type>
std::vector<data_type> cpu_trmv_upper_trans(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
{
    auto result = x;

    for (int row = 0; row < n; row++)
    {
        acc_type prod = 0.0;
        for (int col = 0; col < row; col++)
        {
            prod += static_cast<acc_type>(A[IDX(col, row, lda)]) * static_cast<acc_type>(x[col * incx]);
        }
        if (diag == ICLBLAS_DIAG_UNIT)
        {
            prod += static_cast<acc_type>(x[row * incx]);
        }
        else
        {
            prod += static_cast<acc_type>(A[IDX(row, row, lda)]) * static_cast<acc_type>(x[row * incx]);
        }

        result[row * incx] = static_cast<data_type>(prod);
    }
    return result;
}

template<typename data_type, typename acc_type>
auto cpu_trmv_upper_conj(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
    -> std::enable_if_t<std::is_floating_point<data_type>::value, std::vector<data_type>>
{
    return cpu_trmv_upper_trans<data_type, acc_type>(diag, n, A, lda, x, incx);
}


template<typename data_type, typename acc_type>
auto cpu_trmv_upper_conj(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
    -> std::enable_if_t<!std::is_floating_point<data_type>::value, std::vector<data_type>>
{
    auto result = x;

    for (int row = 0; row < n; row++)
    {
        acc_type prod = 0.0;
        for (int col = 0; col < row; col++)
        {
            prod += static_cast<acc_type>(std::conj(A[IDX(col, row, lda)])) * static_cast<acc_type>(x[col * incx]);
        }
        if (diag == ICLBLAS_DIAG_UNIT)
        {
            prod += static_cast<acc_type>(x[row * incx]);
        }
        else
        {
            prod += static_cast<acc_type>(std::conj(A[IDX(row, row, lda)])) * static_cast<acc_type>(x[row * incx]);
        }

        result[row * incx] = static_cast<data_type>(prod);
    }
    return result;
}

template<typename data_type, typename acc_type>
std::vector<data_type> cpu_trmv_lower_ntrans(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
{
    auto result = x;

    for (int row = 0; row < n; row++)
    {
        acc_type prod = 0.0;
        for (int col = 0; col < row; col++)
        {
            prod += static_cast<acc_type>(A[IDX(row, col, lda)]) * static_cast<acc_type>(x[col * incx]);
        }
        if (diag == ICLBLAS_DIAG_UNIT)
        {
            prod += static_cast<acc_type>(x[row * incx]);
        }
        else
        {
            prod += static_cast<acc_type>(A[IDX(row, row, lda)]) * static_cast<acc_type>(x[row * incx]);
        }

        result[row * incx] = static_cast<data_type>(prod);
    }
    return result;
}

template<typename data_type, typename acc_type>
std::vector<data_type> cpu_trmv_lower_trans(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
{
    auto result = x;

    for (int row = 0; row < n; row++)
    {
        acc_type prod = 0.0;
        if (diag == ICLBLAS_DIAG_UNIT)
        {
            prod += static_cast<acc_type>(x[row * incx]);
        }
        else
        {
            prod += static_cast<acc_type>(A[IDX(row, row, lda)]) * static_cast<acc_type>(x[row * incx]);
        }
        for (int col = row + 1; col < n; col++)
        {
            prod += static_cast<acc_type>(A[IDX(col, row, lda)]) * static_cast<acc_type>(x[col * incx]);
        }

        result[row * incx] = static_cast<data_type>(prod);
    }
    return result;
}

template<typename data_type, typename acc_type>
auto cpu_trmv_lower_conj(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
    -> std::enable_if_t<std::is_floating_point<data_type>::value, std::vector<data_type>>
{
    return cpu_trmv_lower_trans<data_type, acc_type>(diag, n, A, lda, x, incx);
}

template<typename data_type, typename acc_type>
auto cpu_trmv_lower_conj(
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
    -> std::enable_if_t<!std::is_floating_point<data_type>::value, std::vector<data_type>>
{
    auto result = x;

    for (int row = 0; row < n; row++)
    {
        acc_type prod = 0.0;
        if (diag == ICLBLAS_DIAG_UNIT)
        {
            prod += static_cast<acc_type>(x[row * incx]);
        }
        else
        {
            prod += static_cast<acc_type>(std::conj(A[IDX(row, row, lda)])) * static_cast<acc_type>(x[row * incx]);
        }
        for (int col = row + 1; col < n; col++)
        {
            prod += static_cast<acc_type>(std::conj(A[IDX(col, row, lda)])) * static_cast<acc_type>(x[col * incx]);
        }

        result[row * incx] = static_cast<data_type>(prod);
    }
    return result;
}

template<typename data_type, typename acc_type = typename accumulator_type<data_type>::type>
std::vector<data_type> cpu_trmv(
    const iclblasFillMode_t uplo,
    const iclblasOperation_t trans,
    const iclblasDiagType_t diag,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const std::vector<data_type>& x,
    const int incx)
{
    if (uplo == ICLBLAS_FILL_MODE_UPPER)
    {
        if (trans == ICLBLAS_OP_N)
        {
            return cpu_trmv_upper_ntrans<data_type, acc_type>(diag, n, A, lda, x, incx);
        }
        else if (trans == ICLBLAS_OP_T)
        {
            return cpu_trmv_upper_trans<data_type, acc_type>(diag, n, A, lda, x, incx);
        }
        else
        {
            return cpu_trmv_upper_conj<data_type, acc_type>(diag, n, A, lda, x, incx);
        }
    }
    else
    {
        if (trans == ICLBLAS_OP_N)
        {
            return cpu_trmv_lower_ntrans<data_type, acc_type>(diag, n, A, lda, x, incx);
        }
        else if (trans == ICLBLAS_OP_T)
        {
            return cpu_trmv_lower_trans<data_type, acc_type>(diag, n, A, lda, x, incx);
        }
        else
        {
            return cpu_trmv_lower_conj<data_type, acc_type>(diag, n, A, lda, x, incx);
        }
    }
}

template<typename data_type>
data_type row_asum(
    const iclblasFillMode_t uplo,
    const iclblasOperation_t trans,
    const int n,
    const std::vector<data_type>& A,
    const int lda,
    const int row)
{
    data_type result = 0;
    if (uplo == ICLBLAS_FILL_MODE_UPPER)
    {
        if (trans == ICLBLAS_OP_N)
        {
            for (int col = row + 1; col < n; col++)
            {
                result += std::abs(A[IDX(row, col, lda)]);
            }
        }
        else
        {
            for (int col = 0; col < row; col++)
            {
                result += std::abs(A[IDX(col, row, lda)]);
            }
        }
    }
    else
    {
        if (trans == ICLBLAS_OP_N)
        {
            for (int col = 0; col < row; col++)
            {
                result += std::abs(A[IDX(row, col, lda)]);
            }
        }
        else
        {
            for (int col = row + 1; col < n; col++)
            {
                result += std::abs(A[IDX(col, row, lda)]);
            }
        }
    }
    return result;
}

//

namespace iclgpu { namespace tests {

template <class Func>
struct test_trsv : test_base_TMV<Func>
{
    using data_type = typename test_base_TMV<Func>::data_type;
    using data_arr_type = typename test_base_TMV<Func>::data_arr_type;

    data_arr_type A;
    data_arr_type b;

    void init_values() override
    {
        A = get_random_vector<data_type>(this->num * this->lda);
        if (this->diag == ICLBLAS_DIAG_NON_UNIT)
        {
            // Set diagonal elements to ensure diagonal dominant matrix
            for (int i = 0; i < this->num; i++)
            {
                // Diagonal dominant -> |A[i, i]| >= sum |A[i, :]|
                auto diagonal = row_asum<data_type>(this->uplo, this->trans, this->num, A, this->lda, i);
                // Make sure that matrix is not singular by adding random, bigger than zero value
                diagonal += std::abs(get_random_scalar<data_type>()) + data_type(0.25);
                A[IDX(i, i, this->lda)] = diagonal;
            }
        }
        b = cpu_trmv<data_type>(
            this->uplo,
            this->trans,
            this->diag,
            this->num,
            this->A,
            this->lda,
            this->x,
            this->incx
        );
    }

    typename Func::params get_params() override
    {
        return
        {
            this->uplo,
            this->trans,
            this->diag,
            this->num,
            this->A.data(),
            this->lda,
            this->b.data(),
            this->incx
        };
    }
    typename Func::params get_params_ref() override
    {
        return
        {
            this->uplo,
            this->trans,
            this->diag,
            this->num,
            this->A.data(),
            this->lda,
            this->b.data(),
            this->incx
        };
    }
};

}}
