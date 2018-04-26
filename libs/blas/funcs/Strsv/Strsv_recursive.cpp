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

#include "functions/Strsv.hpp"

#include "functions/Sgemv.hpp"

#include "dispatcher.hpp"

// Recursive implementation for solving linear system of equations with single right hand side for trianguluar matrix:
// A * x = b
//
// Matrix A, vectors x and b are divided as follows:
// | A11  0  |   | x1 |   | b1 |
// | A21 A22 | * | x2 | = | b2 |
//
// From this two equations, one for each unknown, can be derived:
// 1) A11 * x1 = b1
// 2) A21 * x1 + A22 * x2 = b2
// => A22 * x2 = b2 - A21 * x1
// => A22 * x2 = b2'
//
// They can be implemented by combining other BLAS functions for smaller problem sizes as follows:
// solve A11 * x1 = b1 for x1 - Strsv
// calculate b2` := b2 - A21 * x1 - Sgemv
// solve A22 * x2 = b2` for x2 - Strsv
//
// We can calculate b2' in place as target array will contain x2 at the end and b2 is not needed anywhere else.
// Matrix A of size n x n will be divided if n is at least split_size.

#define ICLBLAS_FILL_MODE_UPPER (0)
#define ICLBLAS_OP_N (0)

static const int split_size = 32;

namespace iclgpu { namespace functions { namespace implementations {

bool Strsv_recursive::accept(const Strsv::params& params, Strsv::score& score)
{
    if (params.n < split_size) return false;
    score.n = 1.2f;
    return false;
}

event Strsv_recursive::execute(const Strsv::params& params, const std::vector<event>& dep_events)
{
    auto dispatcher = context()->get_dispatcher();

    int first_size = params.n / 2;
    // Round first_size to next power of 2
    first_size--;
    first_size |= first_size >> 1;
    first_size |= first_size >> 2;
    first_size |= first_size >> 4;
    first_size |= first_size >> 8;
    first_size |= first_size >> 16;
    first_size++;

    int second_size = params.n - first_size;

    int multiply_m = second_size;
    int multiply_n = first_size;

    float* first_solve_A = params.A;
    float* multiply_A = params.A;
    float* second_solve_A = params.A;
    float* solve_x = params.x;
    float* rest_x = params.x;

    if (params.uplo == ICLBLAS_FILL_MODE_UPPER)
    {
        if (params.trans == ICLBLAS_OP_N)
        {
            // Swap first and second size to maintain good alignment
            std::swap(first_size, second_size);
            std::swap(multiply_m, multiply_n);
            first_solve_A += second_size + second_size * params.lda;
            multiply_A += second_size * params.lda;
            solve_x += second_size * params.incx;
        }
        else
        {
            std::swap(multiply_m, multiply_n);
            multiply_A += first_size * params.lda;
            second_solve_A += first_size + first_size * params.lda;
            rest_x += first_size * params.incx;
        }
    }
    else
    {
        if (params.trans == ICLBLAS_OP_N)
        {
            multiply_A += first_size;
            second_solve_A += first_size + first_size * params.lda;
            rest_x += first_size * params.incx;
        }
        else
        {
            // Swap first and second size to maintain good alignment
            std::swap(first_size, second_size);
            first_solve_A += second_size + second_size * params.lda;
            multiply_A += second_size;
            solve_x += second_size * params.incx;
        }
    }

    Strsv::params first_solve_params = { params.uplo, params.trans, params.diag, first_size, first_solve_A, params.lda, solve_x, params.incx };
    auto event = dispatcher->execute_function<Strsv>(first_solve_params, dep_events);

    Sgemv::params multiplication_params = { params.trans, multiply_m, multiply_n, -1.f, multiply_A, params.lda, solve_x, params.incx, 1.f,  rest_x, params.incx };
    auto event2 = dispatcher->execute_function<Sgemv>(multiplication_params, { event });

    Strsv::params second_solve_params = { params.uplo, params.trans, params.diag, second_size, second_solve_A, params.lda, rest_x, params.incx };
    auto event3 = dispatcher->execute_function<Strsv>(second_solve_params, { event2 });

    return event3;
}

} } } // namespace iclgpu::functions::implementations
