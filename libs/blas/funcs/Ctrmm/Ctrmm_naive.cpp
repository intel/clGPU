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

#include "functions/Ctrmm.hpp"

static const char* module_name = "Ctrmm_naive";
static const char* kernel_name = "Ctrmm_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Ctrmm_naive::accept(const Ctrmm::params& params, Ctrmm::score& score)
{
    return true;
}

event Ctrmm_naive::execute(const Ctrmm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    int matrix_a_size = params.lda;
    int matrix_b_size = params.ldb;

    //side == 0 -> Left
    //side == 1 -> Right
    if (params.side == 0) matrix_a_size *= params.m;
    if (params.side == 1) matrix_a_size *= params.n;

    if (params.side == 0) matrix_b_size *= params.n;
    if (params.side == 1) matrix_b_size *= params.m;

    kernel->set_arg(0, params.side);
    kernel->set_arg(1, params.uplo);
    kernel->set_arg(2, params.transa);
    kernel->set_arg(3, params.diag);
    kernel->set_arg(4, params.m);
    kernel->set_arg(5, params.n);
    kernel->set_arg(6, params.alpha);

    auto buf_A = engine->get_input_buffer(params.A, matrix_a_size);
    kernel->set_arg(7, buf_A);
    kernel->set_arg(8, params.lda);

    auto buf_B = engine->get_input_buffer(params.B, matrix_b_size);
    kernel->set_arg(9, buf_B);
    kernel->set_arg(10, params.ldb);

    auto buf_C = engine->get_inout_buffer(params.C, params.lda * params.n);
    kernel->set_arg(11, buf_C);
    kernel->set_arg(12, params.ldc);

    auto gws = nd_range(params.m, params.n);
    auto lws = null_range;
    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } 
