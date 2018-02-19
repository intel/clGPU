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

#include "functions/Cgemm.hpp"

static const char* module_name = "Cgemm_naive";
static const char* kernel_name = "Cgemm_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Cgemm_naive::accept(const Cgemm::params& params, Cgemm::score& score)
{
    return true;
}

event Cgemm_naive::execute(const Cgemm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    auto a_buf_size = params.k * params.lda;
    auto b_buf_size = params.n * params.ldb;
    auto c_buf_size = params.n * params.ldc;

    kernel->set_arg(0, params.transa);
    kernel->set_arg(1, params.transb);
    kernel->set_arg(2, params.m);
    kernel->set_arg(3, params.n);
    kernel->set_arg(4, params.k);
    kernel->set_arg(5, params.alpha);
    auto buf_A = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(6, buf_A);
    kernel->set_arg(7, params.lda);
    auto buf_B = engine->get_input_buffer(params.B, b_buf_size);
    kernel->set_arg(8, buf_B);
    kernel->set_arg(9, params.ldb);
    kernel->set_arg(10, params.beta);
    auto buf_C = engine->get_inout_buffer(params.C, c_buf_size);
    kernel->set_arg(11, buf_C);
    kernel->set_arg(12, params.ldc);

    auto gws = nd_range(1);
    auto lws = null_range;
    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
