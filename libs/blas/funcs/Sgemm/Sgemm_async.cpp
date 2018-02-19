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

#include "functions/Sgemm.hpp"

static const char* module_name = "Sgemm_async";
static const char* kernel_name = "Sgemm_async";

namespace iclgpu { namespace functions { namespace implementations {

bool Sgemm_async::accept(const Sgemm::params& params, Sgemm::score& score)
{
    score.n = 1.1f;
    score.m = 1.1f;

    return true;
}

event Sgemm_async::execute(const Sgemm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    size_t a_buf_size = params.k * params.lda;
    size_t b_buf_size = params.n * params.ldb;
    size_t c_buf_size = params.n * params.ldc;

    kernel->set_arg(0, params.transa);
    kernel->set_arg(1, params.transb);
    kernel->set_arg(2, params.k);
    kernel->set_arg(3, params.alpha);
    auto buf_A = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(4, buf_A);
    kernel->set_arg(5, params.lda);
    auto buf_B = engine->get_input_buffer(params.B, b_buf_size);
    kernel->set_arg(6, buf_B);
    kernel->set_arg(7, params.ldb);
    kernel->set_arg(8, params.beta);
    auto buf_C = engine->get_inout_buffer(params.C, c_buf_size);
    kernel->set_arg(9, buf_C);
    kernel->set_arg(10, params.ldc);

    auto gws = nd_range(params.m, params.n);
    kernel->set_options(kernel_options(gws));

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
