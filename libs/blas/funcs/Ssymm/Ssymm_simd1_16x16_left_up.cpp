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

#include "functions/Ssymm.hpp"

static const char* module_name = "Ssymm_simd1_16x16_left_up";
static const char* kernel_name = "Ssymm_simd1_16x16_left_up";

namespace iclgpu { namespace functions { namespace implementations {

// TODO: Add handling for matrix sizes not being multiple of 16,
// for eg: calculating the extra parts in naive kernel and running them in parallel

bool Ssymm_simd1_16x16_left_up::accept(const Ssymm::params& params, Ssymm::score& score)
{
    if (params.uplo != 0 || params.side != 0 || params.m % 16 != 0 || params.n % 16 != 0) {
        return false;
    }
    score.m = 1.2f;
    score.n = 1.2f;
    return true;
}

event Ssymm_simd1_16x16_left_up::execute(const Ssymm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t a_buf_size = params.m * params.lda;
    size_t b_buf_size = params.n * params.ldb;
    size_t c_buf_size = params.n * params.ldc;

    kernel->set_arg(0, params.m);
    kernel->set_arg(1, params.n);
    kernel->set_arg(2, params.alpha);
    auto buf_a = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(3, buf_a);
    kernel->set_arg(4, params.lda);
    auto buf_b = engine->get_input_buffer(params.B, b_buf_size);
    kernel->set_arg(5, buf_b);
    kernel->set_arg(6, params.ldb);
    kernel->set_arg(7, params.beta);
    auto buf_c = engine->get_inout_buffer(params.C, c_buf_size);
    kernel->set_arg(8, buf_c);
    kernel->set_arg(9, params.ldc);

    auto gws = nd_range(params.m, params.n/16);
    auto lws = nd_range(16, 1);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    auto event = kernel->submit(dep_events);

    return event;
}

} } } // namespace iclgpu::functions::implementations
