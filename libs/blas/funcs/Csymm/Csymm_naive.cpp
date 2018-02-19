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

#include "functions/Csymm.hpp"

static const char* module_name = "Csymm_naive";
static const char* kernel_name = "Csymm_naive";

namespace iclgpu { namespace functions { namespace implementations {

#define ICLBLAS_SIDE_LEFT 0

bool Csymm_naive::accept(const Csymm::params& params, Csymm::score& score)
{
    return true;
}

event Csymm_naive::execute(const Csymm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t a_buf_size = params.side == ICLBLAS_SIDE_LEFT ? params.m * params.lda : params.n * params.lda;
    size_t b_buf_size = params.n * params.ldb;
    size_t c_buf_size = params.n * params.ldc;

    kernel->set_arg(0, params.side);
    kernel->set_arg(1, params.uplo);
    kernel->set_arg(2, params.m);
    kernel->set_arg(3, params.n);
    kernel->set_arg(4, params.alpha);
    auto buf_a = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(5, buf_a);
    kernel->set_arg(6, params.lda);
    auto buf_b = engine->get_input_buffer(params.B, b_buf_size);
    kernel->set_arg(7, buf_b);
    kernel->set_arg(8, params.ldb);
    kernel->set_arg(9, params.beta);
    auto buf_c = engine->get_inout_buffer(params.C, c_buf_size);
    kernel->set_arg(10, buf_c);
    kernel->set_arg(11, params.ldc);

    auto gws = nd_range(params.m, params.n);
    auto lws = null_range;
    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
