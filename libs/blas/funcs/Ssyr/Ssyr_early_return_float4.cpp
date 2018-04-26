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

#include "functions/Ssyr.hpp"

static const char* module_name = "Ssyr_early_return_float4";
static const char* kernel_name = "Ssyr_early_return_float4";

static const int vec_size = 4;

namespace iclgpu { namespace functions { namespace implementations {

bool Ssyr_early_return_float4::accept(const Ssyr::params& params, Ssyr::score& score)
{
    if (params.lda % 4 != 0) return false;
    score.lda = 1.55f;
    return true;
}

event Ssyr_early_return_float4::execute(const Ssyr::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    size_t a_buf_size = params.n * params.lda;

    kernel->set_arg(0, params.uplo);
    kernel->set_arg(1, params.n);
    kernel->set_arg(2, params.alpha);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(3, buf_x);
    kernel->set_arg(4, params.incx);
    auto buf_a = engine->get_inout_buffer(params.A, a_buf_size);
    kernel->set_arg(5, buf_a);
    kernel->set_arg(6, params.lda);

    auto side_m = (params.n + vec_size - 1) / vec_size;
    auto gws = nd_range(side_m, params.n);
    auto lws = null_range;
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
