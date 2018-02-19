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

#include "functions/Scnrm2.hpp"

static const char* module_name = "Scnrm2_simd16_single_thread";
static const char* kernel_name = "Scnrm2_simd16_single_thread";

namespace iclgpu { namespace functions { namespace implementations {

#define SIMD 16

bool Scnrm2_simd16_single_thread::accept(const Scnrm2::params& params, Scnrm2::score& score)
{
    if (params.n >= SIMD) score.n = 1.1f;
    return true;
}

event Scnrm2_simd16_single_thread::execute(const Scnrm2::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    size_t result_buf_size = 1;

    kernel->set_arg(0, params.n);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(1, buf_x);
    kernel->set_arg(2, params.incx);
    auto buf_res = engine->get_output_buffer(params.result, result_buf_size);
    kernel->set_arg(3, buf_res);

    auto gws = nd_range(SIMD);
    auto lws = nd_range(SIMD);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);
    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
