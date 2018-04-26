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

#include "functions/Sger.hpp"

static const char* module_name = "Sger_simd16x1x1";
static const char* kernel_name = "Sger_simd16x1x1";

static const int simd = 16;

namespace iclgpu { namespace functions { namespace implementations {

bool Sger_simd16x1x1::accept(const Sger::params& params, Sger::score& score)
{
    score.n = 1.05f;
    return true;
}

event Sger_simd16x1x1::execute(const Sger::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    size_t x_buf_size = params.m * params.incx;
    size_t y_buf_size = params.n * params.incy;
    size_t a_buf_size = params.n * params.lda;

    kernel->set_arg(0, params.m);
    kernel->set_arg(1, params.n);
    kernel->set_arg(2, params.alpha);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(3, buf_x);
    kernel->set_arg(4, params.incx);
    auto buf_y = engine->get_input_buffer(params.y, y_buf_size);
    kernel->set_arg(5, buf_y);
    kernel->set_arg(6, params.incy);
    auto buf_a = engine->get_inout_buffer(params.A, a_buf_size);
    kernel->set_arg(7, buf_a);
    kernel->set_arg(8, params.lda);

    auto side_m = (params.m + simd - 1) / simd;
    auto side_n = (params.n + simd - 1) / simd;

    auto gws = nd_range(side_m * simd, side_n);
    auto lws = nd_range(simd, 1);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
