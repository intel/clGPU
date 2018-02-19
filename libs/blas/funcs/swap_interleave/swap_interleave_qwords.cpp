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

#include "functions/swap_interleave.hpp"

static const char* module_name = "swap_interleave_qwords";
static const char* kernel_name = "swap_interleave_qwords";

namespace iclgpu { namespace functions { namespace implementations {

bool swap_interleave_qwords::accept(const swap_interleave::params& params, swap_interleave::score& score)
{
    if (params.elem_size != 8) return false;
    score.elem_size = 1.1f;
    return true;
}

event swap_interleave_qwords::execute(const swap_interleave::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx * params.elem_size;
    size_t y_buf_size = params.n * params.incy * params.elem_size;

    auto buf_x = engine->get_inout_buffer(params.x, x_buf_size);
    kernel->set_arg(0, buf_x);
    kernel->set_arg(1, params.incx);
    auto buf_y = engine->get_inout_buffer(params.y, y_buf_size);
    kernel->set_arg(2, buf_y);
    kernel->set_arg(3, params.incy);

    auto gws = nd_range(params.n);
    auto lws = null_range;
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
