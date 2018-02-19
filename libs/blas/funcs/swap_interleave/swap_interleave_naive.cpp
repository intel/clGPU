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

static const char* module_name = "swap_interleave_naive";
static const char* kernel_name = "swap_interleave_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool swap_interleave_naive::accept(const swap_interleave::params& params, swap_interleave::score& score)
{
    return true;
}

event swap_interleave_naive::execute(const swap_interleave::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    auto byte_incx = params.incx * params.elem_size;
    auto byte_incy = params.incy * params.elem_size;
    auto x_buf_size = params.n * byte_incx;
    auto y_buf_size = params.n * byte_incy;

    auto buf_x = engine->get_inout_buffer(params.x, x_buf_size);
    kernel->set_arg(0, buf_x);
    kernel->set_arg(1, byte_incx);
    auto buf_y = engine->get_inout_buffer(params.y, y_buf_size);
    kernel->set_arg(2, buf_y);
    kernel->set_arg(3, byte_incy);
    kernel->set_arg(4, params.elem_size);

    auto gws = nd_range(params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
