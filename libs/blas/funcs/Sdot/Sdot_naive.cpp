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

#include "functions/Sdot.hpp"

static const char* module_name = "Sdot_naive";
static const char* kernel_name = "Sdot_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Sdot_naive::accept(const Sdot::params& params, Sdot::score& score)
{

    return true;
}

event Sdot_naive::execute(const Sdot::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    int input_buffer_size1 = params.n * params.incx;
    int input_buffer_size2 = params.n * params.incy;
    int result_buffer_size = 1;

    auto input_buffer1 = engine->get_input_buffer(params.x, input_buffer_size1);
    auto input_buffer2 = engine->get_input_buffer(params.y, input_buffer_size2);
    auto result_buffer = engine->get_output_buffer(params.result, result_buffer_size);

    kernel->set_arg(0, params.n);
    kernel->set_arg(1, input_buffer1);
    kernel->set_arg(2, params.incx);
    kernel->set_arg(3, input_buffer2);
    kernel->set_arg(4, params.incy);
    kernel->set_arg(5, result_buffer);

    auto gws = nd_range(1);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);

}

} } } // namespace iclgpu::functions::implementations
