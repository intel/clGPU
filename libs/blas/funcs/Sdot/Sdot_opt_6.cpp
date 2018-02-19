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

static const char* module_name = "Sdot_opt_6";
static const char* kernel_name = "Sdot_opt_6";
static const char* sum_kernel_name = "sum";
namespace iclgpu { namespace functions { namespace implementations {

bool Sdot_opt_6::accept(const Sdot::params& params, Sdot::score& score)
{
    score.result += 6;
    return true;
}

event Sdot_opt_6::execute(const Sdot::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    auto sum_kernel = engine->get_kernel(sum_kernel_name, module_name);

    int input_buffer_size1 = params.n * params.incx;
    int input_buffer_size2 = params.n * params.incy;
    const int result_buffer_size = 16;

    auto input_buffer1 = engine->get_input_buffer(params.x, input_buffer_size1);
    auto input_buffer2 = engine->get_input_buffer(params.y, input_buffer_size2);
    auto result_buffer = engine->get_temp_buffer<float>(result_buffer_size);
    auto sum_result_buffer = engine->get_output_buffer(params.result, 1);

    kernel->set_arg(0, params.n);
    kernel->set_arg(1, input_buffer1);
    kernel->set_arg(2, params.incx);
    kernel->set_arg(3, input_buffer2);
    kernel->set_arg(4, params.incy);
    kernel->set_arg(5, result_buffer);

    auto gws = nd_range(256 * result_buffer_size);
    auto lws = nd_range(256);

    kernel->set_options({ gws, lws });

    auto kernel1_evt = kernel->submit(dep_events);

    sum_kernel->set_arg(0, result_buffer);
    sum_kernel->set_arg(1, sum_result_buffer);

    auto gws2 = nd_range(16);
    auto lws2 = nd_range(16);

    sum_kernel->set_options({ gws2, lws2 });

    return sum_kernel->submit({ kernel1_evt });
}

} } } // namespace iclgpu::functions::implementations
