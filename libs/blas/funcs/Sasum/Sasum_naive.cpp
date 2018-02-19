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

#include "functions/Sasum.hpp"

static const char* module_name = "Sasum_naive";
static const char* kernel_name = "Sasum_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Sasum_naive::accept(const Sasum::params& params, Sasum::score& score)
{
    score.n = 1.05f;
    return true;
}

event Sasum_naive::execute(const Sasum::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    int input_buffer_size = params.n * params.incx;
    int result_buffer_size = 1;

    auto input_buffer = engine->get_input_buffer(params.x, input_buffer_size);
    auto result_buffer = engine->get_output_buffer(params.result, result_buffer_size);

    kernel->set_arg(0, params.n);
    kernel->set_arg(1, input_buffer);
    kernel->set_arg(2, params.incx);
    kernel->set_arg(3, result_buffer);

    auto gws = nd_range(params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
