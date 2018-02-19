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

static const char* module_name = "Sasum_slm_reduction";
static const char* kernel_name = "Sasum_slm_reduction";

namespace iclgpu { namespace functions { namespace implementations {

#define GROUP_SIZE 256
#define TILE_SIZE GROUP_SIZE

bool Sasum_slm_reduction::accept(const Sasum::params& params, Sasum::score& score)
{
    score.n = 1.1f;
    return true;
}

event Sasum_slm_reduction::execute(const Sasum::params& params, const std::vector<event>& dep_events)
{
    int groups = std::max(1, (params.n / GROUP_SIZE));
    int group_size = std::min(params.n, GROUP_SIZE);

    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);

    size_t result_buf_size = 1;
    auto buf_result = engine->get_output_buffer(params.result, result_buf_size);

    size_t subsum_buf_size = groups;
    auto buf_subsum = engine->get_temp_buffer<float>(subsum_buf_size);

    // First iteration
    kernel->set_arg(0, params.n);
    kernel->set_arg(1, buf_x);
    kernel->set_arg(2, params.incx);
    kernel->set_arg(3, groups > 1 ? buf_subsum : buf_result);

    auto gws = nd_range(groups*group_size);
    auto lws = nd_range(group_size);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    auto event = kernel->submit(dep_events);

    while (groups > 1) {
        kernel->set_arg(0, groups);
        kernel->set_arg(1, buf_subsum);
        kernel->set_arg(2, 1);

        group_size = std::min(groups, GROUP_SIZE);
        groups = std::max(1, (groups / GROUP_SIZE));

        kernel->set_arg(3, groups > 1 ? buf_subsum : buf_result);

        gws = nd_range(groups*group_size);
        lws = nd_range(group_size);

        options = kernel_options(gws, lws);
        kernel->set_options(options);

        event = kernel->submit({ event });
    }

    return event;
}

} } } // namespace iclgpu::functions::implementations
