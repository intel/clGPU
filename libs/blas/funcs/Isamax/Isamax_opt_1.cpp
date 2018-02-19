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

#include "functions/Isamax.hpp"

static const char* module_name = "Isamax_opt_1";
static const char* kernel_name = "Isamax_opt_1";

#define GLOBAL_THREADS 128
#define LOCAL_SIZE 128

namespace iclgpu { namespace functions { namespace implementations {

bool Isamax_opt_1::accept(const Isamax::params& params, Isamax::score& score)
{
    if (params.n >= 128 && params.incx == 1)
    {
        score.n = 1.1f;
        return true;
    }
    else
    {
        return false;
    }
}

event Isamax_opt_1::execute(const Isamax::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    auto buf_size = params.n;

    kernel->set_arg(0, params.n);

    auto buf_x = engine->get_input_buffer(params.x, buf_size);
    kernel->set_arg(1, buf_x);

    kernel->set_arg(2, params.incx);

    auto buf_res = engine->get_output_buffer(params.result, 1);
    kernel->set_arg(3, buf_res);


    auto gws = nd_range(GLOBAL_THREADS);
    auto lws = nd_range(LOCAL_SIZE);

    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } }
