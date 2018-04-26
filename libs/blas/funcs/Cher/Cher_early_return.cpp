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

#include "functions/Cher.hpp"

static const char* module_name = "Cher_early_return";
static const char* kernel_name = "Cher_early_return";

namespace iclgpu { namespace functions { namespace implementations {

bool Cher_early_return::accept(const Cher::params& params, Cher::score& score)
{
    return true;
}

event Cher_early_return::execute(const Cher::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    size_t a_buf_size = params.n * params.lda;

    kernel->set_arg(0, params.uplo);
    kernel->set_arg(1, params.alpha);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(2, buf_x);
    kernel->set_arg(3, params.incx);
    auto buf_a = engine->get_inout_buffer(params.A, a_buf_size);
    kernel->set_arg(4, buf_a);
    kernel->set_arg(5, params.lda);

    auto gws = nd_range(params.n, params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
