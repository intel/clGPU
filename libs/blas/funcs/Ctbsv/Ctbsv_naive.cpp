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

#include "functions/Ctbsv.hpp"

static const char* module_name = "Ctbsv_naive";
static const char* kernel_name = "Ctbsv_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Ctbsv_naive::accept(const Ctbsv::params& params, Ctbsv::score& score)
{
    return true;
}

event Ctbsv_naive::execute(const Ctbsv::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    auto a_buf_size = params.n * params.lda;
    auto x_buf_size = params.n * params.incx;

    kernel->set_arg(0, params.uplo);
    kernel->set_arg(1, params.trans);
    kernel->set_arg(2, params.diag);
    kernel->set_arg(3, params.n);
    kernel->set_arg(4, params.k);
    auto buf_a = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(5, buf_a);
    kernel->set_arg(6, params.lda);
    auto buf_x = engine->get_inout_buffer(params.x, x_buf_size);
    kernel->set_arg(7, buf_x);
    kernel->set_arg(8, params.incx);

    auto gws = nd_range(1);
    auto lws = null_range;
    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
