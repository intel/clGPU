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

#include "functions/Ctpmv.hpp"

static const char* module_name = "Ctpmv_naive";
static const char* kernel_name = "Ctpmv_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Ctpmv_naive::accept(const Ctpmv::params& params, Ctpmv::score& score)
{
    return true;
}

event Ctpmv_naive::execute(const Ctpmv::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    auto src_buf_size = (params.n * (params.n + 1)) / 2;
    auto dst_buf_size = params.n * params.incx;

    kernel->set_arg(0, params.uplo);
    kernel->set_arg(1, params.trans);
    kernel->set_arg(2, params.diag);
    kernel->set_arg(3, params.n);
    auto buf_AP = engine->get_input_buffer(params.AP, src_buf_size);
    kernel->set_arg(4, buf_AP);
    auto buf_x = engine->get_inout_buffer(params.x, dst_buf_size);
    kernel->set_arg(5, buf_x);
    kernel->set_arg(6, params.incx);

    auto buf_parties = engine->get_temp_buffer(params.n * sizeof(iclgpu::complex_t));
    kernel->set_arg(7, buf_parties);

    auto gws = nd_range(1);
    auto lws = null_range;
    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
