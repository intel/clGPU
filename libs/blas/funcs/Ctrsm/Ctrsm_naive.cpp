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

#include "functions/Ctrsm.hpp"

static const char* module_name = "Ctrsm_naive";
static const char* kernel_name = "Ctrsm_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Ctrsm_naive::accept(const Ctrsm::params& params, Ctrsm::score& score)
{
    return true;
}

event Ctrsm_naive::execute(const Ctrsm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t a_buf_size = params.side == 0 ? params.lda * params.m : params.lda * params.n;
    size_t b_buf_size = params.ldb * params.n;

    kernel->set_arg(0, params.side);
    kernel->set_arg(1, params.uplo);
    kernel->set_arg(2, params.trans);
    kernel->set_arg(3, params.diag);
    kernel->set_arg(4, params.m);
    kernel->set_arg(5, params.n);
    kernel->set_arg(6, params.alpha);
    auto buf_a = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(7, buf_a);
    kernel->set_arg(8, params.lda);
    auto buf_b = engine->get_inout_buffer(params.B, b_buf_size);
    kernel->set_arg(9, buf_b);
    kernel->set_arg(10, params.ldb);

    auto gws = nd_range(1);
    auto lws = null_range;
    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
