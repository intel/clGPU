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

#include "functions/Cgbmv.hpp"

static const char* module_name = "Cgbmv_trans";
static const char* kernel_name = "Cgbmv_trans";

namespace iclgpu { namespace functions { namespace implementations {

#define ICLBLAS_OP_T 1

bool Cgbmv_trans::accept(const Cgbmv::params& params, Cgbmv::score& score)
{
    if (params.trans != ICLBLAS_OP_T)
        return false;
    return true;
}

event Cgbmv_trans::execute(const Cgbmv::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t a_buf_size = params.n*params.lda;
    size_t x_buf_size = params.m*params.incx;
    size_t y_buf_size = params.n*params.incy;

    kernel->set_arg(0, params.m);
    kernel->set_arg(1, params.kl);
    kernel->set_arg(2, params.ku);
    kernel->set_arg(3, params.alpha);
    auto buf_a = engine->get_input_buffer(params.A, a_buf_size);
    kernel->set_arg(4, buf_a);
    kernel->set_arg(5, params.lda);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(6, buf_x);
    kernel->set_arg(7, params.incx);
    kernel->set_arg(8, params.beta);
    auto buf_y = engine->get_inout_buffer(params.y, y_buf_size);
    kernel->set_arg(9, buf_y);
    kernel->set_arg(10, params.incy);


    auto gws = nd_range(params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
