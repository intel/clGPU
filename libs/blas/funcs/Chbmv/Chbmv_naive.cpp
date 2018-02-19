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

#include "functions/Chbmv.hpp"

static const char* module_name = "Chbmv_naive";
static const char* kernel_name = "Chbmv_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Chbmv_naive::accept(const Chbmv::params& params, Chbmv::score& score)
{
    return true;
}

event Chbmv_naive::execute(const Chbmv::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    size_t matrix_a_buf_size = (params.lda * params.n);
    size_t vector_x_buf_size = params.n * params.incx;
    size_t vector_y_buf_size = params.n * params.incy;

    kernel->set_arg(0, params.uplo);
    kernel->set_arg(1, params.n);
    kernel->set_arg(2, params.k);
    kernel->set_arg(3, params.alpha);
    auto buf_A = engine->get_input_buffer(params.A, matrix_a_buf_size);
    kernel->set_arg(4, buf_A);
    kernel->set_arg(5, params.lda);
    auto buf_x = engine->get_input_buffer(params.x, vector_x_buf_size);
    kernel->set_arg(6, buf_x);
    kernel->set_arg(7, params.incx);
    kernel->set_arg(8, params.beta);
    auto buf_y = engine->get_inout_buffer(params.y, vector_y_buf_size);
    kernel->set_arg(9, buf_y);
    kernel->set_arg(10, params.incy);

    auto gws = nd_range(params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } }
