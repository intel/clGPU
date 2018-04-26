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

#include "functions/Ssyr2.hpp"

static const char* module_name = "Ssyr2_opt_async_lower";
static const char* kernel_name = "Ssyr2_opt_async_lower";

#define ICLBLAS_FILL_MODE_UPPER (0)
#define ICLBLAS_FILL_MODE_LOWER (1)

namespace iclgpu { namespace functions { namespace implementations {

bool Ssyr2_opt_async_lower::accept(const Ssyr2::params& params, Ssyr2::score& score)
{
    if (params.uplo == ICLBLAS_FILL_MODE_LOWER)
    {
        score.uplo = 1.1f;
        score.n = 1.2f;
        return true;
    }
    else
    {
        return false;
    }
}

event Ssyr2_opt_async_lower::execute(const Ssyr2::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    size_t y_buf_size = params.n * params.incy;
    size_t A_buf_size = params.n * params.lda;

    kernel->set_arg(0, params.n);
    kernel->set_arg(1, params.alpha);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(2, buf_x);
    kernel->set_arg(3, params.incx);
    auto buf_y = engine->get_input_buffer(params.y, y_buf_size);
    kernel->set_arg(4, buf_y);
    kernel->set_arg(5, params.incy);
    auto buf_A = engine->get_inout_buffer(params.A, A_buf_size);
    kernel->set_arg(6, buf_A);
    kernel->set_arg(7, params.lda);

    auto gws = nd_range(params.n, params.n);
    auto lws = null_range;
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
