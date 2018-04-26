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
#include "functions/Sgbmv.hpp"

static const char* module_name = "Sgbmv_simd_one_row_ntrans";
static const char* kernel_name = "Sgbmv_simd_one_row_ntrans";

namespace iclgpu { namespace functions { namespace implementations {

#define ICLBLAS_OP_N (0)

static const int simd = 8;

bool Sgbmv_simd_one_row_ntrans::accept(const Sgbmv::params& params, Sgbmv::score& score)
{
    if (params.trans != ICLBLAS_OP_N) return false;
    score.trans = 1.1f;
    return true;
}

event Sgbmv_simd_one_row_ntrans::execute(const Sgbmv::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t A_buf_size = params.n * params.lda;
    size_t x_buf_size = params.n * params.incx;
    size_t y_buf_size = params.m * params.incy;

    kernel->set_arg(0, params.m);
    kernel->set_arg(1, params.n);
    kernel->set_arg(2, params.kl);
    kernel->set_arg(3, params.ku);
    kernel->set_arg(4, params.alpha);
    auto buf_A = engine->get_input_buffer(params.A, A_buf_size);
    kernel->set_arg(5, buf_A);
    kernel->set_arg(6, params.lda);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(7, buf_x);
    kernel->set_arg(8, params.incx);
    kernel->set_arg(9, params.beta);
    auto buf_y = engine->get_inout_buffer(params.y, y_buf_size);
    kernel->set_arg(10, buf_y);
    kernel->set_arg(11, params.incy);

    auto gws = nd_range(params.m * simd);
    auto lws = nd_range(simd);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
