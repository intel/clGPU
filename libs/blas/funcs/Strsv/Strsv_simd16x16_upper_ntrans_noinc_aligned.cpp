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

#include "functions/Strsv.hpp"

#include "../implementation_helpers.hpp"

static const char* module_name = "Strsv_simd16x16_upper_ntrans_noinc_aligned";
static const char* kernel_name = "Strsv_simd16x16_upper_ntrans";

#define ICLBLAS_FILL_MODE_UPPER (0)
#define ICLBLAS_OP_N (0)

static const int lwg_size = 256;

namespace iclgpu { namespace functions { namespace implementations {

bool Strsv_simd16x16_upper_ntrans_noinc_aligned::accept(const Strsv::params& params, Strsv::score& score)
{
    if (params.uplo != ICLBLAS_FILL_MODE_UPPER || params.trans != ICLBLAS_OP_N || params.incx != 1 || !is_aligned<16, float>(params.x))
        return false;
    score.uplo = 1.1f;
    score.trans = 1.1f;
    score.incx = 1.2f;
    return true;
}

event Strsv_simd16x16_upper_ntrans_noinc_aligned::execute(const Strsv::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t A_buf_size = params.n * params.lda;
    size_t x_buf_size = params.n * params.incx;

    kernel->set_arg(0, params.diag);
    kernel->set_arg(1, params.n);
    auto buf_A = engine->get_input_buffer(params.A, A_buf_size);
    kernel->set_arg(2, buf_A);
    kernel->set_arg(3, params.lda);
    auto buf_x = engine->get_inout_buffer(params.x, x_buf_size);
    kernel->set_arg(4, buf_x);
    kernel->set_arg(5, params.incx);

    auto gws = nd_range(lwg_size);
    auto lws = nd_range(lwg_size);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
