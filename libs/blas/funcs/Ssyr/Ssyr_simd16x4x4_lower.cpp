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

#include "functions/Ssyr.hpp"

static const char* module_name = "Ssyr_simd16x4x4_lower";
static const char* kernel_name = "Ssyr_simd16x4x4_lower";

static const int simd = 16;
static const int vec_size = 4;
static const int tile = vec_size * simd;

#define ICLBLAS_FILL_MODE_LOWER (1)

namespace iclgpu { namespace functions { namespace implementations {

bool Ssyr_simd16x4x4_lower::accept(const Ssyr::params& params, Ssyr::score& score)
{
    if (params.uplo != ICLBLAS_FILL_MODE_LOWER || params.lda % vec_size != 0) return false;
    score.uplo = 1.1f;
    score.lda = 1.4f;
    return true;
}

event Ssyr_simd16x4x4_lower::execute(const Ssyr::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    size_t a_buf_size = params.n * params.lda;

    kernel->set_arg(0, params.n);
    kernel->set_arg(1, params.alpha);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(2, buf_x);
    kernel->set_arg(3, params.incx);
    auto buf_a = engine->get_inout_buffer(params.A, a_buf_size);
    kernel->set_arg(4, buf_a);
    kernel->set_arg(5, params.lda);

    int tiles_one_side = (params.n + tile - 1) / tile;
    int tiles_total = (tiles_one_side + 1) * tiles_one_side / 2;

    auto gws = nd_range(tiles_total * simd);
    auto lws = nd_range(simd);
    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
