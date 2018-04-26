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
#include "functions/Srotm.hpp"

static const char* module_name = "Srotm_noinc_full";
static const char* kernel_name = "Srotm_block_read";

static const int wi_elems = 4;
static const int simd = 16;
static const int vec_size = 1;

static const int tile = wi_elems * simd * vec_size;

namespace iclgpu { namespace functions { namespace implementations {

bool Srotm_noinc_full::accept(const Srotm::params& params, Srotm::score& score)
{
    if (params.param[0] != -1.f || params.n < tile || params.incx != 1 || params.incy != 1) return false;
    score.param = 1.1f;
    score.n = 1.1f;
    score.incx = 1.1f;
    score.incx = 1.1f;
    return true;
}

event Srotm_noinc_full::execute(const Srotm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    auto x_buf_size = params.n * params.incx;
    auto y_buf_size = params.n * params.incy;
    auto param_size = 5;

    kernel->set_arg(0, params.n);
    auto buf_x = engine->get_inout_buffer(params.x, x_buf_size);
    kernel->set_arg(1, buf_x);
    auto buf_y = engine->get_inout_buffer(params.y, y_buf_size);
    kernel->set_arg(2, buf_y);
    auto buf_param = engine->get_input_buffer(params.param, param_size);
    kernel->set_arg(3, buf_param);

    int work_groups = params.n / tile;
    auto gws = nd_range(work_groups * simd);
    auto lws = nd_range(simd);

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
