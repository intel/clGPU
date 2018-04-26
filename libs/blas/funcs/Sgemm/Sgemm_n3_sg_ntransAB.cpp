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

#include "functions/Sgemm.hpp"

#include "iclblas_common.h.cl"


#define TARG_SG_SIZE   16

#define TARG_TILE_M    16
#define TARG_TILE_N     8
#define TARG_TILE_AK   16
#define TARG_TILE_BK   16

constexpr static const int opt_threshold = 4;


static const char* module_name = "Sgemm_n3_sg_ntransAB";
static const char* kernel_name = "Sgemm_n3_sg_ntransAB";

namespace iclgpu { namespace functions { namespace implementations {

bool Sgemm_n3_sg_ntransAB::accept(const Sgemm::params& params, Sgemm::score& score)
{
    if (params.transa == ICLBLAS_OP_N && params.transb == ICLBLAS_OP_N)
    {
        score.transa = 1.10f;
        score.transb = 1.10f;

        if (params.m >= opt_threshold * TARG_TILE_M && params.k >= opt_threshold * TARG_TILE_AK)
            score.transa = 1.50f;
        if (params.n >= opt_threshold * TARG_TILE_N && params.k >= opt_threshold * TARG_TILE_BK)
            score.transb = 1.50f;

        return true;
    }

    return false;
}

event Sgemm_n3_sg_ntransAB::execute(const Sgemm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    
    const size_t a_buf_size = params.k * params.lda;
    const size_t b_buf_size = params.n * params.ldb;
    const size_t c_buf_size = params.n * params.ldc;

    auto buf_A = engine->get_input_buffer(params.A, a_buf_size);
    auto buf_B = engine->get_input_buffer(params.B, b_buf_size);
    auto buf_C = engine->get_inout_buffer(params.C, c_buf_size);

    kernel->set_arg( 0, params.m);
    kernel->set_arg( 1, params.n);
    kernel->set_arg( 2, params.k);
    kernel->set_arg( 3, params.alpha);
    kernel->set_arg( 4, buf_A);
    kernel->set_arg( 5, params.lda);
    kernel->set_arg( 6, buf_B);
    kernel->set_arg( 7, params.ldb);
    kernel->set_arg( 8, params.beta);
    kernel->set_arg( 9, buf_C);
    kernel->set_arg(10, params.ldc);

    const size_t tile_cnt_m = (static_cast<size_t>(params.m) + TARG_TILE_M - 1) / TARG_TILE_M;
    const size_t tile_cnt_n = (static_cast<size_t>(params.n) + TARG_TILE_N - 1) / TARG_TILE_N;

    auto gws = nd_range(TARG_SG_SIZE * tile_cnt_m, tile_cnt_n);
    auto lws = nd_range(TARG_SG_SIZE, 1);
    kernel->set_options(kernel_options(gws, lws));

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
