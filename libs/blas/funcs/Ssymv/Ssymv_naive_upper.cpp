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

#include "functions/Ssymv.hpp"

static const char* module_name = "Ssymv_naive_upper";
static const char* kernel_name = "Ssymv_naive_upper";

namespace iclgpu {
    namespace functions {
        namespace implementations {

            bool Ssymv_naive_upper::accept(const Ssymv::params& params, Ssymv::score& score)
            {
                if (params.uplo == 0)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }

            event Ssymv_naive_upper::execute(const Ssymv::params& params, const std::vector<event>& dep_events)
            {
                auto engine = context()->get_engine();
                auto kernel = engine->get_kernel(kernel_name, module_name);
                size_t buf_matrix_size = params.n * params.lda;
                size_t buf_vector_x_size = params.n * params.incx;
                size_t buf_vector_y_size = params.n * params.incy;

                kernel->set_arg(0, params.n);
                kernel->set_arg(1, params.alpha);
                auto buf_A = engine->get_input_buffer(params.A, buf_matrix_size);
                kernel->set_arg(2, buf_A);
                kernel->set_arg(3, params.lda);
                auto buf_x = engine->get_input_buffer(params.x, buf_vector_x_size);
                kernel->set_arg(4, buf_x);
                kernel->set_arg(5, params.incx);
                kernel->set_arg(6, params.beta);
                auto buf_y = engine->get_inout_buffer(params.y, buf_vector_y_size);
                kernel->set_arg(7, buf_y);
                kernel->set_arg(8, params.incy);

                auto gws = nd_range(1);
                auto lws = null_range;

                kernel->set_options({ gws, lws });

                return kernel->submit(dep_events);
            }

        }
    }
} // namespace iclgpu::functions::implementations
