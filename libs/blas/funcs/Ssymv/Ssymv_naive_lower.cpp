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

static const char* module_name = "Ssymv_naive_lower";
static const char* kernel_name = "Ssymv_naive_lower";

namespace iclgpu {
    namespace functions {
        namespace implementations {

            bool Ssymv_naive_lower::accept(const Ssymv::params& params, Ssymv::score& score)
            {
                if (params.uplo == 1)
                {
                    score.uplo = 2.0f;
                    return true;
                }
                else
                {
                    return true;
                }
            }

            event Ssymv_naive_lower::execute(const Ssymv::params& params, const std::vector<event>& dep_events)
            {
                auto engine = context()->get_engine();
                auto kernel = engine->get_kernel(kernel_name, module_name);

                auto matrix_a_size = params.n * params.n;
                auto vector_size = params.n;

                //N
                kernel->set_arg(0, params.n);

                //Alpha
                kernel->set_arg(1, params.alpha);

                //Matrix A
                auto buf_matrix_a = engine->get_input_buffer(params.A, matrix_a_size);
                kernel->set_arg(2, buf_matrix_a);

                //lda
                kernel->set_arg(3, params.lda);

                //Vector X
                auto buf_vector_x = engine->get_input_buffer(params.x, vector_size);
                kernel->set_arg(4, buf_vector_x);

                //Incx
                kernel->set_arg(5, params.incx);

                //Beta
                kernel->set_arg(6, params.beta);

                //Vector Y
                auto buf_vector_y = engine->get_inout_buffer(params.y, vector_size);
                kernel->set_arg(7, buf_vector_y);

                //Incy
                kernel->set_arg(8, params.incy);

                //Temp Global Buffer
                auto buf_tempL = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(9, buf_tempL);
                auto buf_tempR = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(10, buf_tempR);

                auto gws = nd_range(1);
                auto lws = null_range;

                kernel->set_options({ gws, lws });

                return kernel->submit(dep_events);
            }

        }
    }
} // namespace iclgpu::functions::implementations
