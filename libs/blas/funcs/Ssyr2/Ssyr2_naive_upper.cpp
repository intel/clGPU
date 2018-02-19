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

static const char* module_name = "Ssyr2_naive_upper";
static const char* kernel_name = "Ssyr2_naive_upper";

namespace iclgpu {
    namespace functions {
        namespace implementations {

            bool Ssyr2_naive_upper::accept(const Ssyr2::params& params, Ssyr2::score& score)
            {
                if (params.uplo == 0)
                {
                    score.uplo = 2.0f;
                    return true;
                }
                else
                {
                    return false;
                }
            }

            event Ssyr2_naive_upper::execute(const Ssyr2::params& params, const std::vector<event>& dep_events)
            {
                auto engine = context()->get_engine();
                auto kernel = engine->get_kernel(kernel_name, module_name);

                auto matrix_a_size = params.n * params.n;
                auto vector_size = params.n;

                //UPLO
                kernel->set_arg(0, params.uplo);

                //N
                kernel->set_arg(1, params.n);

                //Alpha
                kernel->set_arg(2, params.alpha);

                //Vector X
                auto buf_vector_x = engine->get_input_buffer(params.x, vector_size);
                kernel->set_arg(3, buf_vector_x);

                //INCX
                kernel->set_arg(4, params.incx);

                //Vector Y
                auto buf_vector_y = engine->get_input_buffer(params.y, vector_size);
                kernel->set_arg(5, buf_vector_y);

                //INCY
                kernel->set_arg(6, params.incy);

                //Matrix A
                auto buf_matrix_a = engine->get_inout_buffer(params.A, matrix_a_size);
                kernel->set_arg(7, buf_matrix_a);

                //LDA
                kernel->set_arg(8, params.lda);

                //Local temp Buffers:

                //TempMatrix1
                auto TempMatrix1 = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(9, TempMatrix1);
                //TempMatrix2
                auto TempMatrix2 = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(10, TempMatrix2);
                //TempMatrix
                auto TempMatrix = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(11, TempMatrix);
                //TempX
                auto TempX = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(12, TempX);
                //TempY
                auto TempY = engine->get_temp_buffer(matrix_a_size * sizeof(float));
                kernel->set_arg(13, TempY);

                auto gws = nd_range(1);
                auto lws = null_range;

                kernel->set_options({ gws, lws });

                return kernel->submit(dep_events);
            }

        }
    }
} // namespace iclgpu::functions::implementations
