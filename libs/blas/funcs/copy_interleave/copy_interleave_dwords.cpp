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

#include "functions/copy_interleave.hpp"

static const char* module_name = "copy_interleave_dwords";
static const char* kernel_name = "copy_interleave_dwords";

namespace iclgpu { namespace functions { namespace implementations {

bool copy_interleave_dwords::accept(const copy_interleave::params& params, copy_interleave::score& score)
{
    if (params.elem_size == sizeof(uint32_t))
    {
        score.elem_size = 1.5;
        return true;
    }
    return false;
}

event copy_interleave_dwords::execute(const copy_interleave::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    auto src_buf_size = params.num * params.elem_size * params.src_pad;
    auto dst_buf_size = params.num * params.elem_size * params.dst_pad;

    auto buf_src = engine->get_input_buffer(params.src, src_buf_size);
    kernel->set_arg(0, buf_src);
    auto buf_dst = engine->get_inout_buffer(params.dst, dst_buf_size);
    kernel->set_arg(1, buf_dst);
    kernel->set_arg(2, params.src_pad);
    kernel->set_arg(3, params.dst_pad);


    auto gws = nd_range(params.num);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
