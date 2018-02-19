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

#pragma once
#include "ocl/ocl_engine.hpp"
#include "ocl_toolkit.hpp"
#include <vector>

namespace iclgpu
{

class ocl_kernel : public kernel_command
{
public:
    ocl_kernel(const std::shared_ptr<ocl_engine>& engine, const cl::Kernel& handle);

    void set_scalar_arg(unsigned idx, const void* ptr, size_t size) override
    {
        _kernel.setArg(idx, size, ptr);
    }

    void set_buffer_arg(unsigned idx, const std::shared_ptr<buffer_binding>& binding) override;

    void set_options(const kernel_options& params) override;

    std::shared_ptr<event> submit(const std::vector<std::shared_ptr<event>>& dependencies = {},
                                  const command_queue&                       queue        = default_queue) override;

private:
    cl::Kernel  _kernel;
    cl::NDRange _gws;
    cl::NDRange _lws;
    std::map<unsigned, std::shared_ptr<buffer_binding>> _buffers;

    static cl::NDRange ocl_range(const nd_range& v)
    {
        switch (v.dimensions())
        {
        case 0:
            return {};
        case 1:
            return {v[0]};
        case 2:
            return {v[0], v[1]};
        case 3:
            return {v[0], v[1], v[2]};
        default:
            throw std::invalid_argument("too many OCL NDRange dimensions");
        }
    }
};

}
