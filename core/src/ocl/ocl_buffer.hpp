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

class ocl_buffer : public buffer
{
public:

    ocl_buffer(const std::shared_ptr<ocl_engine>& engine, size_t size, void* ptr = nullptr);

    size_t size() const override { return _size; }
    void* get_host_ptr() override;

    cl::Event read(const command_queue& queue, const std::vector<cl::Event>& dependencies, void* ptr) const;
    cl::Event enqueue_unmap(const cl::CommandQueue& ocl_queue, const std::vector<cl::Event>& dependencies);

    const cl::Buffer& get_handle() const { return _buffer; }

private:
    size_t       _size;
    void*        _mapped_ptr;
    cl_mem_flags _cl_mem_flags;
    cl::Buffer   _buffer;

    bool use_host_pointer() const { return (_cl_mem_flags & CL_MEM_USE_HOST_PTR) != 0; }
};

}
