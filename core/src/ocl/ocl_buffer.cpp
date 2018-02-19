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

#include "ocl_buffer.hpp"

// cache size
#define ZERO_COPY_ALIGN 0x40
// page size
#define RECOMMENDED_ALIGN 0x1000
// 1M
#define GPU_COPY_THRESHOLD 0x100000

namespace iclgpu
{

cl_mem_flags make_buffer_flags(size_t size, void* ptr)
{
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (ptr == nullptr)
        return flags;

    if ((reinterpret_cast<uintptr_t>(ptr) % ZERO_COPY_ALIGN == 0) && (size % ZERO_COPY_ALIGN == 0))
    {
        flags |= CL_MEM_USE_HOST_PTR;
    }
    else
    {
        flags |= CL_MEM_COPY_HOST_PTR;
    }
    return flags;
}

ocl_buffer::ocl_buffer(const std::shared_ptr<ocl_engine>& engine, size_t size, void* ptr)
    : buffer(engine)
    , _size(size)
    , _mapped_ptr(nullptr)
    , _cl_mem_flags(make_buffer_flags(size, ptr))
    , _buffer(engine->toolkit().get_cl_context(), _cl_mem_flags, size, ptr)
{
    //zero init
    //TODO redesign this area to avoid enqueue
    if (ptr == nullptr)
    {
        const uint8_t zero = 0;
        engine->toolkit().get_cl_queue().enqueueFillBuffer(_buffer, zero, 0, size);
    }
}

cl::Event ocl_buffer::read(const command_queue& queue, const std::vector<cl::Event>& dependencies, void* ptr) const
{
    auto engine = get_engine<ocl_engine>();
    if (!use_host_pointer() && ptr != nullptr)
    {
        cl::Event result;
        engine->toolkit().get_cl_queue(queue).enqueueReadBuffer(get_handle(), false, 0, _size, ptr, &dependencies, &result);
        result.wait();
        return result;
    }

    switch (dependencies.size())
    {
    case 0:
        {
            cl::UserEvent result(engine->toolkit().get_cl_context());
            result.setStatus(CL_COMPLETE);
            return result;
        }
    case 1:
        return dependencies[0];
    default:
        {
            cl::Event result;
            engine->toolkit().get_cl_queue(queue).enqueueMarkerWithWaitList(&dependencies, &result);
            return result;
        }
    }
}

void* ocl_buffer::get_host_ptr()
{
    if (_mapped_ptr != nullptr)
    {
        _mapped_ptr = get_engine<ocl_engine>()
                        ->toolkit()
                        .get_cl_queue()
                        .enqueueMapBuffer(_buffer, true, CL_MAP_READ | CL_MAP_WRITE, 0, _size);
    }
    return _mapped_ptr;
}

cl::Event ocl_buffer::enqueue_unmap(const cl::CommandQueue& ocl_queue, const std::vector<cl::Event>& dependencies)
{
    cl::Event result;
    if (_mapped_ptr != nullptr)
    {
        ocl_queue.enqueueUnmapMemObject(_buffer, _mapped_ptr, &dependencies, &result);
        _mapped_ptr = nullptr;
    }
    return result;
}

}
