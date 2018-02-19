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

#include "ocl/ocl_engine.hpp"
#include "ocl_toolkit.hpp"
#include "primitive_db.hpp"
#include <utility>
#include <cassert>

namespace iclgpu
{

ocl_toolkit::ocl_toolkit(ocl_engine* engine)
    : _engine(engine)
    , _device(get_gpu_device())
    , _ocl_context(_device)
    , _queues{cl::CommandQueue{_ocl_context, _device, CL_QUEUE_PROFILING_ENABLE}}
{
    assert(_engine);
}

const cl::Context& ocl_toolkit::get_cl_context() const
{
    assert(_ocl_context());
    return _ocl_context;
}

cl::CommandQueue& ocl_toolkit::get_cl_queue(const command_queue& queue)
{
    assert(!_queues.empty() && _queues[0]());
    if (queue.id() >= _queues.size())
        throw std::invalid_argument("Queue id " + std::to_string(queue.id()) + " does not exist");
    return _queues[queue.id()];
}

cl::Device ocl_toolkit::get_gpu_device()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device default_device;
    for (auto& p : platforms)
    {
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices)
        {
            if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
            {
                auto vendor_id = d.getInfo<CL_DEVICE_VENDOR_ID>();
                //set Intel GPU device default
                if (vendor_id == 0x8086)
                {
                    return d;
                }
            }
        }
    }
    throw std::runtime_error("No OpenCL GPU device found.");
}

const cl::Program& ocl_toolkit::get_module(const std::string& module_name)
{
    auto it = _programs.find(module_name);
    if (it != _programs.end())
        return it->second;

    auto db = _engine->context()->get<primitive_db>();
    cl::Program::Sources codes = {db->get("complex.h")};
    codes.push_back(db->get(module_name));
    cl::Program program(_ocl_context, codes);
    program.build({_device});
    auto inserted = _programs.emplace(module_name, program);
    assert(inserted.second);
    return inserted.first->second;
}

}
