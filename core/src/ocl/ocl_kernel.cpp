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
#include "ocl_kernel.hpp"
#include "ocl_buffer.hpp"
#include "ocl_event.hpp"
#include <algorithm>
#include <utility>
#include <cassert>
#include <vector>

namespace iclgpu
{

ocl_kernel::ocl_kernel(const std::shared_ptr<ocl_engine>& engine, const cl::Kernel& handle)
    : kernel_command(engine)
    , _kernel(handle) {}

void ocl_kernel::set_buffer_arg(unsigned idx, const std::shared_ptr<buffer_binding>& binding)
{
    if(!binding->is_defined())
        throw std::invalid_argument("blob is not defined");

    auto buffer = binding->get_buffer(get_engine());
    assert(buffer);

    _buffers[idx] = binding;

    auto oclBuffer = down_pointer_cast<ocl_buffer>(buffer);
    _kernel.setArg(idx, oclBuffer->get_handle());
}

void ocl_kernel::set_options(const kernel_options& params)
{
    auto gws = ocl_range(params.work_size());
    if (gws.dimensions() == 0)
        throw std::invalid_argument("GWS dimensions should be more than 0");

    auto lws = ocl_range(params.parallel_size());
    if (lws.dimensions() != 0 && lws.dimensions() != gws.dimensions())
        throw std::invalid_argument("GWS and LWS dimensions mismatch");

    _gws = gws;
    _lws = lws;
}

std::shared_ptr<event> ocl_kernel::submit(const std::vector<std::shared_ptr<event>>& dependencies,
                                          const command_queue&                       queue)
{
    auto engine     = get_engine<ocl_engine>();
    auto dep_events = make_cl_events(dependencies);
    auto ocl_queue  = engine->toolkit().get_cl_queue(queue);

    std::vector<cl::Event> unmap_events;
    for (auto& pair : _buffers)
    {
        auto binding = pair.second;
        if (binding->get_owning_engine() == get_engine())
        {
            binding->reset_host_ptr();
            auto evt = down_pointer_cast<ocl_buffer>(binding->get_buffer(engine))->enqueue_unmap(ocl_queue, dep_events);
            if (evt())
                unmap_events.push_back(evt);
        }
    }

    const auto krnl_wait_events = unmap_events.empty() ? &dep_events : &unmap_events;

    cl::Event krnl_evt;
    ocl_queue.enqueueNDRangeKernel(_kernel, cl::NullRange, _gws, _lws, krnl_wait_events, &krnl_evt);
    std::vector<cl::Event> buf_events;
    for (auto& pair : _buffers)
    {
        auto binding = pair.second;
        cl::Event evt;
        if (binding->is_output() && binding->get_owning_engine() != engine)
        {
            auto buf = down_pointer_cast<ocl_buffer>(binding->get_buffer(engine));
            evt = buf->read(queue, {krnl_evt}, binding->get_host_ptr());
        }

        if (evt.get() != nullptr && std::count(buf_events.begin(), buf_events.end(), evt) == 0)
            buf_events.push_back(evt);
    }

    cl::Event result_evt;
    switch (buf_events.size())
    {
    case 0:
        result_evt = krnl_evt;
        break;
    case 1:
        result_evt = *buf_events.begin();
        break;
    default:
        ocl_queue.enqueueMarkerWithWaitList(&buf_events, &result_evt);
    }
    return std::make_shared<ocl_event>(shared_from_this(), krnl_evt, result_evt);
}

}
