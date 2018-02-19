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
#include "ocl_event.hpp"

namespace iclgpu
{

ocl_event::ocl_event(const std::shared_ptr<engine_object>& obj, const cl::Event& start_event, const cl::Event& end_event)
    : event(obj->get_engine<ocl_engine>())
    , _pinned(obj)
    , _start_event(start_event)
    , _end_event(end_event) {}

std::chrono::nanoseconds ocl_event::wait()
{
    _end_event.wait();
    cl_ulong start;
    _start_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &start);
    cl_ulong end;
    _start_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    return std::chrono::nanoseconds(static_cast<long long>(end - start));
}

std::vector<cl::Event> make_cl_events(const std::vector<std::shared_ptr<event>>& dependencies)
{
    std::vector<cl::Event> result;
    result.reserve(dependencies.size());

    for (auto& evt : dependencies)
    {
        if (auto clEvt = std::dynamic_pointer_cast<ocl_event>(evt))
        {
            result.push_back(clEvt->get_end_handle());
        }
        else
        {
            // TODO improve performance here (e.g. by parrallelism or cl::UserEvent.set())
            evt->wait();
        }
    }
    return result;
}

}
