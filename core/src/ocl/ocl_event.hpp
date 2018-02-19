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
#include "engine.hpp"
#include "ocl_toolkit.hpp"
#include <vector>

namespace iclgpu
{

class ocl_event : public event
{
public:
    ocl_event(const std::shared_ptr<engine_object>& obj, const cl::Event& start_event, const cl::Event& end_event);

    std::chrono::nanoseconds wait() override;

    const cl::Event& get_start_handle() const { return _start_event; }
    const cl::Event& get_end_handle() const { return _end_event; }

private:
    std::shared_ptr<engine_object> _pinned;
    cl::Event                      _start_event;
    cl::Event                      _end_event;
};

std::vector<cl::Event> make_cl_events(const std::vector<std::shared_ptr<event>>& dependencies);

}
