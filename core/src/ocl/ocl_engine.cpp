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
#include "ocl_buffer.hpp"
#include "ocl_event.hpp"
#include "ocl_kernel.hpp"
#include <memory>
#include <cassert>

namespace iclgpu
{

DEFINE_CLASS_ID(ocl_engine)

ocl_engine::ocl_engine(const std::shared_ptr<iclgpu::context>& ctx)
    : element(ctx)
    , _ocl_toolkit(new ocl_toolkit(this)) {}

ocl_engine::~ocl_engine() = default;

std::shared_ptr<kernel_command> ocl_engine::get_kernel(const std::string& name, const std::string& module)
{
    auto module_name = module.empty() ? name : module;
    return std::make_shared<ocl_kernel>(shared_from_this(), cl::Kernel(toolkit().get_module(module_name), name.c_str()));
}

std::shared_ptr<buffer> ocl_engine::create_buffer(size_t size, void* ptr)
{
    if (size == 0) throw std::invalid_argument("size should not be zero.");
    return std::make_shared<ocl_buffer>(shared_from_this(), size, ptr);
}

namespace
{
struct ocl_raise_event_command : raise_event_command
{
    using raise_event_command::raise_event_command;
    std::shared_ptr<event> submit(const std::vector<std::shared_ptr<event>>& dependencies = {},
                                  const command_queue&                       queue        = default_queue) override
    {
        auto engine    = get_engine<ocl_engine>();
        auto cl_events = make_cl_events(dependencies);
        auto ocl_queue = engine->toolkit().get_cl_queue(queue);

        cl::Event evt;
        ocl_queue.enqueueMarkerWithWaitList(&cl_events, &evt);
        return std::make_shared<ocl_event>(shared_from_this(), evt, evt);
    }
};
}

std::shared_ptr<raise_event_command> ocl_engine::get_raise_event_command()
{
    return std::make_shared<ocl_raise_event_command>(shared_from_this());
}

namespace
{
class ocl_commands_sequence : public commands_sequence
{
public:
    using commands_sequence::commands_sequence;

    std::shared_ptr<event> submit(const std::vector<std::shared_ptr<event>>&        dependencies,
                                  const command_queue&                              queue) override
    {
        if (_commands.empty())
        {
            return get_engine()->get_raise_event_command()->submit(dependencies, queue);
        }

        cl::Event start_evt;
        cl::Event end_evt;

        auto deps = dependencies;

        for (auto& cmd : _commands)
        {
            const auto     evt     = cmd->submit(deps, queue);
            if (const auto ocl_evt = std::dynamic_pointer_cast<ocl_event>(evt))
            {
                start_evt = start_evt.get() ? start_evt : ocl_evt->get_start_handle();
                end_evt   = ocl_evt->get_end_handle();
            }
            deps = {evt};
        }
        return std::make_shared<ocl_event>(shared_from_this(), start_evt, end_evt);
    }

};
}

std::shared_ptr<commands_sequence>
ocl_engine::get_commands_sequence(const std::vector<std::shared_ptr<command>>& commands)
{
    auto             result = std::make_shared<ocl_commands_sequence>(shared_from_this());
    for (const auto& cmd : commands)
    {
        result->push_back(cmd);
    }
    return result;
}

std::shared_ptr<commands_parallel>
ocl_engine::get_commands_parallel(const std::vector<std::shared_ptr<command>>& commands)
{
    auto result = std::make_shared<commands_parallel>(shared_from_this());
    for (const auto& cmd : commands)
    {
        result->add(cmd);
    }
    return result;
}

}
