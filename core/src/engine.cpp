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

#include "engine.hpp"

namespace iclgpu
{
direction buffer_binding::set_direction(direction dir)
{
    auto result = _direction;
    _direction  = dir;
    return result;
}

void* buffer_binding::get_host_ptr() const
{
    if (_host_ptr == nullptr)
    {
        auto engine = get_owning_engine();
        if (!engine)
            throw std::logic_error("Blob: neigher host pointer or engine defined");
        auto buffer = get_buffer(engine);
        _host_ptr   = buffer->get_host_ptr();
    }
    return _host_ptr;
}

void buffer_binding::reset_host_ptr() const
{
    if (get_owning_engine())
    {
        _host_ptr = nullptr;
    }
}

void buffer_binding::size(size_t size)
{
    if (_capacity == 0)
        _capacity = _size = size;
    else if (_capacity < size)
        throw std::logic_error("Capacity is less than requested size");
    else
        _size = size;
}

std::shared_ptr<buffer> buffer_binding::get_buffer(const std::shared_ptr<engine>& engine) const
{
    auto it = _buffers.find(engine);
    if (it != _buffers.end())
        return it->second;

    auto buffer = engine->create_buffer(_capacity, get_host_ptr());
    _buffers.insert({engine, buffer});
    return buffer;
}

void commands_sequence::push_back(const std::shared_ptr<command>& command)
{
    if (auto seq = std::dynamic_pointer_cast<commands_sequence>(command))
    {
        for (auto& cmd : seq->_commands)
        {
            push_back(cmd);
        }
    }
    else
    {
        _commands.push_back(command);
    }
}

std::shared_ptr<event> commands_sequence::submit(const std::vector<std::shared_ptr<event>>& dependencies,
                                                        const command_queue&                       queue)
{
    auto       deps = dependencies;
    for (auto& cmd : _commands)
    {
        deps = {cmd->submit(deps, queue)};
    }

    return deps.size() == 1 ? deps[0] : get_engine()->get_raise_event_command()->submit(deps, queue);
}

void commands_parallel::add(const std::shared_ptr<command>& command)
{
    if (auto seq = std::dynamic_pointer_cast<commands_parallel>(command))
    {
        for (auto& cmd : seq->_commands)
        {
            add(cmd);
        }
    }
    else
    {
        _commands.push_back(command);
    }
}

std::shared_ptr<event> commands_parallel::submit(const std::vector<std::shared_ptr<event>>& dependencies,
                                                        const command_queue&                       queue)
{
    std::vector<std::shared_ptr<event>> deps;
    for (auto&                          cmd : _commands)
    {
        deps.push_back(cmd->submit(dependencies, queue));
    }

    return deps.size() == 1 ? deps[0] : get_engine()->get_raise_event_command()->submit(deps, queue);
}

kernel_options::kernel_options(const nd_range& work_size, const nd_range& parallel_size)
    : _work_size(work_size)
  , _parallel_size(parallel_size)
{
    if (_work_size.dimensions() == 0)
        throw std::invalid_argument("work_size");
    if (_parallel_size.dimensions() > 0 && _work_size.dimensions() != _parallel_size.dimensions())
        throw std::invalid_argument("work and parallel sizes dimensions do not match");
}

}
