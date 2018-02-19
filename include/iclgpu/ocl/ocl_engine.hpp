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
#include "context.hpp"
#include "engine.hpp"
#include <memory>

/// @file ocl_engine.hpp
/// OpenCL execution engine implementation

namespace iclgpu
{
/// @addtogroup engine Execution engines
/// @{

class ocl_toolkit;

class ocl_engine : public engine, public context::element<ocl_engine>, public std::enable_shared_from_this<ocl_engine>
{
public:
    explicit ocl_engine(const std::shared_ptr<iclgpu::context>& ctx);

    ~ocl_engine() override;
    std::shared_ptr<kernel_command>      get_kernel(const std::string& name, const std::string& module = std::string()) override;
    std::shared_ptr<buffer>              create_buffer(size_t size, void* ptr) override;
    std::shared_ptr<raise_event_command> get_raise_event_command() override;
    std::shared_ptr<commands_sequence>   get_commands_sequence(const std::vector<std::shared_ptr<command>>& commands) override;
    std::shared_ptr<commands_parallel>   get_commands_parallel(const std::vector<std::shared_ptr<command>>& commands) override;

    const ocl_toolkit& toolkit() const { return *_ocl_toolkit; }
    ocl_toolkit& toolkit() { return *_ocl_toolkit; }

private:
    std::unique_ptr<ocl_toolkit> _ocl_toolkit;
};


/// @}
}
