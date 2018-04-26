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
#include <vector>
#include <unordered_map>
#include <string>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>

namespace iclgpu
{
class ocl_engine;
class primitive_db;
class ocl_primitive_db;

class ocl_toolkit
{
public:
    ocl_toolkit(ocl_engine* engine);
    ~ocl_toolkit(); // -required because ocl_primitive_db is incomplete type
    const cl::Context& get_cl_context() const;
    cl::CommandQueue& get_cl_queue(const command_queue& queue = default_queue);
    static cl::Device get_gpu_device();
    cl::Program build_program(const std::string& module_name);
    const cl::Program& get_module(const std::string& module_name);
    primitive_db* get_primitive_db() const;

private:
    ocl_engine*                                  _engine;
    cl::Device                                   _device;
    cl::Context                                  _ocl_context;
    std::vector<cl::CommandQueue>                _queues;
    std::unordered_map<std::string, cl::Program> _programs;
    std::unique_ptr<ocl_primitive_db>            _primitive_db;
};

}
