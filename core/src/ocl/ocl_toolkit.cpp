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

class ocl_primitive_db : public primitive_db
{
public:
    ocl_primitive_db(ocl_toolkit* toolkit)
        : _toolkit(toolkit)
    {
        _db.insert({
            #include CORE_OCL_KERNELS_DB
        });
    }

    struct headers_ret { size_t size; const char* const* names; const ::cl_program* headers; };
    headers_ret headers() const
    {
        return{ _names.size(), _names.data(), _programs.data() };
    }

    void insert(const value_type& value) override
    {
        primitive_db::insert(value);
        auto& name = value.first;
        auto& code = value.second;
        const auto name_len = name.length();
        if (name_len > 2 && name.compare(name_len - 2, 2, ".h") == 0)
        {

            insert_header(name, code);
        }
    }

private:
    ocl_toolkit* _toolkit;
    std::vector<const char*> _names;
    std::vector<std::unique_ptr<std::string>> _names_store;
    std::vector<::cl_program> _programs;
    std::vector<std::unique_ptr<cl::Program>> _programs_store;

    void insert_header(const std::string& name, const std::string& code)
    {
        assert(_names.size() * 4 == _names.size() + _names_store.size() + _programs.size() + _programs_store.size() );

        auto prog = std::make_unique<cl::Program>(_toolkit->get_cl_context(), code, false);

        for (size_t i = 0; i < _names_store.size(); ++i)
        {
            if (name == *_names_store[i])
            {
                // replace
                _programs[i] = prog->get();
                _programs_store[i] = std::move(prog);
                assert(_programs_store[i]->get() == _programs[i]);
                return;
            }
        }

        auto new_name = std::make_unique<std::string>(name);
        _names.push_back(new_name->c_str());
        _names_store.push_back(std::move(new_name));

        _programs.push_back(prog->get());
        _programs_store.push_back(std::move(prog));

        assert(_names_store.back()->c_str() == _names.back());
        assert(_programs_store.back()->get() == _programs.back());
        assert(_names.size() * 4 == _names.size() + _names_store.size() + _programs.size() + _programs_store.size() );
    }
};

ocl_toolkit::ocl_toolkit(ocl_engine* engine)
    : _engine(engine)
    , _device(get_gpu_device())
    , _ocl_context(_device)
    , _queues{cl::CommandQueue{_ocl_context, _device, CL_QUEUE_PROFILING_ENABLE}}
    , _primitive_db(std::make_unique<ocl_primitive_db>(this))
{
    assert(_engine);
}

ocl_toolkit::~ocl_toolkit() = default;

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

cl::Program ocl_toolkit::build_program(const std::string& module_name)
{
    cl::Program program(_ocl_context, cl::Program::Sources{_primitive_db->get("complex.h"), _primitive_db->get(module_name)});

    auto headers = _primitive_db->headers();

    // This is WA for bad signature, because Khronos committee does not understand how the C/C++ const keyword actually works
    auto header_names = const_cast<const char**>(headers.names);

    auto error = ::clCompileProgram(
            program(),              // cl_program program
            1,                      // cl_uint num_devices
            &_device(),             // const cl_device_id* devices
            NULL,                   // const char* compiler_options
            (cl_uint)headers.size,  // cl_uint num_input_headers
            headers.headers,        // const cl_program *input_headers
            header_names,           // const char **header_include_names
            NULL,                   // void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data)
            NULL);                  // void *user_data

    cl::detail::buildErrHandler(error, "clCompileProgram", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>());

    auto prog = ::clLinkProgram(
            _ocl_context(),   // cl_context context
            1,              // cl_uint num_devices
            &_device(),     // const cl_device_id *device_list
            NULL,           // const char *options
            1,              // cl_uint num_input_programs
            &program(),     // const cl_program *input_programs
            NULL,           // void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data)
            NULL,           // void *user_data
            &error);        // cl_int *errcode_ret

    cl::detail::errHandler(error, "clLinkProgram");

    return cl::Program(prog, false);
}

const cl::Program& ocl_toolkit::get_module(const std::string& module_name)
{
    auto it = _programs.find(module_name);
    if (it != _programs.end())
        return it->second;

    auto program = build_program(module_name);
    auto inserted = _programs.emplace(module_name, program);
    assert(inserted.second);
    return inserted.first->second;
}

primitive_db* ocl_toolkit::get_primitive_db() const { return _primitive_db.get(); }
}
