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
#include <memory>
#include <exception>
#include <typeinfo>
#include <vector>
#include <type_traits>
#include <chrono>
#include <map>
#include <cassert>

namespace iclgpu
{
class primitive_db;

template <class T, class U>
auto down_pointer_cast(const std::shared_ptr<U>& r)
-> typename std::enable_if<std::is_base_of<U, T>::value && !std::is_same<U, T>::value, std::shared_ptr<T>>::type
{
    auto res = std::dynamic_pointer_cast<T>(r);
    if (!res)
        throw std::bad_cast();
    return res;
}

template <class T, class U>
auto down_pointer_cast(const std::shared_ptr<U>& r)
-> typename std::enable_if<std::is_same<T, U>::value, std::shared_ptr<T>>::type
{
    return r;
}

struct engine;

/// @addtogroup engine Execution engines
/// @{

/// @brief The base class for all Engine connected objects
struct engine_object : std::enable_shared_from_this<engine_object>
{
    explicit engine_object(const std::shared_ptr<engine>& engine)
        : _engine(engine) {}

    virtual ~engine_object() = default;

    /// @brief Returns associated Engine object
    template <class EngTy = engine>
    auto get_engine() const
    -> typename std::enable_if<std::is_base_of<engine, EngTy>::value, std::shared_ptr<EngTy>>::type
    {
        return down_pointer_cast<EngTy>(_engine);
    }

private:
    std::shared_ptr<engine> _engine;
};

/// @brief Represents an event object assotiated with a executed Command
struct event : engine_object
{
    using engine_object::engine_object;
    /// @brief Wait for the event completion
    /// @return duration of the process controlled by the Event (e.g. kernel execution time)
    virtual std::chrono::nanoseconds wait() = 0;
};

/// @brief Represents a command queue within an Engine
struct command_queue
{
    typedef size_t id_t;

    command_queue(id_t id)
        : _id(id) {}

    command_queue()
        : _id(0) {}

    id_t id() const { return _id; }
private:
    id_t _id;
};

/// @brief Default command queue
static const command_queue default_queue;

/// @brief Base class for a command
struct command : engine_object
{
    using engine_object::engine_object;

    /// @brief Submit the command on Engine queue
    /// @param dependencies the List of Events to be waited before command execution start
    /// @param queue reference to the CommandQueue in which the command to be submitted
    /// @return Event will be set on command completion
    virtual std::shared_ptr<event> submit(const std::vector<std::shared_ptr<event>>& dependencies = {},
                                          const command_queue&                       queue        = default_queue) = 0;
};

/// @brief Represents memory buffer
struct buffer : engine_object
{
    using engine_object::engine_object;
    /// @brief Returns size of the memory buffer
    virtual size_t size() const = 0;
    /// @brief Returns direct raw pointer to the buffer data
    virtual void* get_host_ptr() = 0;
};

/// @brief Represents data direction for kernel command
enum direction { none = 0x0, input = 0x1, output = 0x2, inout = input | output };

/// @brief Binds host data and memory buffers per engine describing data direction (in/out) for kernel.
class buffer_binding
{
public:

    /// @brief Constructs binding from host-allocated buffer
    /// @param ptr Raw pointer to host allocated data
    /// @param size Size of host allocated data
    /// @param dir Data direction: host->kernel or host<-kernel or both
    buffer_binding(void* ptr, size_t size = 0, direction dir = inout)
        : _direction(dir)
        , _host_ptr(ptr)
        , _capacity(size)
        , _size(size)
        , _owning_engine(nullptr)
        , _buffers({})
    {
        assert(_host_ptr);
    }

    /// @brief Constructs binding from engine-allocated memory buffer
    /// @param buffer Pre-allocated memory buffer
    /// @param dir Data direction: host->kernel or host<-kernel or both
    /// @param size Size of data to be used - should be less or equal to buffer size, if set to 0 then buffer size will be used.
    buffer_binding(const std::shared_ptr<buffer>& buffer, direction dir = inout, size_t size = 0)
        : _direction(dir)
        , _host_ptr(nullptr)
        , _capacity(buffer->size())
        , _size(size != 0 ? size : buffer->size())
        , _owning_engine(buffer->get_engine())
        , _buffers({{buffer->get_engine(), buffer}})
    {
        assert(_capacity >= _size);
    }

    /// @brief Get direction of the binding.
    direction get_direction() const { return _direction; }

    /// @brief Set direction of the binding
    /// @returns Previous value
    direction set_direction(direction dir);

    /// @brief Returns engine of the buffer passed to constructor or NULL if consrtucted from host-allocated data.
    const std::shared_ptr<engine>& get_owning_engine() const { return _owning_engine; }

    /// @brief Raw pointer to data.
    /// @returns Host-allocated pointer or mapped buffer depending on constrcution.
    void* get_host_ptr() const;

    /// @brief Reset internal host pointer to NULL if constructed from buffer.
    void reset_host_ptr() const;

    /// @brief Returns size of data
    size_t size() const { return _size; }

    /// @brief Returns maximum possible size value
    size_t capacity() const { return _capacity; }

    /// @brief Set new size.
    /// @details New size should be less or equal to capacity
    void size(size_t size);

    /// @brief Returns binded buffer for the specified engine
    /// @details If there is no buffer binded - create and register new binded buffer
    std::shared_ptr<buffer> get_buffer(const std::shared_ptr<engine>& engine) const;

    bool is_output() const { return (_direction & output) != 0; }
    bool is_input() const { return (_direction & input) != 0; }

    /// @brief Check if binding is fully defined: size is set and constructed from correct host-allocated pointer of buffer
    bool is_defined() const
    {
        return _size != 0 && (_host_ptr || _owning_engine);
    }

protected:
    direction               _direction;
    mutable void*           _host_ptr;
    size_t                  _capacity;
    size_t                  _size;
    std::shared_ptr<engine> _owning_engine;
    mutable std::map<std::shared_ptr<engine>, std::shared_ptr<buffer>> _buffers;
};

template <typename ElemTy> size_t sizeof_t()       { return sizeof(ElemTy); }
template <> inline size_t         sizeof_t<void>() { return 1; }

/// @brief Represents a command which just raises an event
struct raise_event_command : command
{
    using command::command;
};

/// @brief Represents set of command to be executed sequentially
struct commands_sequence : command
{
    using command::command;

    /// @brief Add new command to the end of sequence
    void push_back(const std::shared_ptr<command>& command);

    std::shared_ptr<event> submit(const std::vector<std::shared_ptr<event>>& dependencies = {},
                                  const command_queue&                       queue        = default_queue) override;
protected:
    std::vector<std::shared_ptr<command>> _commands;
};

/// @brief Represents set of command can be executed in parallel
struct commands_parallel : command
{
    using command::command;

    /// @brief Add new command
    void add(const std::shared_ptr<command>& command);

    std::shared_ptr<event> submit(const std::vector<std::shared_ptr<event>>& dependencies = {},
                                  const command_queue&                       queue        = default_queue) override;
private:
    std::vector<std::shared_ptr<command>> _commands;
};

/// @brief Represents ND (1D, 2D, 3D) ranges for kernels executions.
struct nd_range
{
    nd_range()
        : _size(0)
        , _values{0, 0, 0} {}

    nd_range(size_t x)
        : _size(1)
        , _values{x, 0, 0} {}

    nd_range(size_t x, size_t y)
        : _size(2)
        , _values{x, y, 0} {}

    nd_range(size_t x, size_t y, size_t z)
        : _size(3)
        , _values{x, y, z} {}

    size_t        dimensions() const { return _size; }
    const size_t* values()     const { return _values; }

    const size_t& operator[](size_t idx) const
    {
        assert(idx < _size);
        return _values[idx];
    }

    size_t& operator[](size_t idx)
    {
        assert(idx < _size);
        return _values[idx];
    }

private:
    size_t _size;
    size_t _values[3];
};

/// @brief Null range (unspecified)
static const nd_range null_range;

/// @brief Kernel execution options
struct kernel_options
{
    /// @brief Constructs kernel options with specified global and parallel ranges
    kernel_options(const nd_range& work_size, const nd_range& parallel_size = null_range);

    virtual ~kernel_options() = default;

    /// @brief Range for kernel instances (global worksize)
    const nd_range& work_size() const { return _work_size; }

    /// @brief Range of kernel instances to be executed in paralled (local worksize)
    const nd_range& parallel_size() const { return _parallel_size; }
private:
    nd_range _work_size;
    nd_range _parallel_size;
};

/// @brief Represents kernel command
struct kernel_command : command
{
    using command::command;

    /// @brief Set kernel argument
    /// @param idx Argument index
    /// @param value Argument value
    template <typename T>
    auto set_arg(unsigned idx, const T& value)
    -> typename std::enable_if<!std::is_pointer<T>::value>::type
    {
        set_scalar_arg(idx, &value, sizeof(T));
    }

    /// @brief Set kernel argument (specialized for data buffers)
    /// @param idx Argument index
    /// @param value Argument value
    void set_arg(unsigned idx, const std::shared_ptr<buffer_binding>& value)
    {
        assert(value->is_defined());
        set_buffer_arg(idx, value);
    }

    /// @brief Set kernel execution options
    /// @details Should be overriden for specific engine
    virtual void set_options(const kernel_options& params) = 0;

protected:
    /// @brief Set scalar function argument
    /// @details Should be overriden for specific engine
    /// @param idx Argument index
    /// @param ptr Pointer to scalar value
    /// @param size Size of value
    virtual void set_scalar_arg(unsigned idx, const void* ptr, size_t size) = 0;

    /// @brief Set buffer function argument
    /// @details Should be overriden for specific engine
    /// @param idx Argument index
    /// @param binding Data binding object
    virtual void set_buffer_arg(unsigned idx, const std::shared_ptr<buffer_binding>& binding) = 0;
};

template <typename ElemTy, direction Dir> class blob;

/// @brief Base class for execution engines
struct engine
{
    virtual ~engine() = default;

    /// @brief Return engine-specific kernels DB
    virtual primitive_db* get_primitive_db() = 0;

    /// @brief Create kernel command
    /// @param name Kernel name
    /// @param module (optional) Module name in which kernel is defined
    virtual std::shared_ptr<kernel_command> get_kernel(const std::string& name,
                                                       const std::string& module = std::string()) = 0;

    /// @brief Create memory buffer within the engine
    /// @param size Requested buffer size
    /// @param ptr (optional) Raw pointer from which data should be copied to memory buffer
    virtual std::shared_ptr<buffer> create_buffer(size_t size, void* ptr = nullptr) = 0;

    /// @brief Create raise event command
    virtual std::shared_ptr<raise_event_command> get_raise_event_command() = 0;

    /// @brief Create command sequence
    /// @param commands (optional) Commands to be added into the sequence
    virtual std::shared_ptr<commands_sequence> get_commands_sequence(const std::vector<std::shared_ptr<command>>& commands = {}) = 0;

    /// @brief Create set of parrallel executed commands
    /// @param commands (optional) Commands can be executed in parallel
    virtual std::shared_ptr<commands_parallel> get_commands_parallel(const std::vector<std::shared_ptr<command>>& commands = {}) = 0;

    /// @brief Create temporary buffer with @b num elements of @b T
    template <typename T = char>
    std::shared_ptr<buffer_binding> get_temp_buffer(size_t num)
    {
        if (num == 0) throw std::invalid_argument("size should not be zero.");
        return std::make_shared<buffer_binding>(create_buffer(num * sizeof_t<T>(), nullptr), none);
    }

    /// @brief Get input buffer binding from Blob object
    template <typename T, direction Dir>
    static auto get_input_buffer(const blob<T, Dir>& blob, size_t num)
    -> typename std::enable_if<(Dir & input) != 0, std::shared_ptr<buffer_binding>>::type
    {
        return set_binding_options<T>(blob.get(), num, input);
    }

    /// @brief Get output buffer binding from Blob object
    template <typename T, direction Dir>
    static auto get_output_buffer(const blob<T, Dir>& blob, size_t num)
    -> typename std::enable_if<(Dir & output) != 0, std::shared_ptr<buffer_binding>>::type
    {
        return set_binding_options<T>(blob.get(), num, output);
    }

    /// @brief Get in-out buffer binding from Blob object
    template <typename T, direction Dir>
    static auto get_inout_buffer(const blob<T, Dir>& blob, size_t num)
    -> typename std::enable_if<Dir == inout, std::shared_ptr<buffer_binding>>::type
    {
        return set_binding_options<T>(blob.get(), num, Dir);
    }

private:
    template <typename T>
    static std::shared_ptr<buffer_binding> set_binding_options(std::shared_ptr<buffer_binding>&& binding,
                                                               size_t                            num,
                                                               direction                         dir)
    {
        binding->set_direction(dir);
        binding->size(num * sizeof_t<T>());
        return binding;
    }
};


/// @}
}
