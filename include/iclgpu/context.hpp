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
#include "container.hpp"

namespace iclgpu
{
class dispatcher;
struct engine;

/// @addtogroup context Context management
/// @{

/// @brief supported engine types
enum class engine_type
{
    default_engine, ///< engine type registered as default
    open_cl         ///< Open CL engine
};

/// @brief Global library context provides access to all library objects.
class context : public container<context>, public std::enable_shared_from_this<context>
{
    context()
        : _default_engine_type(engine_type::open_cl) {}

public:
    /// @brief Return function dispatcher
    std::shared_ptr<dispatcher> get_dispatcher();

    /// @brief Return execution engine
    /// @param engine_type engine type to be returned
    std::shared_ptr<engine> get_engine(engine_type type = engine_type::default_engine);

    /// @brief Create Context instance
    static std::shared_ptr<context> create() { return std::shared_ptr<context>(new context); }
private:
    engine_type _default_engine_type;
};

/// @}
} // namespace iclgpu
