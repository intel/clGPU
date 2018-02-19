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
#include "errors.hpp"
#include "functions_base.hpp"
#include "engine.hpp"
#include "context.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

namespace iclgpu
{
/// @addtogroup functions Functions definition and implementation
/// @{

/// @brief Function implementations dispatcher
class dispatcher : public context::element<dispatcher>
{
public:
    using element::element;

    /// @brief Returns list of function implementations which accept specified function parameters
    /// @param params Actual function parameters
    template <class Func>
    functions::scored_impls_list<Func> select(typename Func::params& params) const
    {
        auto selector = context()->get<typename Func::selector>();
        return selector->select(params);
    }

    /// @brief Executes a function
    /// @tparam Func the function to be executed
    /// @param params Actual function parameters
    /// @param dep_events List of events to be waited before start function execution
    /// @returns The event object which indicates whether function execution is completed
    template <class Func>
    typename std::enable_if<std::is_base_of<functions::function_impl_execute<Func>, typename Func::impl>::value,
                            std::shared_ptr<event>>::type
    execute_function(typename Func::params& params, const std::vector<std::shared_ptr<event>>& dep_events = {}) const
    {
        auto impls = select<Func>(params);
        if (impls.size() == 0)
        {
            throw error_unsupported("Function parameters are not supported");
        }

        return impls[0].second->execute(params, dep_events);
    }

    /// @brief Executes a function
    /// @tparam Func the function to be executed
    /// @param params Actual function parameters
    /// @param dep_events List of events to be waited before start function execution
    /// @returns The event object which indicates whether function execution is completed
    template <class Func>
    typename std::enable_if<std::is_base_of<functions::function_impl_command<Func>, typename Func::impl>::value,
                            std::shared_ptr<event>>::type
    execute_function(typename Func::params& params, const std::vector<std::shared_ptr<event>>& dep_events = {}) const
    {
        auto impls = select<Func>(params);
        if (impls.size() == 0)
        {
            throw error_unsupported("Function parameters are not supported");
        }

        auto cmd_builder = impls[0].second->selected();
        if (!cmd_builder) throw std::logic_error("selected() is not implemented.");
        auto cmd = cmd_builder(params);
        return cmd->submit(dep_events);
    }
};

/// @}
} // namespace iclgpu
