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
#include <array>
#include <memory>
#include <complex>
#include <algorithm>
#include "context.hpp"
#include "engine.hpp"
#include "errors.hpp"
#include <functional>
#include <numeric>

namespace iclgpu
{
/// @addtogroup functions Functions definition and implementation
/// @{

using complex_t = std::complex<float>;

/// @brief Represents data Blob object passed to a Function
template <typename ElemTy, direction Dir = none>
class blob
{
    std::shared_ptr<buffer_binding> _buffer_binding;
public:
    blob()
        : _buffer_binding(nullptr) {}

    blob(ElemTy* ptr, size_t size)
        : _buffer_binding(std::make_shared<buffer_binding>(ptr, size * sizeof_t<ElemTy>(), Dir))
    {}

    blob(const std::shared_ptr<buffer>& buffer, size_t size = 0)
        : _buffer_binding(std::make_shared<buffer_binding>(buffer, Dir, size))
    {}

    //TODO Remove below by fixing client code.
    blob(ElemTy* ptr)
        : _buffer_binding(std::make_shared<buffer_binding>(ptr, 0, Dir))
    {}

    std::shared_ptr<buffer_binding> get() const { return _buffer_binding; }

    operator ElemTy*() const { return reinterpret_cast<ElemTy*>(_buffer_binding->get_host_ptr()); }
    operator std::shared_ptr<buffer_binding>() const { return get(); }
    operator bool() const { return _buffer_binding && _buffer_binding->is_defined(); }
};

namespace functions
{
template <class Func>
using implementations_list = std::vector<std::shared_ptr<typename Func::impl>>;

template <class Func>
using scored_impls_list = std::vector<std::pair<float, std::shared_ptr<typename Func::impl>>>;

template <class Func>
implementations_list<Func> get_implementations(const std::shared_ptr<context>& ctx) = delete;

/// @brief base class for function score structures
/// @details Every function parameter has dedicated score representation in such structures
/// with the same name and type float.
/// Default value for each score is 1.0.
/// Final implementation score is calculated using some kind of reduction function, e.g. &Sigma; or &Pi;
/// @sa function_impl::accept, score_builder_dot_product
template <size_t N>
struct function_score
{
    using array_type = std::array<float, N>;

    function_score() { _data.fill(1.0f); }

    array_type&       as_array() { return _data; }
    const array_type& as_array() const { return _data; }
protected:
    array_type _data;
};

//TODO remove this alias by changing implementations code to avoid this alias.
using event = std::shared_ptr<iclgpu::event>;

/// @brief Function implementation base class
/// @tparam Func function type
template <typename Func>
struct function_impl : context::holder
{
    typedef Func func_type;
    using holder::holder;

    /// @brief string representation of the implementation
    virtual const char* name() const = 0;

    /// @brief string representation of function and the implementation
    virtual const char* full_name() const = 0;

    /// @brief Check if the implementation supports actual function parameters values
    /// @param[in] params actual function parameters to be checked
    /// @param[in,out] score assuming implementation performance function_score based on actual parameters
    /// @sa function_score, score_builder_dot_product
    virtual bool accept(const typename Func::params&, typename Func::score&) { return false; }
};

/// @brief Function implementation supporting direct execution
/// @tparam Func function type
template <typename Func>
struct function_impl_execute : function_impl<Func>
{
    using function_impl<Func>::function_impl;

    /// @brief Run function implementation 
    /// @param params Actual function parameters
    /// @param dep_events List of events to be waited before start function execution
    /// @returns The event object which indicates whether function execution is completed
    virtual event execute(const typename Func::params& params, const std::vector<event>& dep_events) = 0;
};


/// @brief Function implementation with 2-step execution
/// @tparam Func function type
template <typename Func>
struct function_impl_command : function_impl<Func>
{
    using function_impl<Func>::function_impl;

    /// @brief Functor creates command which executes function implementation
    using command_builder = std::function<std::shared_ptr<command>(const typename Func::params& params)>;

    /// @brief Creates CommandBuilder for the current function implementation
    virtual command_builder selected() = 0;
};

/// @brief Helper class to calculate implementation score based on 'dot-product'.
/// @tparam Func function type
template <class Func>
struct score_builder_dot_product : context::holder
{
    using holder::holder;
    //TODO implement performance DB biases
    //explicit score_builder_dot_product(const std::shared_ptr<Context>& ctx) : holder(ctx)
    //{
    //    //auto& bias = bias_score.as_array();
    //    //auto performanceDb = context()->getPerformanceDB();
    //    //auto data = performanceDb->getFunctionBias(Func::name());
    //    //// TODO clarify correct logic for: DB data size does not match scores size
    //    //std::copy(std::begin(data), std::end(data), bias.begin());
    //}

    /// @brief Calculates implementation score scalar based on function_score structure filled by implementation
    /// @sa function_score, function_impl::accept
    float calculate_score_value(const typename Func::score& score)
    {
        // very simple implementation which just inner products scores and bias from performance DB
        auto& scores = score.as_array();

        return std::accumulate(std::begin(scores), std::end(scores), 0.f);
        // TODO implement score calculation using perf DB biases e.g.:
        // auto& bias = bias_score.as_array();
        // return std::inner_product(std::begin(scores), std::end(scores), std::begin(bias), 0.0f);
    }

    //typename Func::score bias_score;
};


/// @brief Helper class to select function implementation by calling accept() method for each implementation.
/// @tparam Func function type
/// @tparam ScoreCalculator Helper class to calculate implementation score
template <class Func, class ScoreCalculator = score_builder_dot_product<Func>>
struct selector_accept : context::holder
{
    explicit selector_accept(const std::shared_ptr<iclgpu::context>& ctx)
        : holder(ctx)
        , _score_calculator(ctx->get<ScoreCalculator>()) {}


    /// @brief Creates the list of function implementations which accept specified function parameters
    /// @param params Actual function parameters
    scored_impls_list<Func> select(const typename Func::params& params)
    {
        auto impls = functions::get_implementations<Func>(context());
        if (impls.empty())
        {
            throw error_unimplemented("Function is not implemented");
        }

        scored_impls_list<Func> result;

        for (auto& impl : impls)
        {
            // note: default constructor for Func::score should initialize fields by 1.0.
            typename Func::score score;
            // call accept() for each function implementation,
            // calculate and select the high score implementation
            if (impl->accept(params, score))
            {
                float calculated_score = _score_calculator->calculate_score_value(score);
                result.push_back({calculated_score, impl});
            }
        }

        if (result.empty())
        {
            throw error_unsupported("Function paramenters are not supported");
        }

        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        return result;
    }

private:
    std::shared_ptr<ScoreCalculator> _score_calculator;
};
} //namespace functions

/// @}
} // namespace iclgpu
