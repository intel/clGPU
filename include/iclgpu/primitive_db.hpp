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
#include <unordered_map>
#include <initializer_list>
#include <algorithm>
#include <string>

namespace iclgpu
{
/// @brief Helper class to store kernel sources
class primitive_db
{
public:
    using db_type = std::unordered_map<std::string, std::string>;
    using value_type = db_type::value_type;

    virtual ~primitive_db() = default;

    /// @brief Get kernel source code by it's name
    std::string get(const std::string& id);


    /// @brief Add kernel source to the DB
    virtual void insert(const value_type& value);

    /// @brief Add kernel sources to the DB
    template <class InputIt>
    std::enable_if_t<std::is_convertible<decltype(*std::declval<InputIt>()), value_type>::value>
    insert_range(InputIt first, InputIt last)
    {
        std::for_each(first, last, [this](const value_type& p) -> void { insert(p); });
    }

    /// @brief Add kernel sources to the DB
    template <class Range>
    std::enable_if_t<std::is_convertible<decltype(*std::cbegin(std::declval<Range>())), value_type>::value>
    insert(const Range& range)
    {
        insert_range(std::cbegin(range), std::cend(range));
    }

    /// @brief Add kernel sources to the DB
    void insert(const std::initializer_list<value_type>& ilist)
    {
        insert_range(std::cbegin(ilist), std::cend(ilist));
    }

protected:
    db_type _db;
};

}
