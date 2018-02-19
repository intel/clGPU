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
#include <utility>
#include "context.hpp"

namespace iclgpu
{
/// @brief Helper class to store kernel sources
class primitive_db : public context::element<primitive_db>
{
public:
    explicit    primitive_db(const std::shared_ptr<iclgpu::context>& ctx);

    /// @brief Get kernel source code by it's name
    std::string get(const std::string& id);

    /// @brief Add kernel sources to the DB
    void        insert(std::initializer_list<std::pair<std::string, std::string>> ilist);
private:
    std::unordered_map<std::string, std::string> _db;
};
}
