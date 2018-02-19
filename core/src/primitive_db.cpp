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

#include "primitive_db.hpp"

namespace iclgpu
{
DEFINE_CLASS_ID(primitive_db)

primitive_db::primitive_db(const std::shared_ptr<iclgpu::context>& ctx)
    : element(ctx)
    , _db({
        #include "ocl_kernels.inc"
    })
{}

std::string primitive_db::get(const std::string& id)
{
    return _db.at(id);
}

void primitive_db::insert(std::initializer_list<std::pair<std::string, std::string>> ilist)
{
    _db.insert(ilist.begin(), ilist.end());
}
}
