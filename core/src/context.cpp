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

#include "context.hpp"
#include "dispatcher.hpp"
#include "ocl/ocl_engine.hpp"

namespace iclgpu
{
std::shared_ptr<dispatcher> context::get_dispatcher() { return get<dispatcher>(); }

std::shared_ptr<engine> context::get_engine(engine_type type)
{
    switch (type)
    {
    case engine_type::default_engine: return get_engine(_default_engine_type);
    case engine_type::open_cl: return get<ocl_engine>();
    default: throw std::invalid_argument("unknown engine type");
    }
}
}
