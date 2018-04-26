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

#include <gtest/gtest.h>

#include "engine.hpp"
#include "functions_base.hpp"
#include "primitive_db.hpp"

#include <vector>
#include <string>

namespace iclgpu { namespace tests {

using namespace std;

static const vector<pair<string,string>> ocl_kernels {
{"use_value.h",
R"__krnl(
#define USE_VALUE 1
)__krnl"},

{"ocl_engine_test_include",
R"__krnl(
#include "use_value.h"

__kernel void ocl_engine_test_include(int a, __global int* res)
{
#ifdef USE_VALUE
    res[0] = a * 10;
#else
    res[0] = a * 20;
#endif
}
)__krnl"}

};


struct ocl_engine_test : public ::testing::Test
{
    void SetUp() override
    {
        ctx = context::create();
        eng = ctx->get_engine(engine_type::open_cl);
        eng->get_primitive_db()->insert_range(ocl_kernels.begin(), ocl_kernels.end());
    }

    template<unsigned idx, typename T, typename ...Args>
    void set_args(T&& arg, Args&&... args)
    {
        kernel->set_arg(idx , arg);
        set_args<idx + 1, Args...>(std::forward<Args>(args)...);
    }

    template<unsigned idx, typename T>
    void set_args(T&& arg)
    {
        kernel->set_arg(idx, std::forward<T>(arg));
    }

    template<unsigned idx>
    void set_args(){}

    template<typename ...Args>
    void execute_kernel(const std::string& name, const kernel_options& options, Args... args)
    {
        kernel = eng->get_kernel(name);
        set_args<0>(std::forward<Args>(args)...);
        kernel->set_options(options);
        auto event = kernel->submit();
        event->wait();
    }

    std::shared_ptr<context> ctx;
    std::shared_ptr<engine> eng;
    std::shared_ptr<kernel_command> kernel;
};

TEST_F(ocl_engine_test, include)
{
    int32_t a = 111;
    int32_t expected = a * 10;
    int32_t actual = 0;

    blob<int32_t, output> res(&actual, 1);

    execute_kernel("ocl_engine_test_include", {1}, a, res.get());

    EXPECT_EQ(expected, actual);
}

TEST_F(ocl_engine_test, include2)
{
    int32_t a = 111;
    int32_t expected = a * 20;
    int32_t actual = 0;

    blob<int32_t, output> res(&actual, 1);

    // replace header by empty code
    eng->get_primitive_db()->insert({ "use_value.h", "\n" });

    execute_kernel("ocl_engine_test_include", {1}, a, res.get());

    EXPECT_EQ(expected, actual);
}

}}
