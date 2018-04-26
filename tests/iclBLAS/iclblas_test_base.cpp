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
#include "iclblas_test_base.hpp"

void iclblas_test_environment::SetUp()
{
    auto status = iclblasCreate(&_handle);
ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
}

void iclblas_test_environment::TearDown()
{
    auto status = iclblasDestroy(_handle);
ASSERT_EQ(status, ICLBLAS_STATUS_SUCCESS);
    _handle = NULL;
}

iclblasHandle_t iclblas_test_environment::_handle = NULL;

::testing::Environment* const iclblas_env = ::testing::AddGlobalTestEnvironment(new iclblas_test_environment);

