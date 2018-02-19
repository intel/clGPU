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

#include "iclBLAS.h"
#include "context.hpp"
#include "dispatcher.hpp"
#include "primitive_db.hpp"
#include "iclBLASImpl.hpp"

iclblasContext::iclblasContext()
    : _tag(tag_value), _gen_cl_context(iclgpu::context::create())
    {
        _gen_cl_context->get<iclgpu::primitive_db>()->insert({
        #include "ocl_kernels.inc"
        });
    }

iclblasContext::~iclblasContext()
{
    _gen_cl_context.reset();
    _tag = 0;
}

void iclblasContext::validate(iclblasHandle_t handle)
{
    if (handle == nullptr || handle->_tag != tag_value)
        throw std::invalid_argument("handle");
}

// BLAS C API implementation
extern "C"
iclblasStatus_t iclblasCreate(iclblasHandle_t* handle)
{
    if(!handle)
    {
        return ICLBLAS_STATUS_INVALID_VALUE;
    }
    *handle = new iclblasContext();
    return ICLBLAS_STATUS_SUCCESS;
}

extern "C"
iclblasStatus_t iclblasDestroy(iclblasHandle_t handle)
{
    iclblasContext::validate(handle);
    delete handle;
    return ICLBLAS_STATUS_SUCCESS;
}
