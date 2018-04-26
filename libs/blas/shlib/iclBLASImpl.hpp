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

#include "iclBLAS.h"
#include "context.hpp"
#include "dispatcher.hpp"
#include "errors.hpp"
#include <functional>

// The BLAS implementation
// in global namespace, because it is definition fo C API opaque structure
struct iclblasContext
{
    iclblasContext();
    ~iclblasContext();
    std::shared_ptr<iclgpu::context> get_iclgpuContext() const { return _gen_cl_context; }
    static void validate(iclblasHandle_t handle);
private:
    static const int tag_value = 0xB1A5;
    int _tag;
    std::shared_ptr<iclgpu::context> _gen_cl_context;
};

namespace iclblas {

inline iclgpu::complex_t complex_cast(const oclComplex_t& a) { return a; }
inline iclgpu::complex_t* complex_cast(oclComplex_t* a) { return a; }
inline const iclgpu::complex_t* complex_cast(const oclComplex_t* a) { return a; }

// internal error handling is based on exceptions
// The following function translate exceptions to C API error codes.
inline iclblasStatus_t exception_to_iclblas_status(std::function<void()> func)
{
#ifdef NDEBUG
    try
    {
        func();
        return ICLBLAS_STATUS_SUCCESS;
    }
    catch (const iclgpu::error_unimplemented&)
    {
        return ICLBLAS_STATUS_NOT_SUPPORTED;
    }
    catch (const iclgpu::error_unsupported&)
    {
        return ICLBLAS_STATUS_NOT_SUPPORTED;
    }
    catch (const std::invalid_argument&)
    {
        return ICLBLAS_STATUS_INVALID_VALUE;
    }
    catch (...)
    {
        return ICLBLAS_STATUS_ERROR;
    }
#else
    func();
    return ICLBLAS_STATUS_SUCCESS;
#endif
}


//////////////////////////////////////////////////////
// Validation
template<typename T, iclgpu::direction Dir>
bool validate_blob(const iclgpu::blob<T, Dir>& blb)
{
    return static_cast<T*>(blb) != nullptr;
}

#define DEFINE_HAS_MEMBER(member)                                        \
template <class C>                                                       \
class has_member_##member                                                \
{                                                                        \
private:                                                                 \
    template < class U >                                                 \
    static constexpr bool test ( decltype(U::member)* ) { return true; } \
    template < class U >                                                 \
    static constexpr bool test (...) {return false; }                    \
public:                                                                  \
    static constexpr bool value = test<C>(nullptr);                      \
};

#define VALIDATE_MEMBER(member)                                          \
DEFINE_HAS_MEMBER(member)                                                \
                                                                         \
template<class Params>                                                   \
typename std::enable_if<!has_member_##member<Params>::value, bool>::type \
 validate_param_##member(const Params& params) { return true; }          \
                                                                         \
template<class Params>                                                   \
typename std::enable_if<has_member_##member<Params>::value, bool>::type  \
validate_param_##member(const Params& params)

VALIDATE_MEMBER(n)
{
    if(params.n < 0)
        throw std::invalid_argument("n");
    return params.n > 0;
}

VALIDATE_MEMBER(m)
{
    if(params.m < 0)
        throw std::invalid_argument("m");
    return params.m > 0;
}

VALIDATE_MEMBER(k)
{
    if(params.k < 0)
        throw std::invalid_argument("k");
    return true;
}

VALIDATE_MEMBER(x)
{
    if(!validate_blob(params.x))
        throw std::invalid_argument("x");
    return true;
}

VALIDATE_MEMBER(incx)
{
    if(params.incx == 0)
        throw std::invalid_argument("incx");
    if(params.incx < 0)
        throw iclgpu::error_unsupported("incx");
    return true;
}

VALIDATE_MEMBER(y)
{
    if(!validate_blob(params.y))
        throw std::invalid_argument("y");
    return true;
}

VALIDATE_MEMBER(incy)
{
    if(params.incy == 0)
        throw std::invalid_argument("incy");
    if(params.incy < 0)
        throw iclgpu::error_unsupported("incy");
    return true;
}

VALIDATE_MEMBER(A)
{
    if(!validate_blob(params.A))
        throw std::invalid_argument("A");
    return true;
}

VALIDATE_MEMBER(lda)
{
    if(params.lda <= 0)
        throw std::invalid_argument("lda");
    return true;
}

VALIDATE_MEMBER(B)
{
    if(!validate_blob(params.B))
        throw std::invalid_argument("B");
    return true;
}

VALIDATE_MEMBER(ldb)
{
    if(params.ldb <= 0)
        throw std::invalid_argument("ldb");
    return true;
}

VALIDATE_MEMBER(C)
{
    if(!validate_blob(params.C))
        throw std::invalid_argument("C");
    return true;
}

VALIDATE_MEMBER(ldc)
{
    if(params.ldc <= 0)
        throw std::invalid_argument("ldc");
    return true;
}

VALIDATE_MEMBER(result)
{
    if(!validate_blob(params.result))
        throw std::invalid_argument("result");
    return true;
}

VALIDATE_MEMBER(AP)
{
    if(!validate_blob(params.AP))
        throw std::invalid_argument("AP");
    return true;
}

template<class Params>
bool validate_params(const Params& params)
{
    return validate_param_n(params)
        && validate_param_m(params)
        && validate_param_k(params)
        && validate_param_x(params)
        && validate_param_incx(params)
        && validate_param_y(params)
        && validate_param_incy(params)
        && validate_param_A(params)
        && validate_param_lda(params)
        && validate_param_B(params)
        && validate_param_ldb(params)
        && validate_param_C(params)
        && validate_param_ldc(params);
}

// Generic template implementation for BLAS functions
template<typename Func> void iclblasTemplate_impl(iclblasHandle_t handle, typename Func::params& params) {
    if(!validate_params(params))
        return;

    iclblasContext::validate(handle);
    auto context = handle->get_iclgpuContext();
    auto dispatcher = context->get_dispatcher();
    auto event = dispatcher->execute_function<Func>(params);
    event->wait();
}

}
