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
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>
#include <complex>

#include <context.hpp>
#include <dispatcher.hpp>
#include <functions_base.hpp>

namespace iclgpu { namespace tests {

class test_env : public ::testing::Environment
{
public:
    void SetUp() override;
    void TearDown() override;

    static decltype(iclgpu::context::create()) get_context()
    {
        if (!_ctx) throw std::runtime_error("iclgpu context is not created");
        return _ctx;
    }
private:
    static decltype(iclgpu::context::create()) _ctx;
};


template<class Impl>
typename std::enable_if<std::is_base_of<iclgpu::functions::function_impl_command<typename Impl::func_type>, Impl>::value>::type
run_implementation(typename Impl::func_type::params& params, const std::shared_ptr<Impl>& impl)
{
    auto cmdBuilder = impl->selected();
    auto cmd = cmdBuilder(params);
    auto event = cmd->submit();
    event->wait();
}

template<class Impl>
typename std::enable_if<std::is_base_of<iclgpu::functions::function_impl_execute<typename Impl::func_type>, Impl>::value>::type
run_implementation(typename Impl::func_type::params& params, const std::shared_ptr<Impl>& impl)
{
    auto event = impl->execute(params, {});
    event->wait();
}

template<class Func>
typename std::enable_if<std::is_base_of<iclgpu::functions::function_impl<Func>, typename Func::impl>::value>::type
run_function(typename Func::params& params)
{
    auto ctx = test_env::get_context();
    const auto disp = ctx->get_dispatcher();
    auto impls = disp->select<Func>(params);
    ASSERT_NE(impls.size(), size_t(0u)) << "Function parameters are not supported";

    auto impl = impls[0].second;
    run_implementation(params, impl);
}

template<class Func>
typename std::enable_if<std::is_base_of<iclgpu::functions::function_impl<Func>, typename Func::impl>::value>::type
run_function(typename Func::params& params, const std::string& impl_name)
{
    if (impl_name.empty())
    {
        run_function<Func>(params);
        return;
    }

    auto ctx = test_env::get_context();
    auto impls = iclgpu::functions::get_implementations<Func>(ctx);
    for(auto impl : impls)
    {
        if(impl_name == impl->name())
        {
            typename Func::score score;
            ASSERT_TRUE(impl->accept(params, score)) << "The implementation does not support specified parameters";
            run_implementation(params, impl);
            return;
        }
    }

    FAIL() << "Implementation '" << impl_name << "' for the function '" << Func::name() << "' not found";
}

template<class Impl>
typename std::enable_if<std::is_base_of<iclgpu::functions::function_impl<typename Impl::func_type>, Impl>::value>::type
run_function(typename Impl::func_type::params& params)
{
    auto ctx = test_env::get_context();
    auto impl = ctx->get<Impl>();

    typename Impl::func_type::score score;
    ASSERT_TRUE(impl->accept(params, score)) << "The implementation does not support specified parameters";

    run_implementation(params, impl);
}

template<typename T>
T get_random_scalar() = delete;

template <>
inline float get_random_scalar<float>()
{
    static std::minstd_rand gen;
    const float base = 0.25f;
    static std::uniform_int_distribution<int> dist(-4, 4);
    return base * static_cast<float>(dist(gen));
}

template <>
inline iclgpu::complex_t get_random_scalar<iclgpu::complex_t>()
{
    return iclgpu::complex_t(get_random_scalar<float>(), get_random_scalar<float>());
}

template<typename T, class Allocator = std::allocator<T>>
std::vector<T, Allocator> get_random_vector(size_t size)
{
    std::vector<T, Allocator> result(size);
    std::generate(result.begin(), result.end(), [&] { return get_random_scalar<T>(); });
    return result;
}

inline double relative_error(double reference, double value)
{
    return abs(1.0 - reference / value);
}

inline bool complex_equal(iclgpu::complex_t a, iclgpu::complex_t b )
{
    ::testing::internal::FloatingPoint<float> ra(real(a)), rb(real(b)), ia(imag(a)), ib(imag(b));
    return ra.AlmostEquals(rb) && ia.AlmostEquals(ib);
}

#define EXPECT_COMPLEX_EQ(a,b) EXPECT_PRED2(complex_equal, a, b)

template<class Func, 
         typename = std::enable_if_t<std::is_base_of<iclgpu::functions::function_impl<Func>, typename Func::impl>::value>>
struct func_traits {};

using TestParamTypeVV = ::testing::tuple<
    const char*,    //impl_name
    int,            //num
    int,            //incx
    int>;           //incy

template<class Func>
struct test_base_VV : testing::TestWithParam<TestParamTypeVV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    int num;
    int incx;
    int incy;
    data_arr_type x;
    data_arr_type y;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = ::testing::get<0>(GetParam());
        num = ::testing::get<1>(GetParam());
        incx = ::testing::get<2>(GetParam());
        incy = ::testing::get<3>(GetParam());
        x = get_random_vector<data_type>(num * incx);
        y = get_random_vector<data_type>(num * incy);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypeV = ::testing::tuple<
    const char*,    //impl_name
    int,            //num
    int>;           //incx


template<class Func>
struct test_base_V : testing::TestWithParam<TestParamTypeV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    int num;
    int incx;
    data_arr_type x;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = ::testing::get<0>(GetParam());
        num = ::testing::get<1>(GetParam());
        incx = ::testing::get<2>(GetParam());
        x = get_random_vector<data_type>(num * incx);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypePMV = testing::tuple<
    const char*,    // impl_name
    int,            // uplo
    int,            // num
    int>;           // incx

template <class Func>
struct test_base_PMV : testing::TestWithParam<TestParamTypePMV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    int uplo;
    int num;
    int incx;
    data_arr_type x;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = ::testing::get<0>(GetParam());
        uplo = ::testing::get<1>(GetParam());
        num = ::testing::get<2>(GetParam());
        incx = ::testing::get<3>(GetParam());
        x = get_random_vector<data_type>(num * incx);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypePMVV = ::testing::tuple<
    const char*,    // impl_name
    int,            // uplo
    int,            // num
    int,            // incx
    int>;           // incy

template<class Func>
struct test_base_PMVV : testing::TestWithParam<TestParamTypePMVV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    int uplo;
    int num;
    int incx;
    int incy;
    data_arr_type x;
    data_arr_type y;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = testing::get<0>(GetParam());
        uplo = testing::get<1>(GetParam());
        num = testing::get<2>(GetParam());
        incx = testing::get<3>(GetParam());
        incy = testing::get<4>(GetParam());
        x = get_random_vector<data_type>(num * incx);
        y = get_random_vector<data_type>(num * incy);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

}}
