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

#include "iclblas_common.h.cl"

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

template<typename T, class Allocator = std::allocator<T>>
std::vector<T, Allocator> get_random_vector(int size)
{
    return get_random_vector<T, Allocator>(static_cast<size_t>(abs(size)));
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

inline ::testing::AssertionResult complex_near(const char* expr1,
                                               const char* expr2,
                                               const char* abs_error_expr,
                                               iclgpu::complex_t a,
                                               iclgpu::complex_t b,
                                               double abs_error)
{
    auto diff = std::abs(static_cast<std::complex<double>>(a) - static_cast<std::complex<double>>(b));

    if (diff < abs_error) return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
        << "The difference between " << expr1 << " and " << expr2
        << " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
        << expr1 << " evaluates to " << a << ",\n"
        << expr2 << " evaluates to " << b << ", and\n"
        << abs_error_expr << " evaluates to " << abs_error << ".";
}

#define EXPECT_COMPLEX_EQ(a,b) EXPECT_PRED2(complex_equal, a, b)
#define EXPECT_COMPLEX_NEAR(a,b, diff) EXPECT_PRED_FORMAT3(complex_near, a, b, diff)

template<class Func,
         typename = std::enable_if_t<std::is_base_of<iclgpu::functions::function_impl<Func>, typename Func::impl>::value>>
struct func_traits {};

template<typename data_type>
struct accumulator_type
{
    using type = data_type;
};

template<>
struct accumulator_type<float>
{
    using type = double;
};

template<>
struct accumulator_type<iclgpu::complex_t>
{
    using type = std::complex<double>;
};

template<typename data_type>
using accumulator_type_t = typename accumulator_type<data_type>::type;

template<typename data_type>
constexpr auto blas_abs(data_type number)
{
    using std::abs;
    return abs(number);
}

template<typename complex_type>
constexpr auto blas_abs(std::complex<complex_type> number)
{
    using std::abs;
    return abs(number.real()) + abs(number.imag());
}

template<typename data_type>
using absolute_type_t = decltype(blas_abs(std::declval<data_type>()));

template<typename data_type>
constexpr auto blas_conj(data_type number)
{
    return number;
}

template<typename complex_type>
constexpr auto blas_conj(std::complex<complex_type> number)
{
    using std::conj;
    return conj(number);
}

template<typename data_type>
auto zeroe_imag(data_type x)
{
    return x;
}

template<typename complex_type>
auto zeroe_imag(std::complex<complex_type> x)
{
    return std::complex<complex_type>(x.real(), 0);
}

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

template<class Func, typename _result_type>
struct test_base_VS : test_base_V<Func>
{
    using result_type = _result_type;

    result_type result;
    result_type result_ref;

    typename Func::params get_params() override
    {
        return{
            this->num,
            this->x.data(),
            this->incx,
            &result
        };
    }

    typename Func::params get_params_ref() override
    {
        return{
            this->num,
            this->x.data(),
            this->incx,
            &result_ref
        };
    }
};

using TestParamTypePMV = testing::tuple<
    const char*,        // impl_name
    iclblasFillMode_t,  // uplo
    int,                // num
    int>;               // incx

template <class Func>
struct test_base_PMV : testing::TestWithParam<TestParamTypePMV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasFillMode_t uplo;
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
    const char*,        // impl_name
    iclblasFillMode_t,  // uplo
    int,                // num
    int,                // incx
    int>;               // incy

template<class Func>
struct test_base_PMVV : testing::TestWithParam<TestParamTypePMVV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasFillMode_t uplo;
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

using TestParamTypeTMV = ::testing::tuple<
    const char*,        // impl_name
    iclblasFillMode_t,  // uplo
    iclblasOperation_t, // trans
    iclblasDiagType_t,  // diag
    int,                // num
    int,                // lda_add
    int>;               // incx

template<class Func>
struct test_base_TMV : testing::TestWithParam<TestParamTypeTMV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasFillMode_t uplo;
    iclblasOperation_t trans;
    iclblasDiagType_t diag;
    int num;
    int lda;
    int incx;
    data_arr_type x;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = testing::get<0>(GetParam());
        uplo = testing::get<1>(GetParam());
        trans = testing::get<2>(GetParam());
        diag = testing::get<3>(GetParam());
        num = testing::get<4>(GetParam());
        int lda_add = testing::get<5>(GetParam());
        lda = num + lda_add;
        incx = testing::get<6>(GetParam());

        x = get_random_vector<data_type>(num * incx);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypeGBMVV = ::testing::tuple<
    const char*,        // impl_name
    iclblasOperation_t, // trans
    int,                // m
    int,                // n
    int,                // kl
    int,                // ku
    int,                // lda_add
    int,                // incx
    int>;               // incy

template<class Func>
struct test_base_GBMVV : ::testing::TestWithParam<TestParamTypeGBMVV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasOperation_t trans;
    int m;
    int n;
    int kl;
    int ku;
    int lda;
    int incx;
    int incy;

    data_arr_type A;
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
        trans = testing::get<1>(GetParam());
        m = testing::get<2>(GetParam());
        n = testing::get<3>(GetParam());
        kl = testing::get<4>(GetParam());
        ku = testing::get<5>(GetParam());

        int lda_add = testing::get<6>(GetParam());
        lda = kl + ku + 1 + lda_add;
        incx = testing::get<7>(GetParam());
        incy = testing::get<8>(GetParam());

        A = get_random_vector<data_type>(lda * n);
        size_t size_x = trans == ICLBLAS_OP_N ? n * incx : m * incx;
        size_t size_y = trans == ICLBLAS_OP_N ? m * incy : n * incy;
        x = get_random_vector<data_type>(size_x);
        y = get_random_vector<data_type>(size_y);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypeVVS = ::testing::tuple<
    const char*,    //impl_name
    int,            //num
    int,            //incx
    int,            //incy
    float>;         //param

template<class Func>
struct test_base_VVS : testing::TestWithParam<TestParamTypeVVS>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    int num;
    int incx;
    int incy;
    float flag;

    data_arr_type x;
    data_arr_type y;
    data_arr_type param;

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
        flag = ::testing::get<4>(GetParam());
        x = get_random_vector<data_type>(num * incx);
        y = get_random_vector<data_type>(num * incy);

        param = get_random_vector<data_type>(5);
        param[0] = static_cast<data_type>(flag);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypeSMV = ::testing::tuple<
    const char*,        //impl_name
    iclblasDiagType_t,  //uplo
    int,                //n
    int,                //incx
    int>;               //lda_add

template<class Func>
struct test_base_SMV : testing::TestWithParam<TestParamTypeSMV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasDiagType_t uplo;
    int n;
    int incx;
    int lda;

    data_arr_type x;
    data_arr_type A;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = ::testing::get<0>(GetParam());
        uplo = ::testing::get<1>(GetParam());
        n = ::testing::get<2>(GetParam());
        incx = ::testing::get<3>(GetParam());
        int lda_add = ::testing::get<4>(GetParam());
        lda = n + lda_add;

        x = get_random_vector<data_type>(n * incx);
        A = get_random_vector<data_type>(n * lda);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypeSMVV = ::testing::tuple<
    const char*,        //impl_name
    iclblasDiagType_t,  //uplo
    int,                //n
    int,                //incx
    int,                //incy
    int>;               //lda_add

template<class Func>
struct test_base_SMVV : testing::TestWithParam<TestParamTypeSMVV>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasDiagType_t uplo;
    int n;
    int incx;
    int incy;
    int lda;

    data_arr_type x;
    data_arr_type y;
    data_arr_type A;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = ::testing::get<0>(GetParam());
        uplo = ::testing::get<1>(GetParam());
        n = ::testing::get<2>(GetParam());
        incx = ::testing::get<3>(GetParam());
        incy = ::testing::get<4>(GetParam());
        int lda_add = ::testing::get<5>(GetParam());
        lda = n + lda_add;

        x = get_random_vector<data_type>(n * incx);
        y = get_random_vector<data_type>(n * incy);
        A = get_random_vector<data_type>(n * lda);

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

using TestParamTypeGEMMM = ::testing::tuple<
    const char*,        // impl_name
    iclblasOperation_t, // transa
    iclblasOperation_t, // transb
    int,                // m
    int,                // n
    int,                // k
    int,                // lda_add
    int,                // ldb_add
    bool,               // beta_zero
    int>;               // ldc_add

template<class Func>
struct test_base_GEMMM : testing::TestWithParam<TestParamTypeGEMMM>
{
    using data_type = typename func_traits<Func>::data_type;
    using data_arr_type = decltype(get_random_vector<data_type>(0));

    const char* impl_name;
    iclblasOperation_t transa;
    iclblasOperation_t transb;
    int m;
    int n;
    int k;
    data_type alpha;
    int lda;
    int ldb;
    data_type beta;
    int ldc;

    typename Func::params params;
    typename Func::params params_ref;

    virtual void init_values() {}
    virtual typename Func::params get_params() = 0;
    virtual typename Func::params get_params_ref() = 0;

    void SetUp() override
    {
        impl_name = testing::get<0>(GetParam());
        transa = testing::get<1>(GetParam());
        transb = testing::get<2>(GetParam());
        m = testing::get<3>(GetParam());
        n = testing::get<4>(GetParam());
        k = testing::get<5>(GetParam());

        // TODO Use more random approach and disallow 0.
        alpha = get_random_scalar<data_type>();

        const auto lda_add = testing::get<6>(GetParam());
        lda = (transa == ICLBLAS_OP_N ? m : k) + lda_add;

        const auto ldb_add = testing::get<7>(GetParam());
        ldb = (transb == ICLBLAS_OP_N ? k : n) + ldb_add;

        const auto beta_zero = testing::get<8>(GetParam());
        // TODO Use more random approach and disallow 0.
        beta = beta_zero ? static_cast<data_type>(0) : get_random_scalar<data_type>();

        const auto ldc_add = testing::get<9>(GetParam());
        ldc = m + ldc_add;

        init_values();

        params = get_params();
        params_ref = get_params_ref();
        func_traits<Func>::reference(params_ref);
    }
};

}}
