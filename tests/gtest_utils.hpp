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
#include <complex>
#include <type_traits>
#include "array_ref.hpp"

#define MAX_INDEXED_ERRORS 5

template<typename T>
typename std::enable_if_t<!std::is_floating_point<T>::value, bool>
EqualityHelper(const T& a, const T& b)
{
    return a == b;
}

template<typename T>
typename std::enable_if_t<std::is_floating_point<T>::value, bool>
EqualityHelper(const T& a, const T& b)
{
    using Comparer = ::testing::internal::FloatingPoint<T>;
    return Comparer(a).AlmostEquals(Comparer(b));
}

template<typename T>
bool AlmostEqual(const T& a, const T& b)
{
    return EqualityHelper<T>(a, b);
}

template<typename T>
bool ComplexEqual(const std::complex<T>& a, const std::complex<T>& b)
{
    return EqualityHelper(a.real(), b.real()) && EqualityHelper(a.imag(), b.imag());
}

template<typename T>
bool AlmostEqual(const std::complex<T>& a, const std::complex<T>& b)
{
    return ComplexEqual(a, b);
}

template<typename T>
void PrintTo(const std::complex<T>& a, std::ostream* os)
{
    *os << std::setprecision(std::numeric_limits<T>::digits10 + 2) << a;
}

template<typename T>
::testing::AssertionResult AssertArraysEqual(const char* expected_expr, const char* actual_expr, array_ref<T> expected, array_ref<T> actual)
{
    size_t num_errors = 0;
    size_t err_indexes[MAX_INDEXED_ERRORS];
    const auto N = expected.size() < actual.size() ? expected.size() : actual.size();
    for(size_t i = 0; i < N; ++i)
    {
        if(!AlmostEqual(expected[i], actual[i]))
        {
            if(num_errors < MAX_INDEXED_ERRORS)
            {
                err_indexes[num_errors] = i;
            }
            num_errors++;
        }
    }
    if(num_errors == 0)
    {
        return ::testing::AssertionSuccess();
    }

    const auto indexed_errors = std::min<size_t>(num_errors, MAX_INDEXED_ERRORS);

    auto result = ::testing::AssertionFailure();
    result << expected_expr << " and " << actual_expr << " are different in " << num_errors << " of " << N << " elements.\n";
    result << "First " << indexed_errors << " errors:\n";
    for (size_t i = 0; i < indexed_errors; ++i)
    {
        result << "\t"
               << "index:" << err_indexes[i] << " "
               << expected_expr << ":" << ::testing::PrintToString(expected[err_indexes[i]]) << " "
               << actual_expr   << ":" << ::testing::PrintToString(actual[err_indexes[i]])   << "\n";
    }
    return result;
}

#define EXPECT_COMPLEX_EQ(expected, actual)        EXPECT_PRED2(ComplexEqual<float>, expected, actual)
#define EXPECT_DOUBLE_COMPLEX_EQ(expected, actual) EXPECT_PRED2(ComplexEqual<double>, expected, actual)
#define EXPECT_ARRAYS_EQ(ElemTy, expected, actual) EXPECT_PRED_FORMAT2(AssertArraysEqual<ElemTy>, expected, actual)
#define ASSERT_COMPLEX_EQ(expected, actual)        ASSERT_PRED2(ComplexEqual<float>, expected, actual)
#define ASSERT_DOUBLE_COMPLEX_EQ(expected, actual) ASSERT_PRED2(ComplexEqual<double>, expected, actual)
#define ASSERT_ARRAYS_EQ(ElemTy, expected, actual) ASSERT_PRED_FORMAT2(AssertArraysEqual<ElemTy>, expected, actual)
