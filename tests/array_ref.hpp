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

#include <vector>
#include <cassert>
#include <type_traits>

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
#   include <iterator>
#endif

template<typename T>
class array_ref
{
public:
    constexpr array_ref(): _data(nullptr), _size(0) {}

    template<size_t N>
    array_ref(T (&arr)[N]): _data(arr), _size(N) {}

    array_ref(std::vector<T>& vec): _data(vec.data()), _size(vec.size()) {}

    array_ref(T* ptr, size_t n): _data(ptr), _size(n) {}

    array_ref(const array_ref& other) = default;

    array_ref& operator=(const array_ref& other) = default;

    template<typename U, typename = std::enable_if_t<std::is_standard_layout<T>::value && std::is_standard_layout<U>::value>>
    explicit array_ref(array_ref<U>& other) : _data(reinterpret_cast<T*>(other.data())), _size(other.size() * sizeof(U) / sizeof(T)) {}

    template<typename U, typename = std::enable_if_t<std::is_standard_layout<T>::value && std::is_standard_layout<U>::value>>
    array_ref& operator=(array_ref<U>& other)
    {
        _data = other.data();
        _size = other.size();
        return *this;
    }

    constexpr size_t size() const { return _size; }

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
    typedef stdext::checked_array_iterator<T*> iterator;
    typedef stdext::checked_array_iterator<const T*> const_iterator;

    iterator begin() { return stdext::make_checked_array_iterator(_data, size()); }
    iterator end() { return stdext::make_checked_array_iterator(_data, size(), size()); }

    const_iterator begin() const { return stdext::make_checked_array_iterator(_data, size()); }
    const_iterator end() const { return stdext::make_checked_array_iterator(_data, size(), size()); }
    const_iterator cbegin() const { return stdext::make_checked_array_iterator(_data, size()); }
    const_iterator cend() const { return stdext::make_checked_array_iterator(_data, size(), size()); }
#else
    typedef T* iterator;
    typedef const T* const_iterator;

    iterator begin() { return _data; }
    iterator end() { return _data + size(); }

    const_iterator begin() const { return _data; }
    const_iterator end() const { return _data + size(); }
    const_iterator cbegin() const { return _data; }
    const_iterator cend() const { return _data + size(); }
#endif

    T& operator[](size_t idx) const
    {
        assert(idx < _size);
        return _data[idx];
    }

    T* data() { return _data; }
    constexpr T* data() const { return _data; }

private:
    T* _data;
    size_t _size;
};
