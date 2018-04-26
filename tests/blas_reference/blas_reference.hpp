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

#include <complex>
#include "array_ref.hpp"

int Caxpy_reference(int n, std::complex<float> alpha, array_ref<std::complex<float>> x, int incx, array_ref<std::complex<float>> y, int incy);

int Isamax_reference(int n, array_ref<float> x, int incx);

int Ssyr2_reference(char uplo, int n, float alpha, array_ref<float> x, int incx, array_ref<float> y, int incy, array_ref<float> A, int lda);
