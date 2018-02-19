/* Copyright (c) 2017-2018 Intel Corporation
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef float2 complex_t;

#define I ((complex_t)(0.f, 1.f))

inline float creal(complex_t a) {
    return a.x;
}

inline float cimag(complex_t a) {
    return a.y;
}

inline float cabs(complex_t a) {
    return sqrt(fma(a.x, a.x, a.y * a.y));
}

inline complex_t cmul(complex_t a, complex_t b) {
    return (complex_t)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}

inline complex_t cdiv(complex_t a, complex_t b) {
    float divisor = fma(b.x, b.x, b.y * b.y);
    return (complex_t)( fma(a.x, b.x, a.y * b.y) / divisor, fma(a.y, b.x, -a.x * b.y) / divisor );
}

inline complex_t cmulf(complex_t a, float b) {
    return a * b;
}

inline complex_t cdivf(complex_t a, float b) {
    return a / b;
}

inline complex_t fdivc(float a, complex_t b) {
    float divisor = fma(b.x, b.x, b.y * b.y);
    return (complex_t)( (a * b.x) / divisor, (-a * b.y) / divisor );
}

inline float scabs1(complex_t a) {
    return fabs(creal(a)) + fabs(cimag(a));
}

inline complex_t conjg(complex_t a) {
    return (complex_t)(a.x, -a.y);
}

inline complex_t cneg(complex_t a) {
    return -a;
}

inline complex_t cfmaf(complex_t a, float b, complex_t c) {
    return fma(a, b, c); // It extends 'b' to float2
}

inline float cnorm(complex_t a) {
    return fma(a.x, a.x, a.y * a.y);
}
