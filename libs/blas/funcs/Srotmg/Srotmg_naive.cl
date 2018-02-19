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

__kernel void Srotmg_naive(__global float* d1, __global float* d2, __global float* b1, float bb2, __global float* result)
{
    // d1 >= 0 else zeroe all
    // if fabs(d2*y1^2) >= fabs(d1*x1^2) && d2 < 0 then zeroe all
    // d2*b2 != 0 else identity(-2)

    float dd1 = *d1;
    float dd2 = *d2;
    float bb1 = *b1;

    float p1 = dd1 * bb1;
    float p2 = dd2 * bb2;
    float q1 = p1 * bb1;
    float q2 = p2 * bb2;
    float h11;
    float h12;
    float h21;
    float h22;
    float sflag;

    if (q1 > fabs(q2)) {
        h12 = p2 / p1;
        h21 = -bb2 / bb1;
        float u = 1.f - h12*h21;
        // When u gets cancelled dd1 and dd2 will be INF. To avoid this LAPACK reference implementation skips setting any variables.
        // Hovewer it may lead to undefined behaviour, so we at least set sflag in order to set correct and defined H matrix.
        sflag = 0.f;
        if (u > 0.f) {
            dd1 /= u;
            dd2 /= u;
            bb1 *= u;
        }
    } else {
        h11 = p1/p2;
        h22 = bb1/bb2;
        float u = 1 + h11*h22;
        float v = dd1/u;
        sflag = 1.f;
        dd1 = dd2/u;
        dd2 = v;
        bb1 = bb2*u;
    }
    // Rescaling
    const float gamma = 4096.f;
    const float gamma_sq = gamma * gamma;
    const float gamma_msq = 1.f / gamma_sq;

    if (dd1 != 0 && (fabs(dd1) < gamma_msq || fabs(dd1) > gamma_sq)) {
        // Compared to LAPACK reference implementation, this if has been moved outside of while loop to not reset h12 and h21 at every iteration.
        if (sflag == 0.f) {
            h11 = 1.f;
            h22 = 1.f;
            sflag = -1.f;
        } else {
            h12 = 1.f;
            h21 = -1.f;
            sflag = -1.f;
        }
        while (fabs(dd1) < gamma_msq || fabs(dd1) > gamma_sq) {
            if (fabs(dd1) < gamma_msq) {
                dd1 *= gamma_sq;
                bb1 /= gamma;
                h11 /= gamma;
                h12 /= gamma;
            } else {
                dd1 /= gamma_sq;
                bb1 *= gamma;
                h11 *= gamma;
                h12 *= gamma;
            }
        }
    }
    // --------------------------------------------
    if (dd2 != 0 && (fabs(dd2) < gamma_msq || fabs(dd2) > gamma_sq)) {
        // Exactly like in rescaling of dd1 if loop has been moved, and also in else branch there is checking for sflag value,
        // in case it has already been changed when rescaling dd1.
        if (sflag == 0.f) {
            h11 = 1.f;
            h22 = 1.f;
            sflag = -1.f;
        } else if (sflag == 1.f) {
            h12 = 1.f;
            h21 = -1.f;
            sflag = -1.f;
        }
        while (fabs(dd2) < gamma_msq || fabs(dd2) > gamma_sq) {
            if (fabs(dd2) < gamma_msq) {
                dd2 *= gamma_sq;
                h21 /= gamma;
                h22 /= gamma;
            } else {
                dd2 /= gamma_sq;
                h21 *= gamma;
                h22 *= gamma;
            }
        }
    }

    result[0] = sflag;
    if (sflag == -1.f) {
        result[1] = h11;
        result[2] = h21;
        result[3] = h12;
        result[4] = h22;
    } else if (sflag == 0.f) {
        result[2] = h21;
        result[3] = h12;
    } else { // sflag == -1.f
        result[1] = h11;
        result[4] = h22;
    }

    d1[0] = dd1;
    d2[0] = dd2;
    b1[0] = bb1;
}
