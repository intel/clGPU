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

#define ACCESS(a, m, n, N) a[(n)*(N) + (m)]

__kernel void Ctrsm_naive(int side, int uplo, int trans, int diag, int m, int n, complex_t alpha, __global complex_t* a, int lda, __global complex_t* b, int ldb)
{
    bool left = side == 0;
    bool ltriangle = uplo == 1;
    bool ntrans = trans == 0;
    bool conj = trans == 2;
    bool ndiag = diag == 0;

    if (left) {
        if (ltriangle) {
            if (ntrans) {
                for (int i=0; i<m; i++) {
                    for (int j=0; j<n; j++) {
                        complex_t this_x = ACCESS(b, i, j, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=i+1; k<m; k++) {
                            if (conj) {
                                ACCESS(b, k, j, ldb) -= cmul(conjg(ACCESS(a, k, i, lda)), this_x);
                            } else {
                                ACCESS(b, k, j, ldb) -= cmul(ACCESS(a, k, i, lda), this_x);
                            }
                        }
                        ACCESS(b, i, j, ldb) = cmul(this_x, alpha);
                    }
                }
            } else { // ntrans
                for (int i=m-1; i>=0; i--) {
                    for (int j=0; j<n; j++) {
                        complex_t this_x = ACCESS(b, i, j, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=0; k<i; k++) {
                            if (conj) {
                                ACCESS(b, k, j, ldb) -= cmul(conjg(ACCESS(a, i, k, lda)), this_x);
                            } else {
                                ACCESS(b, k, j, ldb) -= cmul(ACCESS(a, i, k, lda), this_x);
                            }
                        }
                        ACCESS(b, i, j, ldb) = cmul(this_x, alpha);
                    }
                }
            }
        } else { // ltriangle
            if (ntrans) {
                for (int i=m-1; i>=0; i--) {
                    for (int j=0; j<n; j++) {
                        complex_t this_x = ACCESS(b, i, j, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=0; k<i; k++) {
                            if (conj) {
                                ACCESS(b, k, j, ldb) -= cmul(conjg(ACCESS(a, k, i, lda)), this_x);
                            } else {
                                ACCESS(b, k, j, ldb) -= cmul(ACCESS(a, k, i, lda), this_x);
                            }
                        }
                        ACCESS(b, i, j, ldb) = cmul(this_x, alpha);
                    }
                }
            } else { // ntrans
                for (int i=0; i<m; i++) {
                    for (int j=0; j<n; j++) {
                        complex_t this_x = ACCESS(b, i, j, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=i+1; k<m; k++) {
                            if(conj) {
                                ACCESS(b, k, j, ldb) -= cmul(conjg(ACCESS(a, i, k, lda)), this_x);
                            } else {
                                ACCESS(b, k, j, ldb) -= cmul(ACCESS(a, i, k, lda), this_x);
                            }
                        }
                        ACCESS(b, i, j, ldb) = cmul(this_x, alpha);
                    }
                }
            }
        }
    } else { // left
        if (ltriangle) {
            if (ntrans) {
                for (int i=n-1; i>=0; i--) {
                    for (int j=0; j<m; j++) {
                        complex_t this_x = ACCESS(b, j, i, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=0; k<i; k++) {
                            if(conj) {
                                ACCESS(b, j, k, ldb) -= cmul(conjg(ACCESS(a, i, k, lda)), this_x);
                            } else {
                                ACCESS(b, j, k, ldb) -= cmul(ACCESS(a, i, k, lda), this_x);
                            }
                        }
                        ACCESS(b, j, i, ldb) = cmul(this_x, alpha);
                    }
                }
            } else { // ntrans
                for (int i=0; i<n; i++) {
                    for (int j=0; j<m; j++) {
                        complex_t this_x = ACCESS(b, j, i, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=i+1; k<n; k++) {
                            if(conj) {
                                ACCESS(b, j, k, ldb) -= cmul(conjg(ACCESS(a, k, i, lda)), this_x);
                            } else {
                                ACCESS(b, j, k, ldb) -= cmul(ACCESS(a, k, i, lda), this_x);
                            }
                        }
                        ACCESS(b, j, i, ldb) = cmul(this_x, alpha);
                    }
                }
            }
        } else { // ltriangle
            if (ntrans) {
                for (int i=0; i<n; i++) {
                    for (int j=0; j<m; j++) {
                        complex_t this_x = ACCESS(b, j, i, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=i+1; k<n; k++) {
                            if(conj) {
                                ACCESS(b, j, k, ldb) -= cmul(conjg(ACCESS(a, i, k, lda)), this_x);
                            } else {
                                ACCESS(b, j, k, ldb) -= cmul(ACCESS(a, i, k, lda), this_x);
                            }
                        }
                        ACCESS(b, j, i, ldb) = cmul(this_x, alpha);
                    }
                }
            } else { // ntrans
                for (int i=n-1; i>=0; i--) {
                    for (int j=0; j<m; j++) {
                        complex_t this_x = ACCESS(b, j, i, ldb);
                        if (ndiag) {
                            if (conj) this_x = cdiv(this_x, conjg(ACCESS(a, i, i, lda)));
                            else this_x = cdiv(this_x, ACCESS(a, i, i, lda));
                        }
                        for (int k=0; k<i; k++) {
                            if(conj) {
                                ACCESS(b, j, k, ldb) -= cmul(conjg(ACCESS(a, k, i, lda)), this_x);
                            } else {
                                ACCESS(b, j, k, ldb) -= cmul(ACCESS(a, k, i, lda), this_x);
                            }
                        }
                        ACCESS(b, j, i, ldb) = cmul(this_x, alpha);
                    }
                }
            }
        }
    }
}
