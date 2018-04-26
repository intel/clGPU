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

#pragma once

/*****************************************************************************/
// exporting symbols from dynamic library
#ifdef EXPORT_ICLBLAS_SYMBOLS
#   if defined(_MSC_VER)
//  Microsoft
#      define ICLBLAS_API __declspec(dllexport)
#   elif defined(__GNUC__)
//  GCC
#      define ICLBLAS_API __attribute__((visibility("default")))
#   else
#      define ICLBLAS_API
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#else //import dll
#   if defined(_MSC_VER)
//  Microsoft
#      define ICLBLAS_API __declspec(dllimport)
#   elif defined(__GNUC__)
//  GCC
#      define ICLBLAS_API
#   else
#      define ICLBLAS_API
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__ICC)
#   define ICLBLAS_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#   define ICLBLAS_ALWAYS_INLINE __attribute__((always_inline))
#else
#   define ICLBLAS_ALWAYS_INLINE
#   pragma message ("warning Unknown always-inline function attribute.")
#endif

/*****************************************************************************/

/*!
 * @file iclBLAS.h
 * @brief This file contains all of the BLAS related (public) interfaces and objects.
 */

/*
 * or more obtusely but canonical reference here: http://www.netlib.org/blas/
 * Inside Tensorflow is another view of blas.h https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/blas.h
 *
 * OCL BLAS better name than genBLAS because this interface should work for all opencl implementations (though some kernels may not be available
 * if they use special gen extensions like subgroups
*/

/*!
 * @brief Complex type definition
 */
#ifdef __cplusplus
#   include <complex>
    typedef std::complex<float> oclComplex_t;
    inline float Creal(const oclComplex_t& a)       { return a.real(); }
    inline float Cimag(const oclComplex_t& a)       { return a.imag(); }
    inline void  Csetreal(oclComplex_t* a, float r) { a->real(r); }
    inline void  Csetimag(oclComplex_t* a, float i) { a->imag(i); }
#else
    typedef struct _oclComplex_t
    {
        float val[2]; /*!< real (val[0]) and imaginary (val[1]) parts of complex number */
    } oclComplex_t;
    ICLBLAS_ALWAYS_INLINE float Creal(struct _oclComplex_t a)              { return a.val[0]; }
    ICLBLAS_ALWAYS_INLINE float Cimag(struct _oclComplex_t a)              { return a.val[1]; }
    ICLBLAS_ALWAYS_INLINE void  Csetreal(struct _oclComplex_t* a, float r) { a->val[0] = r; }
    ICLBLAS_ALWAYS_INLINE void  Csetimag(struct _oclComplex_t* a, float i) { a->val[1] = i; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************************/
/*!
* @addtogroup datatypes Datatypes
* @{
*/

/*!
 * @brief Operations status codes
 *
 * A more detailed enum description.
*/
typedef enum {
    ICLBLAS_STATUS_SUCCESS = 0,
    ICLBLAS_STATUS_NOT_INITIALIZED = 1,
    ICLBLAS_STATUS_ALLOC_FAILED = 3,
    ICLBLAS_STATUS_INVALID_VALUE = 7,
    ICLBLAS_STATUS_ARCH_MISMATCH = 8,
    ICLBLAS_STATUS_MAPPING_ERROR = 11,
    ICLBLAS_STATUS_EXECUTION_FAILED = 13,
    ICLBLAS_STATUS_INTERNAL_ERROR = 14,
    ICLBLAS_STATUS_NOT_SUPPORTED = 15,
    ICLBLAS_STATUS_LICENSE_ERROR = 16,
    ICLBLAS_STATUS_ERROR,

} iclblasStatus_t;


/*!
 * @brief Opaque structure holding library context
 */
typedef struct iclblasContext *iclblasHandle_t;

/*!
 * @brief Indicates operation to be performed.
 */
typedef enum {
    ICLBLAS_OP_N = 0, /*!< the non-transpose operation is selected */
    ICLBLAS_OP_T = 1, /*!< the transpose operation is selected */
    ICLBLAS_OP_C = 2  /*!< the conjugate transpose operation is selected */
} iclblasOperation_t;

/*!
 * @brief Indicates which part (lower or upper) of matrix is filled.
 */
typedef enum {
    ICLBLAS_FILL_MODE_UPPER = 0, /*!< the upper part of matrix is filled */
    ICLBLAS_FILL_MODE_LOWER = 1  /*!< the lower part of matrix is filled */
} iclblasFillMode_t;

/*!
 * @brief Indicates whether the main diagonal of matrix is unity.
 *
 * In case of ::ICLBLAS_DIAG_UNIT the main diagonal is assumed to contain only unit elements and is not referenced.
 */
typedef enum {
    ICLBLAS_DIAG_NON_UNIT = 0, /*!< the main diagonal contains non-unit elements */
    ICLBLAS_DIAG_UNIT     = 1 /*!< the main diagonal contains only unit elements */
} iclblasDiagType_t;

/*!
 * @brief Indicates on which side (left or right) the matrix in the equation solved by a function.
 */
typedef enum {
    ICLBLAS_SIDE_LEFT = 0, /*!< the matrix is on the left side in equation */
    ICLBLAS_SIDE_RIGHT = 1 /*!< the matrix is on the right side in equation */
} iclblasSideMode_t;
/*! @} */

/*****************************************************************************/
/*!
* @addtogroup utilities Utility functions
* @{
*/

/*!
 * @brief Create library context
 *
 * @param handle pointer to store context handle
 */
ICLBLAS_API iclblasStatus_t iclblasCreate(iclblasHandle_t* handle);

/*!
 * @brief Destroy library context
 *
 * @param handle handle to the library context to be destroyed
 */
ICLBLAS_API iclblasStatus_t iclblasDestroy(iclblasHandle_t handle);
/*! @} */

/*****************************************************************************/
/*!
* @addtogroup BLAS_L1_S BLAS Level 1 Single
* @{
*/

 /*!
 * @brief Copy the elements from the vector x to the vector y
 *
 * @code
 * y = x
 * @endcode
 * Where @b x and @b y are @b n element vectors
 *
 * @param[in] handle handle to the library context
 * @param[in] n     number of elements in @b x and @b y
 * @param[in] x     vector of size at least @b n * @b incx
 * @param[in] incx  stride between elements in @b x; should be at least 1
 * @param[in,out] y vector of size at least @b n * @b incy
 * @param[in] incy  stride between elements in @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasScopy(iclblasHandle_t handle, int n, float *x, int incx, float *y, int incy);

/*!
* @brief Multiply the vector by the scalar
*
* @code
* x = alpha * x
* @endcode
* Where @b x is @b n element vector
*
* @param[in] handle handle to the library context
* @param[in] n      number of elements in @b x
* @param[in] alpha     scalar used in multiplication
* @param[in,out] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasSscal(iclblasHandle_t handle, int n, const float* alpha, float *x, int incx);

/*!
* @brief Multiply the vector x by the scalar and add it to the vector y
*
* @code
* y = alpha * x + y
* @endcode
* Where @b x and @b y are @b n element vectors
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in] alpha     scalar used in multiplication
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in, out] y     vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasSaxpy(iclblasHandle_t handle, int n, const float* alpha, float *x, int incx, float *y, int incy);

/*!
* @brief Computes the Euclidean norm of the vector x
*
* Where @b x is @b n element vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of computed elements
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  Computed Euclidean norm
*/
ICLBLAS_API iclblasStatus_t iclblasSnrm2(iclblasHandle_t handle, int n, float *x, int incx, float *result);

/*!
* @brief Constructs the modified Givens transformation
*
* @param[in] handle handle to the library context
* @param[in,out] d1     scalar result of the computation
* @param[in,out] d2     scalar result of the computation
* @param[in,out] x1     scalar result of the computation
* @param[in] y1         scalar
* @param[out] params    vector of 5 elements, param[0] contain the flag, param[1-4] contain the matrix H
*/
ICLBLAS_API iclblasStatus_t iclblasSrotmg(iclblasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float* params);

/*!
* @brief Computes the first index of the highest value in vector x
*
* Where @b x is n element vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  calculated index
*/
ICLBLAS_API iclblasStatus_t iclblasIsamax(iclblasHandle_t handle, int n, float* x, int incx, int* result);

/*!
* @brief Computes the first index of the lowest value in vector x
*
* Where @b x is n element vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  calculated index
*/
ICLBLAS_API iclblasStatus_t iclblasIsamin(iclblasHandle_t handle, int n, float* x, int incx, int* result);

/*!
* @brief Interchanges two vectors x and y
*
* @code
* y = x, x = y
* @endcode
* Where @b x and @b y are @b n element vectors
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in,out] x vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasSswap(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy);

/*!
* @brief Applies Givens rotation matrix
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in,out] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y     vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[in] c  cosine of the rotation matrix
* @param[in] s  sine of the rotation matrix
*/
ICLBLAS_API iclblasStatus_t iclblasSrot(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, float c, float s);

/*!
* @brief Applies modified Givens rotation matrix
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in,out] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y     vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[in] param  vector of 5 elements, param[0] contain the flag, param[1-4] contain the matrix H
*/
ICLBLAS_API iclblasStatus_t iclblasSrotm(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, float* param);

/*!
* @brief Computes the sum of the absolute values from vector x
*
* Where @b x is n elements vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  result of the calculation
*/
ICLBLAS_API iclblasStatus_t iclblasSasum(iclblasHandle_t handle, int n, float* x, int incx, float* result);

/*!
* @brief Creates the Givens rotation matrix
*
* @param[in] handle handle to the library context
* @param[in,out] a     scalar, later overwritten with r
* @param[in,out] b     scalar, later overwritten with z
* @param[out] c        Contains the parameter c associated with the Givens rotation.
* @param[out] s        Contains the parameter s associated with the Givens rotation.
*/
ICLBLAS_API iclblasStatus_t iclblasSrotg(iclblasHandle_t handle, float* a, float* b, float* c, float* s);

/*!
* @brief Computes the dot product from vector x and vector y
*
* Where @b x and @b y are @b n elements vectors
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in] y     vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[out] result result of the calculation
*/
ICLBLAS_API iclblasStatus_t iclblasSdot(iclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, float* result);
/*! @} */

/*****************************************************************************/
/*!
* @addtogroup BLAS_L1_C BLAS Level 1 Complex
* @{
*/

 /*!
 * @brief Copy the elements from the vector x to the vector y
 *
 * @code
 * y = x
 * @endcode
 * Where @b x and @b y are @b complex n element vectors
 *
 * @param[in] handle handle to the library context
 * @param[in] n     number of elements in @b x and @b y
 * @param[in] x     vector of size at least @b n * @b incx
 * @param[in] incx  stride between elements in @b x; should be at least 1
 * @param[in,out] y vector of size at least @b n * @b incy
 * @param[in] incy  stride between elements in @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCcopy(iclblasHandle_t handle, int n, oclComplex_t *x, int incx, oclComplex_t *y, int incy);

/*!
* @brief Multiply the complex vector by the scalar
*
* @code
* x = alpha * x
* @endcode
* Where @b x is @b complex n element vector
*
* @param[in] handle handle to the library context
* @param[in] n      number of elements in @b x
* @param[in] alpha     complex scalar used in multiplication
* @param[in,out] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasCscal(iclblasHandle_t handle, int n, const oclComplex_t* alpha, oclComplex_t *x, int incx);

/*!
* @brief Multiply the complex vector by the scalar
*
* @code
* x = alpha * x
* @endcode
* Where @b x is @b complex n element vector
*
* @param[in] handle handle to the library context
* @param[in] n      number of elements in @b x
* @param[in] alpha     scalar used in multiplication
* @param[in,out] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasCsscal(iclblasHandle_t handle, int n, const float* alpha, oclComplex_t *x, int incx);

/*!
* @brief Interchanges two complex vectors x and y
*
* @code
* y = x, x = y
* @endcode
* Where @b x and @b y are @b complex n element vectors
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in,out] x complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y complex vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasCswap(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy);

/*!
* @brief Computes the dot product from complex vector x and vector y
*
* @code
* result = x^T * y
* @endcode
* Where @b x and @b y are @b complex n element vectors
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in] y     complex vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[out] result  calculated dot product
*/
ICLBLAS_API iclblasStatus_t iclblasCdotu(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* result);

/*!
* @brief Computes the Euclidean norm of the complex vector x
*
* Where @b x is @b n element vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of computed elements
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  Computed Euclidean norm
*/
ICLBLAS_API iclblasStatus_t iclblasScnrm2(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, float* result);

/*!
* @brief Computes the first index of the highest magnitude value in complex vector x
*
* Where @b x is complex n element vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x
* @param[in] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  calculated index
*/
ICLBLAS_API iclblasStatus_t iclblasIcamax(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, int* result);

/*!
* @brief Computes the first index of the lowest magnitude value in complex vector x
*
* Where @b x is complex n element vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x
* @param[in] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] result  calculated index
*/
ICLBLAS_API iclblasStatus_t iclblasIcamin(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, int* result);

/*!
* @brief Multiply the complex vector x by the complex scalar and add it to the complex vector y
*
* @code
* y = alpha * x + y
* @endcode
* Where @b x and @b y are @b complex n element vectors
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in] alpha     complex scalar used in multiplication
* @param[in] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y     complex vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
*/
ICLBLAS_API iclblasStatus_t iclblasCaxpy(iclblasHandle_t handle, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy);

/*!
* @brief Creates the Givens rotation matrix
*
* @param[in] handle handle to the library context
* @param[in,out] a     complex scalar, later overwritten with r
* @param[in,out] b     complex scalar, later overwritten with z
* @param[out] c     Contains the parameter c associated with the Givens rotation.
* @param[out] s     complex Contains the parameter s associated with the Givens rotation.
*/
ICLBLAS_API iclblasStatus_t iclblasCrotg(iclblasHandle_t handle, oclComplex_t* a, oclComplex_t* b, float* c, oclComplex_t* s);

/*!
* @brief Computes the dot product from complex vector x and vector y
*
* @code
* result = x^H * y
* @endcode
* Where @b x and @b y are @b complex n element vectors;
* and `H` - complex conjugation on elements of @b x
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in] y complex vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[out] result  calculated dot product
*/
ICLBLAS_API iclblasStatus_t iclblasCdotc(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* result);

/*!
* @brief Applies Givens rotation matrix
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in,out] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y     complex vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[in] c  cosine of the rotation matrix
* @param[in] s  sine of the rotation matrix
*/
ICLBLAS_API iclblasStatus_t iclblasCrot(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, const float* c, const oclComplex_t* s);

/*!
* @brief Applies Givens rotation matrix
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x and @b y
* @param[in,out] x     complex vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[in,out] y     complex vector of size at least @b n * @b incy
* @param[in] incy  stride between elements in @b y; should be at least 1
* @param[in] c  cosine of the rotation matrix
* @param[in] s  sine of the rotation matrix
*/
ICLBLAS_API iclblasStatus_t iclblasCsrot(iclblasHandle_t handle, int n, oclComplex_t* x, int incx, oclComplex_t* y, int incy, const float* c, const float* s);

/*!
* @brief Computes the sum of the absolute values from vector x
*
* Where @b x is n elements vector
*
* @param[in] handle handle to the library context
* @param[in] n     number of elements in @b x
* @param[in] x     vector of size at least @b n * @b incx
* @param[in] incx  stride between elements in @b x; should be at least 1
* @param[out] result  result of the calculation
*/
ICLBLAS_API iclblasStatus_t iclblasScasum(iclblasHandle_t handle, int n, oclComplex_t *x, int incx, float* result);
/*! @} */

/*****************************************************************************/
/*!
* @addtogroup BLAS_L2_S BLAS Level 2 Single
* @{
*/

/*!
 * @brief Solves triangular linear system with single right-hand side
 *
 * @code
 * op(A) * x = b
 * @endcode
 * Where @b b and @b x are @b n element vectors and @b A is @b n by @b n, unit or non-unit, upper or lower triangular matrix.
 *
 * Equation to be solved is specified by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * On exit solution @b x overwrites right-hand side vector @b b.
 *
 * No test for singularity or near-singularity is included in this routine.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if matrix @b A is an upper or lower triangular
 * @param[in] trans     indicates equation to be solved as operation on @b A
 * @param[in] diag      indicates if matrix @b A is unit or non-unit triangular
 * @param[in] n         specifies order of matrix @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       first dimension of @b A; should be at least max(1, @b n)
 * @param[in,out] x     array of size at least @b n * @b incx; on entry stores vector @b b, overwritten by vector @b x on exit
 * @param[in] incx      stride between elements in @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasStrsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* A, int lda, float* x, int incx);

/*!
 * @brief Solves triangular banded linear system with single right-hand side
 *
 * @code
 * op(A) * x = b
 * @endcode
 * Where @b b and @b x are @b n element vectors and @b A is @b n by @b n, unit or non-unit, upper or lower triangular banded matrix with @b k sub- or super-diagonals.
 *
 * Operation on matrix @b A when solving is specified by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(k + i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the top left @b k x @b k triangle) are not referenced.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the bottom right @b k x @b k triangle) are not referenced.
 *
 * On exit solution @b x overwrites right-hand side vector @b b.
 *
 * No test for singularity or near-singularity is included in this routine.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if matrix @b A is an upper or lower triangular
 * @param[in] trans     indicates equation to be solved as operation on @b A
 * @param[in] diag      indicates if matrix @b A is unit or non-unit triangular
 * @param[in] n         specifies order of matrix @b A; should be at least 0
 * @param[in] k         number of sub- or super-diagonals of matrix @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       first dimension of @b A; should be at least @b k + 1
 * @param[in,out] x     array of size at least @b n * @b incx; on entry stores vector @b b, overwritten by vector @b x on exit
 * @param[in] incx      stride between elements in @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasStbsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, float* A, int lda, float* x, int incx);

/*!
 * @brief Solves packed triangular linear system with single right-hand side
 *
 * @code
 * op(A) * x = b
 * @endcode
 * Where @b b and @b x are @b n element vectors and @b A is @b n by @b n, unit or non-unit, upper or lower triangular matrix stored in packed format.
 *
 * Equation to be solved is specified by value of @b trans as operation on @b A.
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * On exit solution @b x overwrites right-hand side vector @b b.
 *
 * No test for singularity or near-singularity is included in this routine.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if matrix @b A is an upper or lower triangular
 * @param[in] trans     indicates equation to be solved as operation on @b A
 * @param[in] diag      indicates if matrix @b A is unit or non-unit triangular
 * @param[in] n         specifies order of matrix @b A; should be at least 0
 * @param[in] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 * @param[in,out] x     array of size at least @b n * @b incx; on entry stores vector @b b, overwritten by vector @b x on exit
 * @param[in] incx      stride between elements in @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasStpsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* AP, float* x, int incx);

/*!
 * @brief Performs general matrix rank 1 update
 *
 * @code
 * A = alpha * x * y ^ T + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b m element vector, @b y is @b n elements vector and @b A is @b m by @b n matrix.
 *
 * @param[in] handle    handle to the library context
 * @param[in] m         number of rows in matrix @b A; should be at least 0
 * @param[in] n         number of columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b m * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b m)
 */
ICLBLAS_API iclblasStatus_t iclblasSger(iclblasHandle_t handle, int m, int n, const float* alpha, float* x, int incx, float* y, int incy, float* A, int lda);

/*!
 * @brief Performs symmetrix matrix rank 2 update
 *
 * @code
 * A = alpha * ( x * y ^ T + y * x ^ T ) + A
 * @endcode
 * Where @b alpha is scalar, @b x and @b y are @b n element vectors, and @b A is @b n by @b n symmetric matrix, stored in upper or lower mode.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 */
ICLBLAS_API iclblasStatus_t iclblasSsyr2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *x, int incx, float* y, int incy, float* A, int lda);

/*!
 * @brief Performs symmetrix matrix rank 1 update
 *
 * @code
 * A = alpha * x * x ^ T + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b n element vector and @b A is @b n by @b n symmetric matrix, stored in upper or lower mode.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 */
ICLBLAS_API iclblasStatus_t iclblasSsyr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float*x, int incx, float* A, int lda);

/*!
 * @brief Performs triangular matrix by vector multiplication
 *
 * @code
 * x = op(A) * x
 * @endcode
 * Where @b x is @b n element vector, @b A is @b n by @b n, upper or lower, unit or non-unit triangular matrix, and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if @b A is upper or lower triangular
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] diag      indicates if @b A is unit or non-unit triangular
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 * @param[in,out] x     vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasStrmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* A, int lda, float* x, int incx);

/*!
 * @brief Performs triangular banded matrix by vector multiplication
 *
 * @code
 * x = op(A) * x
 * @endcode
 * Where @b x is @b n element vector, @b A is @b n by @b n, upper or lower, unit or non-unit triangular banded matrix with @b k sub- or super-diagonals,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(k + i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the top left @b k x @b k triangle) are not referenced.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the bottom right @b k x @b k triangle) are not referenced.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if @b A is upper or lower triangular
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] diag      indicates if @b A is unit or non-unit triangular
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] k         number of sub- or super-diagonals of matrix @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least @b k + 1
 * @param[in,out] x     vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasStbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, float* A, int lda, float* x, int incx);

/*!
 * @brief Performs general banded matrix by vector multiplication
 *
 * @code
 * y = alpha * op(A) * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are vectors, @b A is @b m by @b n banded matrix with @b kl subdiagonals and @b ku superdiagonals,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * Matrix @b A is stored coulumn by column with element `A(i, j)` at location `A(ku + i - j, j)` in memory.
 * The elements that do not correspond to elements in banded matrix (top left @b ku x @b ku and bottom right @b kl x @b kl triangles) are not referenced.
 *
 * @param[in] handle    handle to the library context
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] m         number of rows in matrix @b A; should be at least 0
 * @param[in] n         number of columns in matrix @b A; should be at least 0
 * @param[in] kl        number of subdiagonals in matrix @b A; should be at least 0
 * @param[in] ku        number of superdiagonals in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size [@b lda x @b n]
 * @param[in] lda       leading dimension of @b A; should be at least @b kl + @b ku + 1
 * @param[in] x         vector of size at least @b n * @b incx if `trans == ICLBLAS_OP_N` and @b m * @b incx otherwise
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] beta      scalar used for multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b m * @b incy if `trans == ICLBLAS_OP_N` and @b n * @b incy otherwise
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasSgbmv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, float* A, int lda, float* x, int incx, const float* beta, float* y, int incy);

/*!
 * @brief Performs packed triangular matrix by vector multiplication
 *
 * @code
 * x = op(A) * x
 * @endcode
 * Where @b x is @b n element vector, @b A is @b n by @b n, unit or non-unit, upper or lower triangular matrix stored in packed format,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if @b A is upper or lower triangular
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] diag      indicates if @b A is unit or non-unit triangular
 * @param[in] n         number of rows and columns of matrix @b A; should be at least 0
 * @param[in] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 * @param[in,out] x     vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasStpmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, float* AP, float* x, int incx);

/*!
 * @brief Performs symmetric banded matrix by vector multiplication
 *
 * @code
 * y = alpha * A * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are @b n element vectors, @b A is @b n by @b n symmetric banded matrix with @b k subdiagonals and superdiagonals.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(k + i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the top left @b k x @b k triangle) are not referenced.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the bottom right @b k x @b k triangle) are not referenced.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns of @b A; should be at least 0
 * @param[in] k         number of sub- and super-diagonals of matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least @b k + 1
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasSsbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, char n, char k, const float* alpha, float* A, int lda, float* x, int incx, const float* beta, float* y, int incy);

/*!
 * @brief Performs packed symmetric matrix by vector multiplication
 *
 * @code
 * y = alpha * A * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are @b n element vectors, @b A is @b n by @b n symmetric matrix stored in packed format.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b AP
 * @param[in] n         number of rows and columns of matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasSspmv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float* AP, float* x, int incx, const float* beta, float* y, int incy);

/*!
 * @brief Performs packed symmetric matrix rank 2 update
 *
 * @code
 * A = alpha * ( x * y ^ T + y * x ^ T ) + A
 * @endcode
 * Where @b alpha is a scalar, @b x and @b y are @b n element vectors and @b A is @b n by @b n symmetric matrix stored in packed format.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b AP
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements of @b y; should be at least 1
 * @param[in,out] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 */
ICLBLAS_API iclblasStatus_t iclblasSspr2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *x, int incx, float* y, int incy, float* AP);

/*!
 * @brief Performs packed symmetric matrix rank 1 update
 *
 * @code
 * A = alpha * x * x ^ T + A
 * @endcode
 * Where @b alpha is a scalar, @b x is @b n element vector and @b A is @b n by @b n symmetric matrix stored in packed format.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b AP
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 * @param[in,out] AP    array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 */
ICLBLAS_API iclblasStatus_t iclblasSspr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *x, int incx, float* AP);

/*!
 * @brief Performs symmetric matrix by vector multiplication
 *
 * @code
 * y = alpha * A * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are @b n element vectors and @b A is the @b n x @b n symmetric matrix in lower or upper mode.
 *
 * @param[in] handle handle to the library context
 * @param[in] uplo  indicates if lower or upper part of matrix is stored in @b A
 * @param[in] n     number of rows and columns in @b A; should be at least 0
 * @param[in] alpha scalar used in multiplication
 * @param[in] A     array of size [@b lda x @b n]
 * @param[in] lda   first dimension of @b A; must be at least max(1, @b n)
 * @param[in] x     vector of size at least @b n * @b incx
 * @param[in] incx  stride between elements in @b x; should be at least 1
 * @param[in] beta  scalar used in multiplication; if `beta == 0`, y does not have to be initialized
 * @param[in,out] y vector of size at least @b n * @b incy
 * @param[in] incy  stride between elements in @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasSsymv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, float *A, int lda, float *x, int incx, const float* beta, float *y, int incy);

/*!
 * @brief Performs general matrix by vector multiplication
 *
 * @code
 * y = alpha * op(A) * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are vectors, @b A is @b m by @b n matrix,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ T
 * @endcode
 *
 * @param[in] handle    handle to the library context
 * @param[in] trans     indicates operation used in multiplication
 * @param[in] m         number of rows of matrix @b A; should be at least 0
 * @param[in] n         number of columns of matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size at least [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of array @b A; should be at least max(1, @b m)
 * @param[in] x         vector of size at least @b n * @b incx if `trans == ICLBLAS_OP_N` and @b m * @b incx otherwise
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b m * @b incy if `trans == ICLBLAS_OP_N` and @b n * @b incy otherwise
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasSgemv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, const float* alpha, float *A, int lda, float *x, int incx, const float* beta, float *y, int incy);
/*! @} */

/*****************************************************************************/
/*! @addtogroup BLAS_L2_C BLAS Level 2 Complex
* @{
 */

 /*!
 * @brief Performs general banded matrix by vector multiplication
 *
 * @code
 * y = alpha * op(A) * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are vectors, @b A is @b m by @b n banded matrix with @b kl subdiagonals and @b ku superdiagonals,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * Matrix @b A is stored coulumn by column with element `A(i, j)` at location `A(ku + i - j, j)` in memory.
 * The elements that do not correspond to elements in banded matrix (top left @b ku x @b ku and bottom right @b kl x @b kl triangles) are not referenced.
 *
 * @param[in] handle    handle to the library context
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] m         number of rows in matrix @b A; should be at least 0
 * @param[in] n         number of columns in matrix @b A; should be at least 0
 * @param[in] kl        number of subdiagonals in matrix @b A; should be at least 0
 * @param[in] ku        number of superdiagonals in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least @b kl + @b ku + 1
 * @param[in] x         vector of size at least @b n * @b incx if `trans == ICLBLAS_OP_N` and @b m * @b incx otherwise
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] beta      scalar used for multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b m * @b incy if `trans == ICLBLAS_OP_N` and @b n * @b incy otherwise
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCgbmv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, int kl, int ku, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy);

/*!
 * @brief Performs general matrix rank 1 update
 *
 * @code
 * A = alpha * x * y ^ T + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b m element vector, @b y is @b n elements vector, and @b A is @b m by @b n matrix.
 *
 * @param[in] handle    handle to the library context
 * @param[in] m         number of rows in matrix @b A; should be at least 0
 * @param[in] n         number of columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b m * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b m)
 */
ICLBLAS_API iclblasStatus_t iclblasCgeru(iclblasHandle_t handle, int m, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* A, int lda);

/*!
 * @brief Performs general matrix rank 1 update
 *
 * @code
 * A = alpha * x * y ^ H + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b m element vector, @b y is @b n elements vector, and @b A is @b m by @b n matrix.
 *
 * @param[in] handle    handle to the library context
 * @param[in] m         number of rows in matrix @b A; should be at least 0
 * @param[in] n         number of columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b m * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b m)
 */
ICLBLAS_API iclblasStatus_t iclblasCgerc(iclblasHandle_t handle, int m, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* A, int lda);

/*!
 * @brief Performs general matrix by vector multiplication
 *
 * @code
 * y = alpha * op(A) * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are vectors, @b A is @b m by @b n matrix,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * @param[in] handle    handle to the library context
 * @param[in] trans     indicates operation used in multiplication
 * @param[in] m         number of rows of matrix @b A; should be at least 0
 * @param[in] n         number of columns of matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size at least [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of array @b A; should be at least max(1, @b m)
 * @param[in] x         vector of size at least @b n * @b incx if `trans == ICLBLAS_OP_N` and @b m * @b incx otherwise
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b m * @b incy if `trans == ICLBLAS_OP_N` and @b n * @b incy otherwise
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCgemv(iclblasHandle_t handle, iclblasOperation_t trans, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy);

/*!
 * @brief Performs Hermitian matrix rank 1 update
 *
 * @code
 * A = alpha * x * x ^ H + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b n element vector, and @b A is @b n by @b n Hermitian matrix stored in upper or lower mode.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 */
ICLBLAS_API iclblasStatus_t iclblasCher(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, oclComplex_t* x, int incx, oclComplex_t* A, int lda);

/*!
 * @brief Performs Hermitian matrix by vector multiplication
 *
 * @code
 * y = alpha * A * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are @b n element vectors, and @b A is @b n by @b n Hermitian matrix stored in upper or lower mode.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0` then @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasChemv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy);

/*!
 * @brief Performs triangular matrix by vector multiplication
 *
 * @code
 * x = op(A) * x
 * @endcode
 * Where @b x is @b n element vector, @b A is @b n by @b n, upper or lower, unit or non-unit triangular matrix, and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if @b A is upper or lower triangular
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] diag      indicates if @b A is unit or non-unit triangular
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 * @param[in,out] x     vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCtrmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* A, int lda, oclComplex_t* x, int incx);

/*!
 * @brief Performs Hermitian matrix rank 2 update
 *
 * @code
 * A = alpha * x * y ^ H + conj(alpha) * y * x ^ H + A
 * @endcode
 * Where @b alpha is scalar, @b x and y are @b n element vectors, and @b A is @b n by @b n Hermitian matrix stored in upper or lower mode.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 */
ICLBLAS_API iclblasStatus_t iclblasCher2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* A, int lda);

/*!
 * @brief Performs packed Hermitian matrix by vector multiplication
 *
 * @code
 * y = alpha * A * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are @b n element vectors, @b A is @b n by @b n Hermitian matrix in upper or lower mode, stored in packed format.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b AP
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0` then @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasChpmv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* AP, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy);

/*!
 * @brief Performs packed Hermitian matrix rank 1 update
 *
 * @code
 * A = alpha * x * x ^ H + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b n element vector, and @b A is @b n by @b n Hermitian matrix in upper or lower mode, stored in packed format.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower triangle is stored in @b AP
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in,out] AP    array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 */
ICLBLAS_API iclblasStatus_t iclblasChpr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const float* alpha, oclComplex_t*x, int incx, oclComplex_t* AP);

/*!
 * @brief Performs packed Hermitian matrix rank 2 update
 *
 * @code
 * A = alpha * x * y ^ H + conj(alpha) * y * x ^ H + A
 * @endcode
 * Where @b alpha is scalar, @b x and y are @b n element vectors, and @b A is @b n by @b n Hermitian matrix in upper or lower mode, stored in packed format.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b AP
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in] y         vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements in @b y; should be at least 1
 * @param[in,out] AP    array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 */
ICLBLAS_API iclblasStatus_t iclblasChpr2(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* y, int incy, oclComplex_t* AP);

/*!
 * @brief Performs packed triangular matrix by vector multiplication
 *
 * @code
 * x = op(A) * x
 * @endcode
 * Where @b x is @b n element vector, @b A is @b n by @b n, unit or non-unit, upper or lower triangular matrix stored in packed format,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if @b A is upper or lower triangular
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] diag      indicates if @b A is unit or non-unit triangular
 * @param[in] n         number of rows and columns of matrix @b A; should be at least 0
 * @param[in] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 * @param[in,out] x     vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCtpmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* AP, oclComplex_t* x, int incx);

/*!
 * @brief Performs triangular banded matrix by vector multiplication
 *
 * @code
 * x = op(A) * x
 * @endcode
 * Where @b x is @b n element vector, @b A is @b n by @b n, upper or lower, unit or non-unit triangular banded matrix with @b k sub- or super-diagonals,
 * and `op(A)` is indicated by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(k + i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the top left @b k x @b k triangle) are not referenced.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the bottom right @b k x @b k triangle) are not referenced.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if @b A is upper or lower triangular
 * @param[in] trans     indicates operation used for multiplication
 * @param[in] diag      indicates if @b A is unit or non-unit triangular
 * @param[in] n         number of rows and columns in @b A; should be at least 0
 * @param[in] k         number of sub- or super-diagonals of matrix @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least @b k + 1
 * @param[in,out] x     vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCtbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, oclComplex_t* A, int lda, oclComplex_t* x, int incx);

/*!
 * @brief Solves triangular banded linear system with single right-hand side
 *
 * @code
 * op(A) * x = b
 * @endcode
 * Where @b b and @b x are @b n element vectors and @b A is @b n by @b n, unit or non-unit, upper or lower triangular banded matrix with @b k sub- or super-diagonals.
 *
 * Operation on matrix @b A when solving is specified by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(k + i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the top left @b k x @b k triangle) are not referenced.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the bottom right @b k x @b k triangle) are not referenced.
 *
 * On exit solution @b x overwrites right-hand side vector @b b.
 *
 * No test for singularity or near-singularity is included in this routine.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if matrix @b A is an upper or lower triangular
 * @param[in] trans     indicates equation to be solved as operation on @b A
 * @param[in] diag      indicates if matrix @b A is unit or non-unit triangular
 * @param[in] n         specifies order of matrix @b A; should be at least 0
 * @param[in] k         number of sub- or super-diagonals of matrix @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       first dimension of @b A; should be at least @b k + 1
 * @param[in,out] x     array of size at least @b n * @b incx; on entry stores vector @b b, overwritten by vector @b x on exit
 * @param[in] incx      stride between elements in @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCtbsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, int k, oclComplex_t* A, int lda, oclComplex_t* x, int incx);

/*!
 * @brief Solves triangular linear system with single right-hand side
 *
 * @code
 * op(A) * x = b
 * @endcode
 * Where @b b and @b x are @b n element vectors and @b A is @b n by @b n, unit or non-unit, upper or lower triangular matrix.
 *
 * Equation to be solved is specified by value of @b trans as follows:
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * On exit solution @b x overwrites right-hand side vector @b b.
 *
 * No test for singularity or near-singularity is included in this routine.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if matrix @b A is an upper or lower triangular
 * @param[in] trans     indicates equation to be solved as operation on @b A
 * @param[in] diag      indicates if matrix @b A is unit or non-unit triangular
 * @param[in] n         specifies order of matrix @b A; should be at least 0
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       first dimension of @b A; should be at least max(1, @b n)
 * @param[in,out] x     array of size at least @b n * @b incx; on entry stores vector @b b, overwritten by vector @b x on exit
 * @param[in] incx      stride between elements in @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCtrsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* A, int lda, oclComplex_t* x, int incx);

/*!
 * @brief Solves packed triangular linear system with single right-hand side
 *
 * @code
 * op(A) * x = b
 * @endcode
 * Where @b b and @b x are @b n element vectors and @b A is @b n by @b n, unit or non-unit, upper or lower triangular matrix stored in packed format.
 *
 * Equation to be solved is specified by value of @b trans as operation on @b A.
 * @code
 * trans == ICLBLAS_OP_N    op(A) = A
 *
 * trans == ICLBLAS_OP_T    op(A) = A ^ T
 *
 * trans == ICLBLAS_OP_C    op(A) = A ^ H
 * @endcode
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(j + 1) * j / 2 + i]`.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is packed column by column with element `A(i, j)` stored in memory at location `AP[(2 * n - j - 1) * j / 2 + i]`.
 *
 * On exit solution @b x overwrites right-hand side vector @b b.
 *
 * No test for singularity or near-singularity is included in this routine.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if matrix @b A is an upper or lower triangular
 * @param[in] trans     indicates equation to be solved as operation on @b A
 * @param[in] diag      indicates if matrix @b A is unit or non-unit triangular
 * @param[in] n         specifies order of matrix @b A; should be at least 0
 * @param[in] AP        array of size at least @b n * (@b n + 1) / 2 containing matrix @b A stored in packed format
 * @param[in,out] x     array of size at least @b n * @b incx; on entry stores vector @b b, overwritten by vector @b x on exit
 * @param[in] incx      stride between elements in @b x; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasCtpsv(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int n, oclComplex_t* AP, oclComplex_t* x, int incx);

/*!
 * @brief Performs Hermitian banded matrix by vector multiplication
 *
 * @code
 * y = alpha * A * x + beta * y
 * @endcode
 * Where @b alpha and @b beta are scalars, @b x and @b y are @b n element vectors, @b A is @b n by @b n Hermitian banded matrix with @b k subdiagonals and superdiagonals.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_UPPER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(k + i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the top left @b k x @b k triangle) are not referenced.
 *
 * If `uplo == ::ICLBLAS_FILL_MODE_LOWER` then the matrix @b A is stored column by column with element `A(i, j)` at location `A(i - j, j)` in memory.
 * The elements that don't correspond to elements in banded matrix (the bottom right @b k x @b k triangle) are not referenced.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns of @b A; should be at least 0
 * @param[in] k         number of sub- and super-diagonals of matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] A         array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least @b k + 1
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements of @b x; should be at least 1
 * @param[in] beta      scalar used in multiplication; if `beta == 0`, @b y does not have to be initialized
 * @param[in,out] y     vector of size at least @b n * @b incy
 * @param[in] incy      stride between elements of @b y; should be at least 1
 */
ICLBLAS_API iclblasStatus_t iclblasChbmv(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* x, int incx, const oclComplex_t* beta, oclComplex_t* y, int incy);

/*!
 * @brief Performs symmetrix matrix rank 1 update
 *
 * @code
 * A = alpha * x * x ^ T + A
 * @endcode
 * Where @b alpha is scalar, @b x is @b n element vector and @b A is @b n by @b n symmetric matrix, stored in upper or lower mode.
 *
 * @param[in] handle    handle to the library context
 * @param[in] uplo      indicates if upper or lower part of matrix is stored in @b A
 * @param[in] n         number of rows and columns in matrix @b A; should be at least 0
 * @param[in] alpha     scalar used in multiplication
 * @param[in] x         vector of size at least @b n * @b incx
 * @param[in] incx      stride between elements in @b x; should be at least 1
 * @param[in,out] A     array of size [@b lda x @b n] storing matrix @b A
 * @param[in] lda       leading dimension of @b A; should be at least max(1, @b n)
 */
ICLBLAS_API iclblasStatus_t iclblasCsyr(iclblasHandle_t handle, iclblasFillMode_t uplo, int n, const oclComplex_t* alpha, oclComplex_t* x, int incx, oclComplex_t* A, int lda);
/*! @} */

/*****************************************************************************/
/*! @addtogroup BLAS_L3_S BLAS Level 3 Single
* @{
*/

/*!
* @brief Performs symmetric matrix by matrix multiplication
*
* @code
* y = alpha * A * B + beta * C  if side == ICLBLAS_SIDE_LEFT
* y = alpha * B * A + beta * C  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha and @b beta are scalars, @b B and @b C are @b n x @b m element matrices and @b A is symmetric matrix in lower or upper mode.
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b B
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] m      number of rows in @b B and @b C
* @param[in] n      number of columns in @b B and @b C
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasSsymm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, int m, int n, const float* alpha, float* A, int lda, float* B, int ldb, const float* beta, float* C, int ldc);

/*!
* @brief Performs symmetric rank-k update
*
* @code
* C = alpha * op(A) * op(A)^T + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars, @b op(A) is @b n x @b k and @b C is symmetric matrix in lower or upper mode.
*
* Aditionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^T   if trans == ICLBLAS_OP_T
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b C
* @param[in] trans  indicates operation op(A) for matrix @b A
* @param[in] n      number of rows in @b op(A) and @b C
* @param[in] k      number of columns in @b op(A)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b lda x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_T
* @param[in] lda    first dimension of matrix @b A
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b n)
*/
ICLBLAS_API iclblasStatus_t iclblasSsyrk(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const float* alpha, float* A, int lda, const float* beta, float* C, int ldc);

/*!
* @brief Performs symmetric rank-2k update
*
* @code
* C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars, @b op(A) and @b op(B) are @b n x @b k and @b C is symmetric matrix in lower or upper mode.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^T   if trans == ICLBLAS_OP_T
* @endcode
*
* Additionally operation on matrix @b B is specified by @b trans value as followed:
* @code
* op(B) = B     if trans == ICLBLAS_OP_N
* op(B) = B^T   if trans == ICLBLAS_OP_T
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b C
* @param[in] trans  indicates operation op(A) and op(B) for matrix @b A and @b B
* @param[in] n      number of rows in @b op(A), @b op(B) and @b C
* @param[in] k      number of columns in @b op(A) and @b op(B)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b lda x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_T
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b k] where ldb is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b ldb x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_T
* @param[in] ldb    first dimension of matrix @b B
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b n)
*/
ICLBLAS_API iclblasStatus_t iclblasSsyr2k(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const float* alpha, float* A, int lda, float* B, int ldb, const float* beta, float* C, int ldc);

/*!
* @brief Performs matrix by matrix multiplication
*
* @code
* C = alpha * op(A) * op(B) + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars and @b A, @b B and @b C are matrices with dimmensions @b m x @b k for @b op(A), @b k x @b n for @b op(B) and @b m x @b n for @b C.
*
* Additionally operation on matrix @b A is specified by @b transa value as followed:
* @code
* op(A) = A     if transa == ICLBLAS_OP_N
* op(A) = A^T   if transa == ICLBLAS_OP_T
* op(A) = A^H   if transa == ICLBLAS_OP_C
* @endcode
*
* Additionally operation on matrix @b B is specified by @b transb value as followed:
* @code
* op(B) = B     if transb == ICLBLAS_OP_N
* op(B) = B^T   if transb == ICLBLAS_OP_T
* op(B) = B^H   if transb == ICLBLAS_OP_C
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] transa indicates operation op(A) for matrix @b A
* @param[in] transb indicates operation op(B) for matrix @b B
* @param[in] m      number of rows in @b op(A) and @b C
* @param[in] n      number of columns in @b op(B) and @b C
* @param[in] k      number of columns in @b op(A) and rows in @b op(B)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b m) if transa == ICLBLAS_OP_N and [@b lda x @b m] where lda is at least max (1, @b k) if transa == ICLBLAS_OP_T || transa == ICLBLAS_OP_C
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n] where ldb is at least max (1, @b k) if transb == ICLBLAS_OP_N and [@b ldb x @b k] where ldb is at least max (1, @b n) if transb == ICLBLAS_OP_T || transb == ICLBLAS_OP_C
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasSgemm(iclblasHandle_t handle, iclblasOperation_t transa, iclblasOperation_t transb, int m, int n, int k, const float* alpha, float* A, int lda, float* B, int ldb, const float* beta, float* C, int ldc);

/*!
* @brief Solves triangular linear system with multiple right-hand-sides
*
* @code
* op(A) * X = alpha * B  if side == ICLBLAS_SIDE_LEFT
* X * op(A) = alpha * B  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha is scalars, @b X and @b B are @b m x @b n matrices, and @b A is triangular matrix in lower or upper mode with or without main diagonal.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^T   if trans == ICLBLAS_OP_T
* op(A) = A^H   if trans == ICLBLAS_OP_C
* @endcode
*
* The solution X is overwritten on B on exit.
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b X
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] trans  indicates operation op(A) for matrix @b A
* @param[in] diag   indicates if main diagonal of matrix @b A is unitary or not
* @param[in] m      number of rows in @b B
* @param[in] n      number of columns in @b B
* @param[in] alpha  scalar used in multiplication; if @b alpha == 0, @b A and @b B do not have to be initialized
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in,out] B  array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasStrsm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int m, int n, const float* alpha, float* A, int lda, float* B, int ldb);

/*!
* @brief Performs triangular matrix by matrix multiplication
*
* @code
* C = alpha * op(A) * B  if side == ICLBLAS_SIDE_LEFT
* C = alpha * B * op(A)  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha is scalars, @b B and @b C are @b m x @b n matrices, and @b A is triangular matrix in lower or upper mode with or without main diagonal.
*
* Additionally operation on matrix @b A is specified by @b transa value as followed:
* @code
* op(A) = A     if transa == ICLBLAS_OP_N
* op(A) = A^T   if transa == ICLBLAS_OP_T
* op(A) = A^H   if transa == ICLBLAS_OP_C
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b B
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] transa indicates operation op(A) for matrix @b A
* @param[in] diag   indicates if main diagonal of matrix @b A is unitary or not
* @param[in] m      number of rows in @b B
* @param[in] n      number of columns in @b B
* @param[in] alpha  scalar used in multiplication; if @b alpha == 0, @b A and @b B do not have to be initialized
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[out] C     array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasStrmm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t transa, iclblasDiagType_t diag, int m, int n, const float* alpha, float* A, int lda, float* B, int ldb, float* C, int ldc);
/*! @} */

/*****************************************************************************/
/*! @addtogroup BLAS_L3_C BLAS Level 3 Complex
* @{
*/

/*!
* @brief Performs matrix by matrix multiplication
*
* @code
* C = alpha * op(A) * op(B) + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars and @b A, @b B and @b C are matrices with dimmensions @b m x @b k for @b op(A), @b k x @b n for @b op(B) and @b m x @b n for @b C.
*
* Additionally operation on matrix @b A is specified by @b transa value as followed:
* @code
* op(A) = A     if transa == ICLBLAS_OP_N
* op(A) = A^T   if transa == ICLBLAS_OP_T
* op(A) = A^H   if transa == ICLBLAS_OP_C
* @endcode
*
* Additionally operation on matrix @b B is specified by @b transb value as followed:
* @code
* op(B) = B     if transb == ICLBLAS_OP_N
* op(B) = B^T   if transb == ICLBLAS_OP_T
* op(B) = B^H   if transb == ICLBLAS_OP_C
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] transa indicates operation op(A) for matrix @b A
* @param[in] transb indicates operation op(B) for matrix @b B
* @param[in] m      number of rows in @b op(A) and @b C
* @param[in] n      number of columns in @b op(B) and @b C
* @param[in] k      number of columns in @b op(A) and rows in @b op(B)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b m) if transa == ICLBLAS_OP_N and [@b lda x @b m] where lda is at least max (1, @b k) if transa == ICLBLAS_OP_T || transa == ICLBLAS_OP_C
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n] where ldb is at least max (1, @b k) if transb == ICLBLAS_OP_N and [@b ldb x @b k] where ldb is at least max (1, @b n) if transb == ICLBLAS_OP_T || transb == ICLBLAS_OP_C
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasCgemm(iclblasHandle_t handle, iclblasOperation_t transa, iclblasOperation_t transb, int m, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc);

/*!
* @brief Performs symmetric matrix by matrix multiplication
*
* @code
* y = alpha * A * B + beta * C  if side == ICLBLAS_SIDE_LEFT
* y = alpha * B * A + beta * C  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha and @b beta are scalars, @b B and @b C are @b n x @b m element matrices and @b A is symmetric matrix in lower or upper mode.
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b B
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] m      number of rows in @b B and @b C
* @param[in] n      number of columns in @b B and @b C
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasCsymm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc);

/*!
* @brief Performs symmetric rank-2k update
*
* @code
* C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars, @b op(A) and @b op(B) are @b n x @b k and @b C is symmetric matrix in lower or upper mode.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^T   if trans == ICLBLAS_OP_T
* @endcode
*
* Additionally operation on matrix @b B is specified by @b trans value as followed:
* @code
* op(B) = B     if trans == ICLBLAS_OP_N
* op(B) = B^T   if trans == ICLBLAS_OP_T
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b C
* @param[in] trans  indicates operation op(A) and op(B) for matrix @b A and @b B
* @param[in] n      number of rows in @b op(A), @b op(B) and @b C
* @param[in] k      number of columns in @b op(A) and @b op(B)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b lda x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_T
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b k] where ldb is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b ldb x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_T
* @param[in] ldb    first dimension of matrix @b B
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b n)
*/
ICLBLAS_API iclblasStatus_t iclblasCsyr2k(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc);

/*!
* @brief Performs symmetric rank-k update
*
* @code
* C = alpha * op(A) * op(A)^T + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars, @b op(A) is @b n x @b k and @b C is symmetric matrix in lower or upper mode.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^T   if trans == ICLBLAS_OP_T
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b C
* @param[in] trans  indicates operation op(A) for matrix @b A
* @param[in] n      number of rows in @b op(A) and @b C
* @param[in] k      number of columns in @b op(A)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b lda x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_T
* @param[in] lda    first dimension of matrix @b A
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b n)
*/
ICLBLAS_API iclblasStatus_t iclblasCsyrk(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, const oclComplex_t* beta, oclComplex_t* C, int ldc);

/*!
* @brief Solves triangular linear system with multiple right-hand-sides
*
* @code
* op(A) * X = alpha * B  if side == ICLBLAS_SIDE_LEFT
* X * op(A) = alpha * B  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha is scalars, @b X and @b B are @b m x @b n matrices, and @b A is triangular matrix in lower or upper mode with or without main diagonal.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^T   if trans == ICLBLAS_OP_T
* op(A) = A^H   if trans == ICLBLAS_OP_C
* @endcode
*
* The solution X is overwritten on B on exit.
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b X
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] trans  indicates operation op(A) for matrix @b A
* @param[in] diag   indicates if main diagonal of matrix @b A is unitary or not
* @param[in] m      number of rows in @b B
* @param[in] n      number of columns in @b B
* @param[in] alpha  scalar used in multiplication; if @b alpha == 0, @b A and @b B do not have to be initialized
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in,out] B  array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasCtrsm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t trans, iclblasDiagType_t diag, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb);

/*!
* @brief Performs Hermitian rank-k update
*
* @code
* C = alpha * op(A) * op(A)^H + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars, @b op(A) is @b n x @b k and @b C is Hermitian matrix in lower or upper mode.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^H   if trans == ICLBLAS_OP_C
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b C
* @param[in] trans  indicates operation op(A) for matrix @b A
* @param[in] n      number of rows in @b op(A) and @b C
* @param[in] k      number of columns in @b op(A)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b lda x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_C
* @param[in] lda    first dimension of matrix @b A
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b n)
*/
ICLBLAS_API iclblasStatus_t iclblasCherk(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const float* alpha, oclComplex_t* A, int lda, const float* beta, oclComplex_t* C, int ldc);

/*!
* @brief Performs Hermitian rank-2k update
*
* @code
* C = alpha * op(A) * op(B)^H + \conjugate{alpha} * op(B) * op(A)^H + beta * C
* @endcode
*
* Where @b alpha and @b beta are scalars, @b op(A) and @b op(B) are @b n x @b k and @b C is Hermitian matrix in lower or upper mode.
*
* Additionally operation on matrix @b A is specified by @b trans value as followed:
* @code
* op(A) = A     if trans == ICLBLAS_OP_N
* op(A) = A^H   if trans == ICLBLAS_OP_C
* @endcode
*
* Additionally operation on matrix @b B is specified by @b trans value as followed:
* @code
* op(B) = B     if trans == ICLBLAS_OP_N
* op(B) = B^H   if trans == ICLBLAS_OP_C
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b C
* @param[in] trans  indicates operation op(A) and op(B) for matrix @b A and @b B
* @param[in] n      number of rows in @b op(A), @b op(B) and @b C
* @param[in] k      number of columns in @b op(A) and @b op(B)
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b k] where lda is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b lda x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_C
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b k] where ldb is at least max (1, @b n) if trans == ICLBLAS_OP_N and [@b ldb x @b n] where lda is at least max (1, @b k) if side == ICLBLAS_OP_C
* @param[in] ldb    first dimension of matrix @b B
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b n)
*/
ICLBLAS_API iclblasStatus_t iclblasCher2k(iclblasHandle_t handle, iclblasFillMode_t uplo, iclblasOperation_t trans, int n, int k, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const float* beta, oclComplex_t* C, int ldc);

/*!
* @brief Performs triangular matrix by matrix multiplication
*
* @code
* C = alpha * op(A) * B  if side == ICLBLAS_SIDE_LEFT
* C = alpha * B * op(A)  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha is scalars, @b B and @b C are @b m x @b n matrices, and @b A is triangular matrix in lower or upper mode with or without main diagonal.
*
* Additionally  operation on matrix @b B is specified by @b transa value as followed:
* @code
* op(A) = A     if transa == ICLBLAS_OP_N
* op(A) = A^T   if transa == ICLBLAS_OP_T
* op(A) = A^H   if transa == ICLBLAS_OP_C
* @endcode
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b B
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] transa indicates operation op(A) for matrix @b A
* @param[in] diag   indicates if main diagonal of matrix @b A is unitary or not
* @param[in] m      number of rows in @b B
* @param[in] n      number of columns in @b B
* @param[in] alpha  scalar used in multiplication; if @b alpha == 0, @b A and @b B do not have to be initialized
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[out] C     array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasCtrmm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, iclblasOperation_t transa, iclblasDiagType_t diag, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, oclComplex_t* C, int ldc);

/*!
* @brief Performs Hermitian matrix by matrix multiplication
*
* @code
* alpha * A * B + beta * C  if side == ICLBLAS_SIDE_LEFT
* alpha * B * A + beta * C  if side == ICLBLAS_SIDE_RIGHT
* @endcode
*
* Where @b alpha and @b beta are scalars, @b B and @b C are @b m x @b n matrices, and @b A is Hermitian matrix in lower or upper mode.
*
* @param[in] handle handle to the library context
* @param[in] side   indicates right or left sided multiplication of matrix @b A and @b B
* @param[in] uplo   indicates if lower or upper part of matrix is stored in @b A
* @param[in] m      number of rows in @b B and @b C
* @param[in] n      number of columns in @b B and @b C
* @param[in] alpha  scalar used in multiplication
* @param[in] A      array of size [@b lda x @b m] where lda is at least max (1, @b m) if side == ICLBLAS_SIDE_LEFT and [@b lda x @b n] where lda is at least max (1, @b n) if side == ICLBLAS_SIDE_RIGHT
* @param[in] lda    first dimension of matrix @b A
* @param[in] B      array of size [@b ldb x @b n]
* @param[in] ldb    first dimension of matrix @b B; must be at least max(1, @b m)
* @param[in] beta   scalar used in multiplication; if @b beta == 0, @b C does not have to be initialized
* @param[in,out] C  array of size [@b ldc x @b n]
* @param[in] ldc    first dimension of matrix @b C; must be at least max(1, @b m)
*/
ICLBLAS_API iclblasStatus_t iclblasChemm(iclblasHandle_t handle, iclblasSideMode_t side, iclblasFillMode_t uplo, int m, int n, const oclComplex_t* alpha, oclComplex_t* A, int lda, oclComplex_t* B, int ldb, const oclComplex_t* beta, oclComplex_t* C, int ldc);
/*! @} */

#ifdef __cplusplus
} // extern "C"
#endif
