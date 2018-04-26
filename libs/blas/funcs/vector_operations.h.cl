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

#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#define PRIMITIVE_CAT(a, b) a ## b
#define CAT(a, b) PRIMITIVE_CAT( a, b )

#define VEC1
#define VEC2 2
#define VEC3 3
#define VEC4 4
#define VEC8 8
#define VEC16 16

#define UNIVERSAL_VEC_ACCESS(vec, elem) vec[elem]

#define VEC_ACCESS1(vec, elem) vec
#define VEC_ACCESS2(vec, elem) UNIVERSAL_VEC_ACCESS( vec, elem )
#define VEC_ACCESS3(vec, elem) UNIVERSAL_VEC_ACCESS( vec, elem )
#define VEC_ACCESS4(vec, elem) UNIVERSAL_VEC_ACCESS( vec, elem )
#define VEC_ACCESS8(vec, elem) UNIVERSAL_VEC_ACCESS( vec, elem )
#define VEC_ACCESS16(vec, elem) UNIVERSAL_VEC_ACCESS( vec, elem )

#define VEC_ACCESSN(vec_size, vec, elem) CAT( VEC_ACCESS, vec_size ) ( vec, elem )

#define VECN(vec_size) CAT( VEC, vec_size )

#define TYPEN(type, vec_size) CAT( type, VECN( vec_size ) )

#define FLOATN(vec_size) TYPEN( float, vec_size )
#define UINTN(vec_size) TYPEN( uint, vec_size )

#define LIST_LOAD1(ptr, ind, inc) ptr[(ind)]
#define LIST_LOAD2(ptr, ind, inc) LIST_LOAD1( ptr, ind, inc ), LIST_LOAD1( ptr, (ind) + (inc), inc )
#define LIST_LOAD3(ptr, ind, inc) LIST_LOAD2( ptr, ind, inc ), LIST_LOAD1( ptr, (ind) + 2 * (inc), inc )
#define LIST_LOAD4(ptr, ind, inc) LIST_LOAD2( ptr, ind, inc ), LIST_LOAD2( ptr, (ind) + 2 * (inc), inc )
#define LIST_LOAD8(ptr, ind, inc) LIST_LOAD4( ptr, ind, inc ), LIST_LOAD4( ptr, (ind) + 4 * (inc), inc )
#define LIST_LOAD16(ptr, ind, inc) LIST_LOAD8( ptr, ind, inc ), LIST_LOAD8( ptr, (ind) + 8 * (inc), inc )

#define LIST_LOADN(type, vec_size, ptr, ind, inc) ( TYPEN( type, vec_size ) ) ( CAT( LIST_LOAD, vec_size ) (ptr, ind, inc) )
#define FLOAT_LIST_LOADN(vec_size, ptr, ind, inc) LIST_LOADN( float, vec_size, ptr, ind, inc)

#define AS_FLOATN(vec_size) CAT( as_, FLOATN( vec_size ) )
#define AS_UINTN(vec_size) CAT( as_, UINTN( vec_size ) )

#define BLOCK_READN(vec_size) CAT( intel_sub_group_block_read, VECN( vec_size ) )
#define BLOCK_WRITEN(vec_size) CAT( intel_sub_group_block_write, VECN( vec_size ) )

#define UNIVERSAL_VLOAD(vec_size) CAT( vload, vec_size )
#define VLOAD1(ind, ptr) ptr[ind]
#define VLOAD2(ind, ptr) UNIVERSAL_VLOAD( 2 ) ( ind, ptr )
#define VLOAD3(ind, ptr) UNIVERSAL_VLOAD( 3 ) ( ind, ptr )
#define VLOAD4(ind, ptr) UNIVERSAL_VLOAD( 4 ) ( ind, ptr )
#define VLOAD8(ind, ptr) UNIVERSAL_VLOAD( 8 ) ( ind, ptr )
#define VLOAD16(ind, ptr) UNIVERSAL_VLOAD( 16 ) ( ind, ptr )
#define VLOADN(vec_size) CAT( VLOAD, vec_size )

#define UNIVERSAL_VSTORE(vec_size) CAT( vstore, vec_size )
#define VSTORE1(val, ind, ptr) ptr[ind] = val
#define VSTORE2(val, ind, ptr) UNIVERSAL_VSTORE( 2 ) ( val, ind, ptr )
#define VSTORE3(val, ind, ptr) UNIVERSAL_VSTORE( 3 ) ( val, ind, ptr )
#define VSTORE4(val, ind, ptr) UNIVERSAL_VSTORE( 4 ) ( val, ind, ptr )
#define VSTORE8(val, ind, ptr) UNIVERSAL_VSTORE( 8 ) ( val, ind, ptr )
#define VSTORE16(val, ind, ptr) UNIVERSAL_VSTORE( 16 ) ( val, ind, ptr )
#define VSTOREN(vec_size) CAT( VSTORE, vec_size )

#define VEC_SUM1(vec) (vec)
#define VEC_SUM2(vec) ( (vec).x + (vec).y )
#define VEC_SUM3(vec) ( VEC_SUM2( vec ) + (vec).z )
#define VEC_SUM4(vec) ( VEC_SUM2( vec ) + ((vec).z + (vec).w) )
#define VEC_SUM8(vec) ( VEC_SUM4( vec ) + (((vec).s4 + (vec).s5) + ((vec).s6 + (vec).s7)) )
#define VEC_SUM16(vec) ( VEC_SUM8( vec ) + ((((vec).s8 + (vec).s9) + ((vec).sa + (vec).sb)) + (((vec).sc + (vec).sd) + ((vec).se + (vec).sf))) )

#define VEC_SUMN(vec_size, vec) CAT( VEC_SUM, vec_size ) ( vec )

/* Predefined shortcuts for vectors of size VEC_SIZE */
#ifdef VEC_SIZE

#define VEC_ACCESS_VS(vec, elem) VEC_ACCESSN( VEC_SIZE, vec, elem )

#define TYPE_VS(type) TYPEN( type, VEC_SIZE )

#define FLOAT_VS FLOATN( VEC_SIZE )
#define UINT_VS UINTN( VEC_SIZE )

#define LIST_LOAD_VS(type, vec, ind, inc) LIST_LOADN( type, VEC_SIZE, vec, ind, inc )

#define FLOAT_LIST_LOAD_VS(vec, ind, inc) FLOAT_LIST_LOADN( VEC_SIZE, vec, ind, inc )

#define AS_FLOAT_VS AS_FLOATN( VEC_SIZE )
#define AS_UINT_VS AS_UINTN( VEC_SIZE )

#define BLOCK_READ_VS BLOCK_READN( VEC_SIZE )
#define BLOCK_WRITE_VS BLOCK_WRITEN( VEC_SIZE )

#define VLOAD_VS VLOADN( VEC_SIZE )
#define VSTORE_VS VSTOREN( VEC_SIZE )

#define VEC_SUM_VS( vector ) VEC_SUMN( VEC_SIZE, vector )

#endif /* VEC_SIZE */

#endif /* VECTOR_OPERATIONS_H */
