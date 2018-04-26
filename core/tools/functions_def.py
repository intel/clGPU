#!/usr/bin/env python3
#
# Copyright (c) 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To add new kernel please add a .cl file to kernels directory
# the database name will be the part of the file name up to first '.' character
# the trailing characters are a tag to allow multiple primitive implementations
"""Lexical analyzer for iclGPU functions definitions file"""

from shlex import shlex
import sys

__all__ = ["FunctionsDef", "ParseError"]


class ParseError(SyntaxError):
    """functions definition file syntax error """
    pass


class FunctionsDef:
    """Parse functions definitions file"""

    def __init__(self, file, file_name=None):
        self.keywords = dict(
            open_brace=('{',),
            close_brace=('}',),
            list_separator=(',',),
            colon=(':',),
            function=('function',),
            blob=('blob',),
            inout=('input', 'output', 'inout'),
            struct=('struct',),
            params=('params',),
            implementations=('implementations',),
            impl_type=('impl_type',),
            type=(
                'char', 'uchar',
                'short', 'ushort',
                'int', 'uint',
                'long', 'ulong',
                'float', 'double',
                'complex',
                'void',
            ),
        )
        self.c_type_map = {
            'char': 'int8_t', 'uchar': 'uint8_t',
            'short': 'int16_t', 'ushort': 'uint16_t',
            'int': 'int32_t', 'uint': 'uint32_t',
            'long': 'int64_t', 'ulong': 'uint64_t',
            'float': 'float', 'double': 'double',
            'complex': 'complex_t',
            'void': 'void',
        }
        self.cl_type_map = {
            'char': 'char', 'uchar': 'uchar',
            'short': 'short', 'ushort': 'ushort',
            'int': 'int', 'uint': 'uint',
            'long': 'long', 'ulong': 'ulong',
            'float': 'float', 'double': 'double',
            'complex': 'complex_t',
            'void': 'char',
        }
        self.inout_map = {'input': 'input', 'output': 'output', 'inout': 'inout'}

        self.known_names = dict()
        self.__tokenizer = shlex(file, file_name)

    def register_name(self, name, type_):
        known_name = self.known_names.get(name)
        if known_name is None:
            self.known_names[name] = {'type': type_, 'line': self.__lineno()}
        else:
            self.__syntax_error(
                'name "{}" is used for {} at line {}'.format(name, known_name['type'], known_name['line']))

    def __syntax_error(self, msg):
        raise ParseError(self.__tokenizer.error_leader() + msg)

    def __tokenize(self):
        return next(self.__tokenizer)

    def __lineno(self):
        return self.__tokenizer.lineno

    def __expect(self, token, expected):
        keys = (expected,) if isinstance(expected, (str, bytes)) else expected
        for key in keys:
            if token in self.keywords[key]:
                return key
        else:
            expected_strings = [i for j in [self.keywords[k] for k in keys] for i in j]
            self.__syntax_error('unexpected token: "{}". expected: {}'.format(token, expected_strings))

    def __select(self, token, selector):
        key = self.__expect(token, selector.keys())
        return selector[key](token)

    def __parse_list(self, element_def, key_builder=None):
        self.__expect(self.__tokenize(), 'open_brace')
        elements = []
        keys = set()
        while 1:
            element = element_def()
            if key_builder is not None:
                key = key_builder(element)
                if key not in keys:
                    keys.add(key)
                else:
                    self.__syntax_error('list element "{}" is already defined'.format(key))
            elements.append(element)
            token = self.__tokenize()
            if token not in self.keywords['list_separator']:
                break
        self.__expect(token, 'close_brace')
        return elements

    def __parse_param_type(self, blob='', inout='inout', struct=''):
        def parse_blob_(token):
            if blob or struct:
                self.__syntax_error('"blob" is not allowed here')
            return self.__parse_param_type(token, inout, struct)

        def parse_inout_(token):
            if not blob:
                self.__syntax_error('"blob" is was expected')
            return self.__parse_param_type(blob, token, struct)

        def parse_struct_(token):
            if struct:
                self.__syntax_error('"struct" used twice')
            struct_name = self.__tokenize()
            known_name = self.known_names.get(struct_name)
            if known_name is None:
                self.__syntax_error('struct "{}" is undefined'.format(struct_name))
            if known_name['type'] != 'struct':
                self.__syntax_error(
                    '{} is defined as {} at line {}'.format(struct_name, known_name['type'], known_name['line']))
            self.uses.add(struct_name)
            return struct_name, blob, inout, token

        return self.__select(self.__tokenize(), {
            'type': lambda token: (token, blob, struct, inout),
            'blob': parse_blob_,
            'inout': parse_inout_,
            'struct': parse_struct_
        })

    def __parse_param(self):
        type_, blob, struct, inout = self.__parse_param_type()

        struct_prefix = 'struct ' if struct else ''
        c_prefix = ('blob<' if blob == 'blob' else '') + struct_prefix
        c_postfix = ', ' + self.inout_map[inout] + '>' if blob == 'blob' else ''
        c_type = c_prefix + (type_ if struct else self.c_type_map[type_]) + c_postfix

        cl_postfix = '*' if blob else ''
        cl_type = struct_prefix + (type_ if struct else self.cl_type_map[type_]) + cl_postfix
        return {
            'name': self.__tokenize(),
            'type': type_,
            'blob': blob,
            'struct': struct,
            'c_type': c_type,
            'cl_type': cl_type,
            'inout': inout,
            'inout_type': self.inout_map[inout]
        }

    def __parse_implementation(self):
        return self.__tokenize()

    def __parse_impl_type(self):
        self.__expect(self.__tokenize(), 'colon')
        return self.__tokenize()

    def __parse_function_elem(self):
        return self.__select(self.__tokenize(), {
            'params': lambda token:
                ('params', self.__parse_list(self.__parse_param, lambda element: element['name'])),
            'implementations': lambda token:
                ('implementations', self.__parse_list(self.__parse_implementation, lambda element: element)),
            'impl_type': lambda token:
                ('impl_type', self.__parse_impl_type())
        })

    def __parse_struct_elem(self):
        type_, blob, struct, inout = self.__parse_param_type(blob='blob')
        prefix = 'struct ' if struct == 'struct' else ''
        c_type = prefix + (type_ if struct == 'struct' else self.c_type_map[type_])
        cl_type = prefix + (type_ if struct == 'struct' else self.cl_type_map[type_])
        return {
            'name': self.__tokenize(),
            'type': type_,
            'struct': struct,
            'c_type': c_type,
            'cl_type': cl_type
        }

    def __iter__(self):
        return self

    def __next__(self):
        type_ = self.__tokenize()
        name = self.__tokenize()
        self.uses = set()
        elements = self.__select(type_, {
            'function': lambda token: self.__parse_list(self.__parse_function_elem, lambda element: element[0]),
            'struct': lambda token: self.__parse_list(self.__parse_struct_elem, lambda element: element['name'])
        })
        self.register_name(name, type_)

        if type_ == 'function':
            result = dict(elements, name=name, type=type_, uses=self.uses)
            params = result.get('params')
            if params is None:
                self.__syntax_error('missing parameters definition for function: ' + name)
            if len(params) < 1:
                self.__syntax_error('at least 1 parameter should be defined for function ' + name)
            return result
        elif type_ == 'struct':
            if len(elements) < 1:
                self.__syntax_error('at least 1 field should be defined for structure ' + name)
            return dict(fields=elements, name=name, type=type_, uses=self.uses)

    def next(self):
        return self.__next__()

if __name__ == '__main__':
    file_name = 'functions.fdef'
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    for item in FunctionsDef(open(file_name), file_name):
        print(item)
