@echo off
: Copyright (c) 2017-2018 Intel Corporation
: 
: Licensed under the Apache License, Version 2.0 (the "License");
: you may not use this file except in compliance with the License.
: You may obtain a copy of the License at
: 
:      http://www.apache.org/licenses/LICENSE-2.0
: 
: Unless required by applicable law or agreed to in writing, software
: distributed under the License is distributed on an "AS IS" BASIS,
: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
: See the License for the specific language governing permissions and
: limitations under the License.

@setlocal

set ROOT_DIR=%~dp0
set SOLUTION_DIR=%ROOT_DIR%\build

echo Creating Visual Studio 2015 (x64) files in "%SOLUTION_DIR%"...
cd /d "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR%" && cd "%SOLUTION_DIR%" && cmake -G "Visual Studio 14 2015 Win64" -T "host=x64" "%ROOT_DIR%" && cmake --build "%SOULUTION_DIR%" && "%SOLUTION_DIR%\out\Debug\test_iclBLAS.exe"

echo Done.
pause
