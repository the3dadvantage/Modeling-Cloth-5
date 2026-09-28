@echo off
rem Builds mc_collide.dll into the addon folder (..\..\Python).
rem Needs Visual Studio 2022+ with the C++ workload.  Safe to run while
rem Blender is open: the bridge loads a copy of the DLL, not this file.
setlocal
set HERE=%~dp0
set OUT=%HERE%..\..\Python\mc_collide.dll
set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat
)
call "%VCVARS%" >nul 2>nul
if not exist "%HERE%obj" mkdir "%HERE%obj"
cl /nologo /O2 /fp:precise /EHsc /std:c++17 /W3 /LD "%HERE%mc_collide.cpp" /Fo"%HERE%obj\\" /Fe"%OUT%" /link /NOLOGO /IMPLIB:"%HERE%obj\mc_collide.lib"
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILT %OUT%

