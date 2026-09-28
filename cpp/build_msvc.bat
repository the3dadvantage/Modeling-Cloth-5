@echo off
rem Builds both native libraries into addon\lib\windows-x64\ , the Windows
rem counterpart of cpp\build_unix.sh.  Needs Visual Studio with the C++ workload.
rem
rem   cpp\build_msvc.bat
rem
rem cpp\mc_collide\build.bat still exists and writes mc_collide.dll straight into
rem Python\ for development; this one produces the pair that
rem tools\build_addon.py packages.
setlocal
set HERE=%~dp0
set ROOT=%HERE%..
set OUT=%ROOT%\addon\lib\windows-x64

set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat
)
if not exist "%VCVARS%" (
    echo Could not find vcvars64.bat -- is the C++ workload installed?
    exit /b 1
)
call "%VCVARS%" >nul 2>nul

if not exist "%OUT%" mkdir "%OUT%"
if not exist "%HERE%obj" mkdir "%HERE%obj"

echo building mc_collide.dll
cl /nologo /O2 /fp:precise /EHsc /std:c++17 /W3 /LD "%HERE%mc_collide\mc_collide.cpp" /Fo"%HERE%obj\\" /Fe"%OUT%\mc_collide.dll" /link /NOLOGO /IMPLIB:"%HERE%obj\mc_collide.lib"
if errorlevel 1 (
    echo BUILD FAILED: mc_collide
    exit /b 1
)

echo building mc_cloth_solver.dll
cl /nologo /O2 /fp:precise /EHsc /std:c++17 /W3 /LD "%HERE%solver\cloth_solver.cpp" /I"%HERE%solver" /Fo"%HERE%obj\\" /Fe"%OUT%\mc_cloth_solver.dll" /link /NOLOGO /IMPLIB:"%HERE%obj\mc_cloth_solver.lib"
if errorlevel 1 (
    echo BUILD FAILED: cloth_solver
    exit /b 1
)

echo.
echo BUILT into %OUT%
dir /b "%OUT%"
