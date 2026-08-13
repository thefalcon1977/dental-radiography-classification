@ECHO OFF

pushd %~dp0

if "%SPHINXBUILD%" == "" (
	if exist "%~dp0..\.venv\Scripts\sphinx-build.exe" (
		set SPHINXBUILD=%~dp0..\.venv\Scripts\sphinx-build.exe
	) else (
		set SPHINXBUILD=sphinx-build
	)
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found.
	echo.Activate .venv or: pip install sphinx sphinx-rtd-theme
	exit /b 1
)

popd
