@echo off
echo.
echo ⚛️  Loom Poly - Polymorphic Semantic Search Demo
echo -----------------------------------------------
echo.
echo ⏳ Step 1: Downloading and Converting GloVe 6B (300d)...
echo (Note: This will download ~822MB if not already present)
echo.
go run converter/main.go
if %errorlevel% neq 0 (
    echo ❌ Conversion failed.
    pause
    exit /b %errorlevel%
)
echo.
echo 🚀 Step 2: Starting Interactive Search Demo...
echo.
go run main.go corpus.go
pause
