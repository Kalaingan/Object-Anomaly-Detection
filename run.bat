@echo off
setlocal

:: Check for argument
if "%~1"=="" (
    call :show_help
    exit /b 1
)

:: Process arguments
if "%~1"=="train" (
    echo Training the model...
    python app.py
    exit /b 0
)

if "%~1"=="deploy" (
    echo Starting Streamlit application...
    streamlit run streamlit_app.py
    exit /b 0
)

if "%~1"=="test" (
    if "%~2"=="" (
        echo Error: No image path provided for testing.
        echo Usage: run.bat test path\to\image.jpg
        exit /b 1
    )
    echo Testing model on image: %~2
    python test_model.py "%~2"
    exit /b 0
)

if "%~1"=="help" (
    call :show_help
    exit /b 0
)

echo Error: Unknown option '%~1'
call :show_help
exit /b 1

:show_help
echo Usage: run.bat [OPTION]
echo Run different components of the anomaly detection application.
echo.
echo Options:
echo   train       Train the model using the provided dataset
echo   deploy      Run the Streamlit web application
echo   test IMAGE  Test the model on a single image
echo   help        Show this help message
echo.
echo Examples:
echo   run.bat train
echo   run.bat deploy
echo   run.bat test path\to\image.jpg
exit /b 0 
