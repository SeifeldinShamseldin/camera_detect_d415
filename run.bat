@echo off
REM Intel RealSense D415 Camera Detection System
REM Windows startup script

setlocal enabledelayedexpansion

:: Colors for output (limited in Windows batch)
SET "ESC="

:: Function to print status messages
:print_status
echo [INFO] %~1
exit /b

:print_success
echo [SUCCESS] %~1
exit /b

:print_warning
echo [WARNING] %~1
exit /b

:print_error
echo [ERROR] %~1
exit /b

:: Check if Docker is installed
:check_docker
call :print_status "Checking Docker installation..."

docker --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not installed. Please install Docker Desktop for Windows."
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker Compose is not installed."
    pause
    exit /b 1
)

:: Check if Docker daemon is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker daemon is not running. Please start Docker Desktop."
    pause
    exit /b 1
)

call :print_success "Docker is ready"
exit /b

:: Setup X11 forwarding for Windows
:setup_x11
call :print_status "Setting up X11 for Windows..."

:: Set display environment variable
if "%DISPLAY%"=="" (
    set DISPLAY=host.docker.internal:0.0
)

:: Check if VcXsrv or similar is running
tasklist /fi "imagename eq vcxsrv.exe" 2>nul | find /i "vcxsrv.exe" >nul
if %errorlevel% neq 0 (
    call :print_warning "VcXsrv (X11 server) not detected. Please start XLaunch or similar X11 server."
)

call :print_status "X11 setup completed"
exit /b

:: Create necessary directories
:create_directories
call :print_status "Creating necessary directories..."

if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "config" mkdir config

:: Create X11 auth file for Windows
if not exist "C:\temp" mkdir C:\temp
if not exist "C:\temp\.docker.xauth" (
    echo. > C:\temp\.docker.xauth
)

call :print_success "Directories created"
exit /b

:: Build Docker image
:build_image
call :print_status "Building Docker image..."

docker-compose build
if %errorlevel% neq 0 (
    call :print_error "Failed to build Docker image"
    pause
    exit /b 1
)

call :print_success "Docker image built successfully"
exit /b

:: Run the application
:run_application
call :print_status "Starting Camera Detection System..."

call :setup_x11
call :create_directories

:: Export environment variables
set XAUTHORITY=C:\temp\.docker.xauth

:: Run the application
docker-compose up
if %errorlevel% equ 0 (
    call :print_success "Application stopped gracefully"
) else (
    call :print_error "Application encountered an error"
    pause
    exit /b 1
)
exit /b

:: Show application logs
:show_logs
call :print_status "Showing application logs..."
docker-compose logs -f
exit /b

:: Stop application
:stop_application
call :print_status "Stopping application..."
docker-compose down
call :print_success "Application stopped"
exit /b

:: Clean up Docker resources
:clean_docker
call :print_status "Cleaning up Docker resources..."

:: Stop and remove containers
docker-compose down --remove-orphans

:: Remove unused images
docker image prune -f

:: Ask about volume cleanup
set /p "cleanup_volumes=Remove unused volumes? (y/N): "
if /i "%cleanup_volumes%"=="y" (
    docker volume prune -f
)

call :print_success "Docker cleanup completed"
exit /b

:: Run in development mode
:run_development
call :print_status "Starting in development mode..."

call :setup_x11
call :create_directories

set XAUTHORITY=C:\temp\.docker.xauth

:: Check if development compose file exists
if not exist "docker-compose.dev.yml" (
    call :print_warning "Development compose file not found. Running in standard mode."
    docker-compose up
) else (
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
)
exit /b

:: Show usage
:show_usage
echo Usage: %~n0 [COMMAND]
echo.
echo Commands:
echo   build     Build Docker image only
echo   run       Run the application (default)
echo   dev       Run in development mode
echo   logs      Show application logs
echo   stop      Stop the application
echo   clean     Clean up Docker resources
echo   help      Show this help message
echo.
echo Examples:
echo   %~n0                 # Run the application
echo   %~n0 build          # Build Docker image
echo   %~n0 dev            # Run in development mode
echo.
pause
exit /b

:: Main script logic
:main
set "command=%~1"
if "%command%"=="" set "command=run"

call :print_status "Intel RealSense D415 Camera Detection System"
call :print_status "============================================="

:: Check Docker for most commands
if not "%command%"=="help" (
    call :check_docker
    if %errorlevel% neq 0 exit /b 1
)

if "%command%"=="build" (
    call :build_image
) else if "%command%"=="run" (
    call :build_image
    if %errorlevel% equ 0 call :run_application
) else if "%command%"=="dev" (
    call :build_image
    if %errorlevel% equ 0 call :run_development
) else if "%command%"=="logs" (
    call :show_logs
) else if "%command%"=="stop" (
    call :stop_application
) else if "%command%"=="clean" (
    call :clean_docker
) else if "%command%"=="help" (
    call :show_usage
) else if "%command%"=="-h" (
    call :show_usage
) else if "%command%"=="--help" (
    call :show_usage
) else (
    call :print_error "Unknown command: %command%"
    call :show_usage
    exit /b 1
)

exit /b

:: Call main function
call :main %*