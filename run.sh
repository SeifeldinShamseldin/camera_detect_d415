#!/bin/bash

# Intel RealSense D415 Camera Detection System
# Cross-platform startup script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to setup X11 forwarding
setup_x11() {
    local os=$(detect_os)
    
    case $os in
        "linux")
            print_status "Setting up X11 for Linux..."
            xhost +local:docker 2>/dev/null || print_warning "Could not set xhost permissions"
            export DISPLAY=${DISPLAY:-:0}
            ;;
        "macos")
            print_status "Setting up X11 for macOS..."
            if ! command_exists xquartz; then
                print_warning "XQuartz not found. Please install: brew install --cask xquartz"
            fi
            export DISPLAY=${DISPLAY:-:0}
            xhost +localhost 2>/dev/null || print_warning "Could not set xhost permissions"
            ;;
        "windows")
            print_status "Setting up X11 for Windows..."
            export DISPLAY=${DISPLAY:-host.docker.internal:0.0}
            ;;
        *)
            print_warning "Unknown OS detected. X11 setup may not work correctly."
            ;;
    esac
}

# Function to check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker Desktop."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_success "Docker is ready"
}

# Function to check USB devices (Linux only)
check_usb_devices() {
    local os=$(detect_os)
    
    if [[ "$os" == "linux" ]]; then
        print_status "Checking for Intel RealSense devices..."
        if lsusb 2>/dev/null | grep -i intel >/dev/null; then
            print_success "Intel RealSense device detected"
        else
            print_warning "No Intel RealSense device found via lsusb"
        fi
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data logs models config
    
    # Create X11 auth file if it doesn't exist
    if [[ ! -f /tmp/.docker.xauth ]]; then
        touch /tmp/.docker.xauth
        if command_exists xauth && [[ -n "$DISPLAY" ]]; then
            xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 2>/dev/null || true
        fi
    fi
    
    print_success "Directories created"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    
    if docker-compose build; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run the application
run_application() {
    print_status "Starting Camera Detection System..."
    
    # Set up environment
    setup_x11
    create_directories
    
    # Export environment variables for docker-compose
    export DISPLAY
    export XAUTHORITY=/tmp/.docker.xauth
    
    # Run the application
    if docker-compose up; then
        print_success "Application stopped gracefully"
    else
        print_error "Application encountered an error"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker image only"
    echo "  run       Run the application (default)"
    echo "  dev       Run in development mode"
    echo "  logs      Show application logs"
    echo "  stop      Stop the application"
    echo "  clean     Clean up Docker resources"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run the application"
    echo "  $0 build          # Build Docker image"
    echo "  $0 dev            # Run in development mode"
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f
}

# Function to stop application
stop_application() {
    print_status "Stopping application..."
    docker-compose down
    print_success "Application stopped"
}

# Function to clean up Docker resources
clean_docker() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (optional)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    print_success "Docker cleanup completed"
}

# Function to run in development mode
run_development() {
    print_status "Starting in development mode..."
    
    setup_x11
    create_directories
    
    export DISPLAY
    export XAUTHORITY=/tmp/.docker.xauth
    
    # Run with development profile
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
}

# Main script logic
main() {
    local command=${1:-run}
    
    print_status "Intel RealSense D415 Camera Detection System"
    print_status "============================================="
    
    # Check Docker installation for most commands
    if [[ "$command" != "help" ]]; then
        check_docker
        check_usb_devices
    fi
    
    case $command in
        "build")
            build_image
            ;;
        "run")
            build_image
            run_application
            ;;
        "dev")
            build_image
            run_development
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_application
            ;;
        "clean")
            clean_docker
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"