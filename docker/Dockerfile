FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    # Python and pip
    python3 \
    python3-pip \
    python3-dev \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Intel RealSense dependencies
    libudev-dev \
    libusb-1.0-0-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    # Additional libraries
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    # Display support
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Intel RealSense SDK
RUN mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y \
        librealsense2-dkms \
        librealsense2-utils \
        librealsense2-dev \
        librealsense2-dbg \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY ../requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy application code
COPY .. .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV DISPLAY=:0

# Create non-root user for security
RUN useradd -m -u 1000 camera_user && \
    chown -R camera_user:camera_user /app

# Switch to non-root user
USER camera_user

# Expose port for web interface (if applicable)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pyrealsense2 as rs; print('RealSense OK')" || exit 1

# Default command
CMD ["python3", "main.py"]