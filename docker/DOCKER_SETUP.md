# Docker Setup Guide for Intel RealSense D415 Camera Detection

This guide provides comprehensive instructions for running the Intel RealSense D415 camera detection system using Docker on any operating system.

## Prerequisites

### Required Software
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Docker Compose v2.0+
- Intel RealSense D415 Camera

### System Requirements
- USB 3.0 port for RealSense camera
- 4GB+ RAM
- GPU support (optional, for acceleration)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/SeifeldinShamseldin/camera_detect_d415.git
cd camera_detect_d415
```

### 2. Build and Run
```bash
# Build the Docker image
docker-compose build

# Run the application
docker-compose up
```

## Platform-Specific Setup

### Windows Setup

#### Prerequisites
1. Install Docker Desktop for Windows
2. Enable WSL2 backend
3. Install VcXsrv or Xming for X11 forwarding

#### X11 Display Setup
1. Install VcXsrv Windows X Server
2. Start XLaunch with these settings:
   - Display number: 0
   - Start no client
   - Disable access control
3. Set environment variable:
```powershell
$env:DISPLAY = "host.docker.internal:0.0"
```

#### Run Commands
```powershell
# Create X11 auth file
New-Item -Path "C:\temp\.docker.xauth" -ItemType File -Force

# Run with display forwarding
docker-compose up
```

### macOS Setup

#### Prerequisites
1. Install Docker Desktop for Mac
2. Install XQuartz for X11 support

#### X11 Display Setup
1. Install XQuartz:
```bash
brew install --cask xquartz
```

2. Start XQuartz and configure:
   - Go to XQuartz → Preferences → Security
   - Check "Allow connections from network clients"
   - Restart XQuartz

3. Set display forwarding:
```bash
# Allow X11 forwarding
xhost +localhost

# Set display environment
export DISPLAY=:0
```

#### Run Commands
```bash
# Create X11 auth file
touch /tmp/.docker.xauth

# Run the application
docker-compose up
```

### Linux Setup

#### Prerequisites
1. Install Docker Engine
2. Install Docker Compose
3. Add user to docker group

#### Setup Commands
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login again, then verify
docker --version
docker-compose --version
```

#### X11 Display Setup
```bash
# Allow X11 forwarding
xhost +local:docker

# Create X11 auth file
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
```

#### Run Commands
```bash
# Set display environment
export DISPLAY=$DISPLAY

# Run the application
docker-compose up
```

## Configuration Options

### Environment Variables
Create a `.env` file in the project root:

```env
# Display settings
DISPLAY=:0
QT_X11_NO_MITSHM=1

# Camera settings
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# Detection settings
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.4

# Performance settings
USE_GPU=true
NUM_THREADS=4
```

### Docker Compose Profiles

#### Basic Mode (Default)
```bash
docker-compose up
```

#### With Web Interface
```bash
docker-compose --profile web up
```

#### Development Mode
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Volume Mounts

The Docker setup includes several volume mounts:

- `./data:/app/data` - Persistent data storage
- `./logs:/app/logs` - Application logs
- `./models:/app/models` - AI model files
- `./config:/app/config` - Configuration files

## GPU Support

### NVIDIA GPU Support

1. Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Uncomment GPU section in docker-compose.yml:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Troubleshooting

### Camera Not Detected
```bash
# Check USB devices
lsusb | grep Intel

# Verify camera permissions
ls -la /dev/bus/usb

# Run with privileged mode
docker-compose up --privileged
```

### Display Issues
```bash
# Test X11 forwarding
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw alpine sh -c "apk add --no-cache xeyes && xeyes"

# Check X11 authentication
echo $DISPLAY
xauth list
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check available memory
docker system df

# Increase Docker memory limit
# (Docker Desktop → Settings → Resources → Memory)
```

## Advanced Configuration

### Custom Network
```yaml
networks:
  camera-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Health Checks
```yaml
healthcheck:
  test: ["CMD", "python3", "-c", "import pyrealsense2 as rs; print('OK')"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      memory: 2G
```

## Development Mode

For development, use the development compose file:

```bash
# Create development override
cp docker-compose.override.yml.example docker-compose.override.yml

# Run in development mode
docker-compose up
```

Development features:
- Hot code reloading
- Debug ports exposed
- Development dependencies installed
- Volume mounts for source code

## Production Deployment

### Build Production Image
```bash
# Build optimized production image
docker build -t camera-detection:prod -f Dockerfile.prod .

# Run production container
docker run -d \
  --name camera-detection-prod \
  --device=/dev/bus/usb \
  --privileged \
  -p 8080:8080 \
  camera-detection:prod
```

### Docker Swarm Deployment
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml camera-stack
```

## Monitoring and Logging

### View Logs
```bash
# Follow all logs
docker-compose logs -f

# Follow specific service
docker-compose logs -f camera-detection

# View last 100 lines
docker-compose logs --tail=100 camera-detection
```

### Monitoring
```bash
# Resource usage
docker-compose top

# Container stats
docker stats $(docker-compose ps -q)
```

## Backup and Restore

### Backup Data
```bash
# Backup all volumes
docker run --rm -v camera_detect_d415_camera-data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Backup models
docker run --rm -v camera_detect_d415_camera-models:/models -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /models .
```

### Restore Data
```bash
# Restore data volume
docker run --rm -v camera_detect_d415_camera-data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data

# Restore models volume
docker run --rm -v camera_detect_d415_camera-models:/models -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /models
```

## Support

For issues and support:
- Check the troubleshooting section above
- Review Docker logs: `docker-compose logs`
- Open an issue on GitHub: [Issues](https://github.com/SeifeldinShamseldin/camera_detect_d415/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.