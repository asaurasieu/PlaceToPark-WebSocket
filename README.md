# Parking Project

A real-time parking space detection system powered by AI. The system processes video feeds to identify available parking spaces, making it easier to find parking in busy areas.

## Project Overview

The project is structured into several key components:

### Core Components

- **Server_side/**: The main server implementation with WebSocket connections
  - `server_react.py`: Primary server handling real-time communication
  - `server_in_web.py`: Alternative server implementation
  - `frames/`: Storage for processed video frames
  - `AWS_app.py`: AWS cloud integration
- **Final_models/**: Production-ready models, featuring the optimized ResNet18 implementation
- **Datasets/**: Training and validation data for model development
- **resnet_models/**: Various iterations of the ResNet model
- **Yolo_models/**: Initial YOLO implementation (superseded by ResNet18)

### Supporting Components

- **templates/**: Web interface templates
- **Additional_scripts/**: Utility scripts and helper functions
- **grayscale_mask/**: Image processing and mask generation tools
- **Model_results/**: Model evaluation metrics and results
- **training_folder/**: Model training configurations and scripts

## Technology Stack

The project leverages modern tools and frameworks:

- Custom Python WebSocket server for real-time processing
- WebSocket protocol for live communication
- PyTorch and TorchVision for deep learning implementation
- OpenCV for computer vision tasks
- Pillow for image processing
- yt-dlp for video stream handling
- Flask for web framework
- Jinja2 for template rendering
- NumPy for scientific computing
- Requests for HTTP communication

## Development Journey

The system evolved through several iterations:

1. **Initial Version**: YOLO-based detection system
2. **Current Version**: ResNet18-based classification
   - Enhanced accuracy and performance
   - Optimized for real-time processing
   - Improved resource efficiency

## Setup Guide

To get started with the project:

1. Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the server:

```bash
cd Server_side
python server_react.py
```

4. Visualization Options:
   - **React App**: Access the real-time visualization through the React application
   - **Website Template**: Alternatively, use the HTML template by running `server_in_web.py`

## System Architecture

The WebSocket server provides:

- Real-time video stream processing
- Live parking space status updates
- Client-server communication
- Frame-by-frame parking space analysis

## Model Training

The models were trained using data from the `Datasets/` directory. Training configurations and processes are documented in the `training_folder/` directory.

## License
For questions or contributions, please open an issue or submit a pull request.
