## Project Structure

The repository is organized into the following main directories:

### Core Components

- **Server_side/**: Contains the server implementation
  - `Server.py`: Main server application for handling video processing and WebSocket connections
  - `capture3.py`: Video capture and processing module
  - `frames/`: Directory for storing processed video frames

### Model Components

- **Final_model/**: Contains the production-ready model
  - `model_version6.pth`: The trained ResNet18 model weights

### Development and Training

- **training_folder/**: Contains model training scripts and configurations
- **resnet_models/**: Development versions of the ResNet model
- **Yolo_models/**: Previous YOLO-based implementations
- **Datasets/**: Training and validation data
- **Evaluation/**: Model evaluation metrics and results
- **Model_results/**: Performance analysis and results

### Supporting Components

- **Additional_scripts/**: Utility scripts and helper functions
- **grayscale_mask/**: Image processing tools for mask generation

## Technology Stack

- Python
- PyTorch and TorchVision for deep learning
- OpenCV for computer vision
- WebSocket for real-time communication

## Model Information

The system uses a ResNet18-based model trained on parking space detection. The model processes video frames in real-time to identify available parking spaces.

## Deployment

The application can be started using the `start-server.sh` script, which sets up the necessary environment and runs the server.

## Development Notes

- The project has evolved from an initial YOLO-based implementation to the current ResNet18-based solution
- The current model (version6) provides improved accuracy and real-time performance
- The server implementation supports WebSocket connections for real-time updates
