# Video 2 Mesh CLI Tool

## Overview

The **Video 2 Mesh CLI Tool** is a command-line application designed to process images from videos and create 3D mesh models. This tool offers functionalities including extracting frames from videos, generating mesh models, and merging multiple mesh models into one. It's designed for users who need to efficiently convert video data into 3D models.

## Requirements

Before running the script, ensure that you meet the following requirements:

- **Python 3.9**:  
  Make sure Python 3.9 is installed on your system. You can check your Python version by running:
  ```bash
  python --version


If Python is not installed, download and install it from [python.org](https://www.python.org/).

### Required Python Libraries

The script depends on several Python libraries. Install them using `pip`:

```bash
pip install trimesh pyfiglet colorama   
```
### Docker:

Docker is required for running specific mesh processing commands. 
You can check if Docker is installed by running:
```bash
docker --version 
```

## Running the Script

Replace /path/to/chooseFrames/2 with the actual path to the src directory on your system.
Run the Script:
Execute the script using Python:
```bash
python CLITool.py
```

## Using the CLI Tool

After running the script, you will be presented with a welcome message and options to choose from. The tool provides several main commands:
Full Process: Converts two video files into one mesh model.
```bash
-full_process
```
Follow the prompts to input video paths and adjust processing parameters.
Video to Images: Extracts the best frames from a video to create a mesh model.
```bash
-video2images
```
Provide the video path and adjust the frame extraction parameters if needed.
Create Mesh: Creates a mesh model from a set of images located in a specified directory.

```bash
-create_mesh
```
Specify the directory containing the images, and the tool will generate the mesh model.
Merge Meshes: Merges two existing mesh models into one unified model.

```bash
-merge
```
Provide the paths to the two mesh models you want to merge.
Help: Displays help information and instructions for using the tool.

```bash
-help
```

## Docker Handling

The tool checks if Docker is running before performing mesh processing tasks. If Docker is not running, the tool will attempt to start Docker automatically on macOS or Windows. If Docker fails to start, the tool will exit.
## Troubleshooting
- Docker Issues:Ensure Docker is installed and running. The tool will attempt to start Docker automatically, but manual intervention may be required.
- File Paths: Make sure to use valid paths and ensure that the video files, images, and mesh models are correctly organized
