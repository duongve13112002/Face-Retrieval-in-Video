# Face Retrieval in Video

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Method](#method)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project focuses on face retrieval in video sequences. Face retrieval refers to the process of identifying and locating faces in video frames based on a query image. This system can be utilized for various applications such as video surveillance, content indexing, and personalized video retrieval.

## Features

- **Face Detection:** Identify and locate faces within video frames.
- **Face Recognition:** Recognize and match faces with a given query image.
- **Frame Extraction:** Efficient extraction of frames from video for processing.
- **Visualization:** Display bounding boxes around detected faces and highlight matches.

## Installation

To install and run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/duongve13112002/Face-Retrieval-in-Video.git
   cd Face-Retrieval-in-Video
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the face retrieval system, follow these steps:

1. Prepare your query image and video file.
2. Run the face retrieval script:
   ```bash
   python face_retrieval.py --query query_image.jpg --video input_video.mp4
   ```

3. The script will process the video and display frames with detected faces, highlighting those that match the query image.

### Command-Line Arguments

- `--query`: Path to the query image.
- `--video`: Path to the input video file.
- `--output`: (Optional) Path to save the output video with annotations.
- `--threshold`: (Optional) Confidence threshold for face recognition (default is 0.5).

## Method
- Step 1: Extract keyframes from the video:
  1. **Extract Candidate Frames**: Identify potential key frames by calculating differences between subsequent frames and selecting the most different frame within a window, reducing the number of frames to process.

  2. **Cluster Similar Candidate Frames**: Group similar frames by processing each frame (scaling, converting to greyscale, applying cosine transformation) and using HDBSCAN for clustering, which doesn't require specifying the number of clusters in advance.

  3. **Select Best Frames from Each Cluster**: Choose the best frame from each cluster based on brightness and image blur index (Laplacian score). Discard all other frames in the cluster as they contain similar content.
- Step 2: Use the facial recognition model in the DeepFace library to identify the position of faces in the keyframes and query images. Then, input these into deep learning models to extract features and use a similarity measurement function to compare the features of each face in each keyframe with the features of the faces in the query images. If they meet the threshold, save that keyframe.
 + To enhance performance with masked faces, use additional lineart images of the faces to improve the overall effectiveness of the method.

## Contributing

We welcome contributions to improve this project! To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the following resources and individuals:

- [Deepface](https://github.com/serengil/deepface) by Serengil
- [OpenCV](https://opencv.org/) by OpenCV team
---

For any questions or issues, please open an issue in the GitHub repository. Happy coding!
