# Road Defect Detection and Tracking

A computer vision system for detecting and tracking road defects in video streams using YOLOv11, ByteTrack and Supervison library. The system is trained on the RDD2022 dataset's Indian road images and can detect four types of road defects commonly found on Indian roads.

## Features

- **Real-time Detection**: Detects four types of road defects:
  - Longitudinal Crack
  - Transverse Crack
  - Alligator Crack
  - Pothole
- **Object Tracking**: Uses ByteTrack for consistent tracking across frames
- **Video Processing**: Processes video files and outputs annotated results
- **Line Zone Counting**: Includes line zone functionality for defect counting
- **Customizable Annotations**: Bounding boxes, labels, and confidence scores

## Model

The system uses a YOLOv11 model trained on the RDD2022 (Road Damage Dataset 2022) dataset, specifically focusing on Indian road conditions. The model file `Road_defects_US_India.pt` should be placed in the project root directory.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Road-Defects-Detection-and-Tracking.git
cd Road-Defects-Detection-and-Tracking
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your trained model file `Road_defects_US_India.pt` in the project root directory
2. Update the video path in the script:
   ```python
   SOURCE_VIDEO_PATH = "path/to/your/video.mp4"
   ```
3. Run the detection and tracking:
   ```bash
   python road_defect_tracker.py
   ```

The output video with bounding boxes and tracking information will be saved as `result_road_defects.mp4`.

## Configuration

### Tracking Parameters
- `track_activation_threshold`: 0.25 - Minimum confidence for track activation
- `lost_track_buffer`: 30 - Frames to keep lost tracks
- `minimum_matching_threshold`: 0.8 - Minimum IoU for track matching
- `minimum_consecutive_frames`: 3 - Minimum frames for track confirmation

### Detection Classes
The system is configured to detect the following road defects:
- Longitudinal Crack
- Transverse Crack
- Alligator Crack
- Pothole

## File Structure

```
road-defect-detection-tracking/
├── road_defect_tracker.py     # Main detection and tracking script
├── Road_defects_US_India.pt   # Trained YOLOv8 model (not included)
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore file
└── setup.py                 # Package setup file
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for better performance)
- Sufficient RAM for video processing

## Model Training

The model was trained on the RDD2022 dataset focusing on Indian road conditions. If you want to retrain or fine-tune the model, you'll need:
- RDD2022 dataset
- YOLOv8 training pipeline
- Appropriate hardware for training

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- RDD2022 dataset creators
- Ultralytics for YOLOv8
- Supervision library for tracking utilities
- ByteTrack for object tracking algorithm

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{road-defect-detection-tracking,
  title={Road Defect Detection and Tracking},
  author={Waleed Ijaz},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://https://github.com/Waleed-Ijaz/Road-Defects-Detection-and-Tracking}}
}
```

## Support

For questions and support, please open an issue on GitHub.
