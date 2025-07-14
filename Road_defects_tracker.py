import os
import numpy as np
from ultralytics import YOLO
import supervision as sv

# ─── Paths & Model ─────────────────────────────────────────────────────────────
HOME = os.getcwd()
SOURCE_VIDEO_PATH = os.path.join(HOME, "input_video.mp4")  # Update with your video path
TARGET_VIDEO_PATH = os.path.join(HOME, "result_road_defects.mp4")
model = YOLO("Road_defects_US_India.pt")  # Make sure this file is in your project root

# ─── Class Filtering ───────────────────────────────────────────────────────────
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack', 'Pothole']
SELECTED_CLASS_IDS = [
    {v: k for k, v in CLASS_NAMES_DICT.items()}[name]
    for name in SELECTED_CLASS_NAMES
]

# ─── Tracking Setup ───────────────────────────────────────────────────────────
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
fps = video_info.fps
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=fps,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# ─── Annotators ────────────────────────────────────────────────────────────────
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
line_zone = sv.LineZone(start=sv.Point(50, 1500), end=sv.Point(3840-50, 1500))
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Process each frame for detection and tracking.
    
    Args:
        frame: Input video frame
        index: Frame index
        
    Returns:
        Annotated frame with bounding boxes and tracking information
    """
    results = model(frame, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(results)
    dets = dets[np.isin(dets.class_id, SELECTED_CLASS_IDS)]
    dets = byte_tracker.update_with_detections(dets)
    
    # Annotate frame
    ann = frame.copy()
    ann = box_annotator.annotate(scene=ann, detections=dets)
    
    labels = [
        f"#{tid} {CLASS_NAMES_DICT[cid]} {conf:0.2f}"
        for conf, cid, tid in zip(dets.confidence, dets.class_id, dets.tracker_id)
    ]
    ann = label_annotator.annotate(scene=ann, detections=dets, labels=labels)
    
    line_zone.trigger(dets)
    ann = line_zone_annotator.annotate(ann, line_counter=line_zone)
    
    return ann


def main():
    """Main function to run road defect detection and tracking."""
    if not os.path.exists(SOURCE_VIDEO_PATH):
        print(f"Error: Video file not found at {SOURCE_VIDEO_PATH}")
        print("Please update SOURCE_VIDEO_PATH with the correct path to your video file.")
        return
    
    if not os.path.exists("Road_defects_US_India.pt"):
        print("Error: Model file 'Road_defects_US_India.pt' not found.")
        print("Please ensure the trained model file is in the project root directory.")
        return
    
    print(f"Processing video: {SOURCE_VIDEO_PATH}")
    print(f"Output will be saved to: {TARGET_VIDEO_PATH}")
    print(f"Detecting classes: {SELECTED_CLASS_NAMES}")
    
    # ─── Run & Export ─────────────────────────────────────────────────────────────
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
    
    print(f"✓ Annotated video saved: {TARGET_VIDEO_PATH}")


if __name__ == "__main__":
    main()