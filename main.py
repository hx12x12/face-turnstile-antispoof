import argparse
import os
import urllib.request
from download_models import download_yolo_face_model
from src.turnstile_pipeline import RealTimeTurnstilePipeline


SAMPLE_VIDEO_URL = (
    "https://raw.githubusercontent.com/intel-iot-devkit/"
    "sample-videos/master/head-pose-face-detection-female.mp4"
)


def ensure_video_source(source_str: str) -> str:
    """
    Checks if source_str is a valid camera index or existing file.
    If the file does not exist, downloads a sample face video automatically.
    """
    if source_str.isdigit():
        return source_str

    if not os.path.exists(source_str):
        print(f"[!] Video file '{source_str}' not found.")
        target_path = "sample.mp4"
        if not os.path.exists(target_path):
            print(f"[*] Downloading sample face detection video to '{target_path}'...")
            urllib.request.urlretrieve(SAMPLE_VIDEO_URL, target_path)
            print("[+] Sample video downloaded successfully.")
        return target_path

    return source_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Face Turnstile System with Anti-Spoofing")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. '0') or video file path")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO weights file")
    parser.add_argument("--frame-skip", type=int, default=3, help="Inference skip interval N frames")
    args = parser.parse_args()

    # Resolve model path automatically if omitted
    model_path = args.model if args.model else download_yolo_face_model()

    # Validate source file or download fallback sample video
    validated_source = ensure_video_source(args.source)

    # Convert camera index string to integer if applicable
    source = int(validated_source) if validated_source.isdigit() else validated_source

    pipeline = RealTimeTurnstilePipeline(
        yolo_model_path=model_path,
        frame_skip=args.frame_skip
    )
    pipeline.run_video_stream(source=source)


if __name__ == "__main__":
    main()