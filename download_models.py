import os
import urllib.request
from ultralytics import YOLO


def download_yolo_face_model(output_dir: str = "models") -> str:
    """
    Ensures a valid YOLO detection model weight file exists inside the models/ folder.
    Downloads from a reliable HuggingFace mirror with User-Agent headers, with fallback to standard YOLOv8.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "yolov8n-face.pt")

    if os.path.exists(target_path):
        return target_path

    # Mirror URL for YOLOv8 Face weights
    download_url = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"

    print(f"[*] Downloading YOLOv8 face detection model into '{target_path}'...")
    try:
        request = urllib.request.Request(
            download_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(request) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        print("[+] Download complete.")
        return target_path
    except Exception as error:
        print(f"[-] Mirror download failed ({error}).")
        print("[!] Falling back to standard Ultralytics YOLOv8 weights...")
        
        fallback_path = os.path.join(output_dir, "yolov8n.pt")
        # Initialize YOLO which triggers native download if weights are missing
        _ = YOLO("yolov8n.pt")
        return fallback_path


if __name__ == "__main__":
    download_yolo_face_model()