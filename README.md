# Real-Time Face Recognition Turnstile System with Anti-Spoofing

An end-to-end, high-performance real-time face recognition turnstile pipeline featuring passive anti-spoofing and object tracking optimized for 30+ FPS execution.

## System Architecture

The pipeline consists of four modular components:
1. **Face Detection & Tracking**: YOLOv8 face detector paired with ByteTrack (`bytetrack.yaml`) for multi-object identity persistence across frames.
2. **Frame-Skipping Strategy**: Full inference (detection + embedding matching + liveness check) runs once every $N$ frames (default $N=3$). Bounding boxes and identities are cached and tracked in between to maintain real-time throughput.
3. **Passive Anti-Spoofing**: Multi-factor non-intrusive liveness verification:
   - **Laplacian Variance**: Measures image sharpness to reject blurry printed photos.
   - **2D FFT Spectral Analysis**: Evaluates high-frequency spectral energy ratios to detect Moiré noise patterns typical of mobile/LCD screen playbacks.
   - **YCrCb Chrominance Consistency**: Checks standard deviation of chrominance channels to differentiate real human skin tones from screen color compression.
4. **Face Matching Engine**: Cosine similarity evaluation against L2-normalized feature embeddings stored locally.

---

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME