GitHub broke the structure because code blocks were never closed with closing triple backticks (```), causing GitHub to view the rest of your text as one long code box. Additionally, tables on GitHub require vertical pipe characters (`|`) to format properly.

Select everything in your `README.md` file in Notepad, delete it, and paste this **exact** raw text:

```markdown
# Real-Time Face Recognition Turnstile System with Anti-Spoofing

An end-to-end, high-performance real-time face recognition turnstile pipeline featuring passive anti-spoofing and object tracking optimized for 30+ FPS execution.

## System Architecture

The pipeline consists of four modular components:
1. **Face Detection & Tracking**: YOLOv8 face detector paired with ByteTrack (`bytetrack.yaml`) for multi-object identity persistence across frames.
2. **Frame-Skipping Strategy**: Full inference (detection + embedding matching + liveness check) runs once every N frames (default N=3). Bounding boxes and identities are cached and tracked in between to maintain real-time throughput.
3. **Passive Anti-Spoofing**: Multi-factor non-intrusive liveness verification:
   - **Laplacian Variance**: Measures image sharpness to reject blurry printed photos.
   - **2D FFT Spectral Analysis**: Evaluates high-frequency spectral energy ratios to detect Moiré noise patterns typical of mobile/LCD screen playbacks.
   - **YCrCb Chrominance Consistency**: Checks standard deviation of chrominance channels to differentiate real human skin tones from screen color compression.
4. **Face Matching Engine**: Cosine similarity evaluation against L2-normalized feature embeddings stored locally.

---

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hx12x12/face-turnstile-antispoof.git
   cd face-turnstile-antispoof

```

2. **Getting Started**:
* **Install dependencies**:
```bash
pip install -r requirements.txt

```


* **Download pre-trained weights**:
```bash
python download_models.py

```


* **Run Turnstile Pipeline**:
* *Webcam:* `python main.py --source 0`
* *Video File:* `python main.py --source sample.mp4 --frame-skip 3`




3. **Interactive Controls**:
* **R Key**: Register current detected face into authorized disk database.
* **C Key**: Reset authorized user database (marks all detections as unauthorized).
* **Q / ESC Key**: Exit application.



---

### Key Trade-off Findings & Benchmarks

| Frame Skip (N) | Average FPS | Detection Latency (ms) | Tracking Stability | Spoof Rejection Rate |
| --- | --- | --- | --- | --- |
| N = 1 (Every frame) | ~14 - 18 FPS | ~58 ms | Excellent | 98.40% |
| N = 3 (Default) | ~34 - 42 FPS | ~22 ms | High | 97.80% |
| N = 5 | ~52 - 60 FPS | ~13 ms | Moderate (Fast motion jitter) | 95.10% |

* Setting `frame_skip = 3` provides the optimal balance, surpassing the target 30 FPS throughput requirement while maintaining accurate identity mapping via ByteTrack.
* FFT Moiré analysis successfully filters out >97% of 1080p screen playback spoof attempts without adding deep-learning inference overhead.

```