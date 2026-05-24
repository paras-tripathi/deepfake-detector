# DeepShield — AI Deepfake Detection System

<div align="center">

![DeepShield Banner](https://img.shields.io/badge/DeepShield-Deepfake%20Detection-00dc96?style=for-the-badge&logo=shield&logoColor=white)

[![Live Demo](https://img.shields.io/badge/🔍%20Live%20Demo-Hugging%20Face-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Paras-tripathi/Deepshield)
[![Model](https://img.shields.io/badge/🤖%20Model-Hugging%20Face-FFD21E?style=for-the-badge)](https://huggingface.co/Paras-tripathi/deepfake-detector)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/paras-tripathi/deepfake-detector)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

**97.35% Accuracy · 20,000+ Training Frames · Production Deployed**

</div>

---

## Overview

DeepShield is a production-grade AI system that detects deepfake images and videos using a novel dual-branch deep learning architecture. It combines spatial feature extraction via EfficientNet-B4 with frequency domain analysis using FFT, achieving **97.35% validation accuracy** on the FaceForensics++ benchmark dataset.

The system provides not just a prediction, but also **visual explainability** through GradCAM heatmaps that highlight exactly which facial regions were manipulated — making it suitable for forensic, journalistic, and security applications.

---

## Live Demo

> Try it now — no setup required

**[🔍 Open DeepShield](https://huggingface.co/spaces/Paras-tripathi/Deepshield)**

Upload any image or video containing a human face. The system will:
- Detect and align the face using MTCNN
- Classify it as **REAL** or **FAKE** with a confidence score
- Generate a **GradCAM heatmap** showing manipulated regions

---

## Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | **97.35%** |
| Training Frames | **20,000** |
| Detection Time (Image) | **< 5 seconds** |
| Model Size | **75.8 MB** |
| Training Epochs | **5** |

---

## Architecture

```
Input Image/Video
       ↓
  Face Detection (MTCNN)
       ↓
  Face Alignment + Preprocessing
       ↓
  ┌─────────────────────────────┐
  │   Dual-Branch Extractor     │
  │  ┌──────────┐ ┌──────────┐  │
  │  │ Spatial  │ │Frequency │  │
  │  │EfficientB4│ │FFT Branch│  │
  │  └──────────┘ └──────────┘  │
  └─────────────────────────────┘
       ↓ Feature Fusion
  Sigmoid Classifier
       ↓
  REAL / FAKE + Confidence
       ↓
  GradCAM Heatmap (XAI)
```

**Why dual-branch?** Standard CNNs operating on pixel space miss frequency-domain artifacts that deepfake generators leave behind. The FFT branch captures these artifacts, significantly improving detection of high-quality deepfakes.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | PyTorch 2.2, EfficientNet-B4 (Transfer Learning) |
| Face Detection | MTCNN (facenet-pytorch) |
| Explainability | GradCAM |
| Frequency Analysis | PyTorch FFT |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Model Registry | Hugging Face Hub |
| Version Control | GitHub |
| Training | Google Colab (Tesla T4 GPU) |
| Dataset | FaceForensics++ (c40) |

---

## Dataset

Trained on **FaceForensics++** — the industry-standard benchmark for facial manipulation detection.

| Split | Count |
|-------|-------|
| Real frames | 10,000 |
| Fake frames | 10,000 |
| Total | 20,000 |
| Train / Val | 80% / 20% |

Fake categories covered: **Deepfakes**, **Face2Face**, **FaceSwap**, **NeuralTextures**, **FaceShifter**, **DeepFakeDetection (Google)**

---

## Project Structure

```
deepfake-detector/
├── src/
│   ├── face_detector.py      # MTCNN face detection + alignment
│   ├── preprocessor.py       # Normalization + tensor conversion
│   ├── model.py              # EfficientNet-B4 + FFT dual-branch
│   ├── gradcam.py            # GradCAM explainability
│   └── config_reader.py      # Centralized config management
├── pipeline/
│   ├── base_pipeline.py      # Abstract base pipeline
│   ├── image_pipeline.py     # End-to-end image detection
│   └── video_pipeline.py     # Frame-sampled video detection
├── app/
│   ├── fastapi_app.py        # REST API backend
│   └── deepshield_app.html   # Web UI
├── configs/
│   └── config.yaml           # All hyperparameters centralized
├── Dockerfile                # Container for deployment
└── requirements.txt
```

---

## Key Features

**Modular Architecture** — Each component is independent. Adding or removing features (audio detection, batch processing, PDF reports) requires only a new module and a config flag — no existing code needs modification.

**Explainability (XAI)** — GradCAM generates heatmaps showing which facial regions influenced the model's decision. This makes the system transparent and suitable for forensic use cases.

**Dual-Branch Model** — Combines spatial features from EfficientNet-B4 with frequency-domain artifacts detected via FFT. This approach mirrors techniques used in winning solutions of the DFDC competition.

**Production-Ready** — Dockerized FastAPI backend, model weights hosted on Hugging Face Hub, and a polished web UI — not a Jupyter notebook demo.

---

## Local Setup

```bash
# Clone repository
git clone https://github.com/paras-tripathi/deepfake-detector.git
cd deepfake-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.fastapi_app:app --reload

# Open browser
# http://localhost:8000
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | System status |
| `/predict/image` | POST | Detect deepfake in image |
| `/predict/video` | POST | Detect deepfake in video |

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@face.jpg"
```

**Example response:**
```json
{
  "label": "FAKE",
  "confidence": 87.3,
  "faces_detected": 1,
  "artifact_score": "High",
  "heatmap": "<base64_encoded_image>"
}
```

---

## References

- [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971) — Rossler et al., ICCV 2019
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) — Tan & Le, ICML 2019
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) — Selvaraju et al., ICCV 2017
- [DeepFake Detection Challenge (DFDC)](https://arxiv.org/abs/2006.07397) — Dolhansky et al., 2020

---

## Author

**Paras Tripathi**

[![GitHub](https://img.shields.io/badge/GitHub-paras--tripathi-181717?style=flat&logo=github)](https://github.com/paras-tripathi)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Paras--tripathi-FFD21E?style=flat)](https://huggingface.co/Paras-tripathi)

---

<div align="center">
<sub>Built with PyTorch · Deployed on Hugging Face · Trained on FaceForensics++</sub>
</div>
