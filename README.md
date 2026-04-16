# Image-quality-evaluator (Generative AI)

> No-reference and full-reference image quality evaluation for generative model outputs.

Evaluates generated images using perceptual and statistical metrics — SSIM, PSNR, sharpness, and a simplified FID — with a ResNet-50 feature backbone. Designed to slot into generative model training pipelines as an automated evaluation step.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/image-quality-evaluator.git
cd image-quality-evaluator
pip install -r requirements.txt

# Full-reference evaluation (generated vs real)
python src/evaluate.py \
  --generated data/samples/generated/ \
  --real      data/samples/real/ \
  --output    data/results/report.json

# No-reference (generated images only)
python src/evaluate.py --generated data/samples/generated/
```

---

## Metrics

| Metric | Type | What it measures |
|--------|------|-----------------|
| SSIM | Full-reference | Structural similarity to reference image |
| PSNR | Full-reference | Pixel-level fidelity (dB) |
| Sharpness | No-reference | Laplacian variance — perceived detail |
| FID (feature-level) | Full-reference | Distribution distance between real/generated embeddings |

---

## Sample output

```json
[
  {
    "file": "img_001.jpg",
    "psnr": 24.31,
    "ssim": 0.7812,
    "sharpness": 0.0043
  },
  {
    "file": "img_002.jpg",
    "psnr": 21.88,
    "ssim": 0.7201,
    "sharpness": 0.0031
  }
]

Summary (2 images)
  Avg SSIM : 0.7507
  Avg PSNR : 23.10 dB
```

---

## Architecture

```
Generated image ──► ResNet-50 backbone (frozen) ──► Feature embeddings
                                                        │
                                     ┌──────────────────┤
                                     │                  │
                               SSIM / PSNR         FID score
                             (pixel-level)     (distribution-level)
```

---

## Project structure

```
image-quality-evaluator/
├── src/
│   ├── model.py      # ResNet-50 feature extractor + quality scorer head
│   ├── metrics.py    # SSIM, PSNR, sharpness, FID implementations
│   └── evaluate.py   # Batch evaluation CLI
├── data/
│   ├── samples/
│   │   ├── generated/   # Images to evaluate
│   │   └── real/        # Reference images (optional)
│   └── results/         # JSON reports (gitignored)
├── requirements.txt
└── README.md
```

---

## Skills demonstrated

| ML competency | Implementation |
|---|---|
| Model evaluation | SSIM, PSNR, sharpness, FID |
| Deep learning (CV) | ResNet-50 feature extraction |
| Generative AI | Evaluation pipeline for image generation |
| Data processing | Batch image loading, normalisation |
| Production mindset | CLI, JSON reporting, folder-level batch runs |

---

## Roadmap

- [ ] LPIPS (learned perceptual similarity) integration
- [ ] Wandb logging for training-time evaluation
- [ ] Clip-score for text-to-image alignment
- [ ] Dashboard visualisation of metric distributions

---

## License

MIT
