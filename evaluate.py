"""
evaluate.py — CLI for batch image quality evaluation.
Accepts a folder of generated images and (optionally) reference images.

Usage:
  python src/evaluate.py --generated data/samples/generated/ --real data/samples/real/
  python src/evaluate.py --generated data/samples/generated/  # no-reference only
"""
import argparse, json, os
from pathlib import Path
from metrics import evaluate_pair, sharpness, to_tensor

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def run_batch(generated_dir: str, real_dir: str = None) -> list[dict]:
    gen_paths = sorted(p for p in Path(generated_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    results = []

    for gen_path in gen_paths:
        entry = {'file': gen_path.name}
        if real_dir:
            real_path = Path(real_dir) / gen_path.name
            if real_path.exists():
                entry.update(evaluate_pair(str(real_path), str(gen_path)))
            else:
                print(f"  [warn] no matching real image for {gen_path.name}")
                entry['sharpness'] = sharpness(to_tensor(str(gen_path)))
        else:
            entry['sharpness'] = sharpness(to_tensor(str(gen_path)))
        results.append(entry)
        print(f"  {gen_path.name}: {entry}")

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Batch image quality evaluation')
    p.add_argument('--generated', required=True, help='Folder of generated images')
    p.add_argument('--real',      default=None,  help='Folder of reference images (optional)')
    p.add_argument('--output',    default='data/results/report.json')
    a = p.parse_args()

    print(f"\n[image-quality-evaluator]\nGenerated: {a.generated}")
    if a.real:
        print(f"Reference: {a.real}")
    print()

    results = run_batch(a.generated, a.real)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump(results, f, indent=2)

    avg_ssim = sum(r.get('ssim', 0) for r in results) / len(results) if results else 0
    avg_psnr = sum(r.get('psnr', 0) for r in results) / len(results) if results else 0
    print(f"\nSummary ({len(results)} images)")
    print(f"  Avg SSIM : {avg_ssim:.4f}")
    print(f"  Avg PSNR : {avg_psnr:.2f} dB")
    print(f"  Report   → {a.output}\n")
