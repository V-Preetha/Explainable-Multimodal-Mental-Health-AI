"""export face embeddings implementation for the curated submission repository."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.face.train import ConvNeXtEmotion, FERDataset  # noqa: E402

from src.numerical.common import root  # noqa: E402

R = root()
CHECKPOINT = R / "models" / "convnext_face_winner.pt"
SPLIT_PATH = R / "configs" / "face_split.json"
OUT_PATH = R / "outputs" / "embeddings" / "face.pt"


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]
    print(f"[FUSION] loading read-only face checkpoint: {CHECKPOINT} (val_macro_f1={checkpoint['val_macro_f1']:.4f})", flush=True)

    model = ConvNeXtEmotion(classes=len(classes), embedding_dim=256).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    export = {}
    started = time.time()
    for name in ("train", "validation", "test"):
        rows = [(Path(p), int(label)) for p, label in split[name]]
        loader = DataLoader(FERDataset(rows, image_size, train=False), batch_size=48, shuffle=False, num_workers=0)
        embeddings, labels, paths = [], [], []
        for step, (x, y, path_batch) in enumerate(loader, 1):
            z = model.embedding(model.backbone(x.to(device)))
            embeddings.append(z.cpu())
            labels.append(y)
            paths.extend(path_batch)
            if step % 20 == 0:
                print(f"[FUSION] face embeddings [{name}] batch {step}/{len(loader)}", flush=True)
        export[name] = {"embeddings": torch.cat(embeddings), "labels": torch.cat(labels), "paths": paths}
        print(f"[FUSION] face embeddings [{name}] done: {len(paths)} rows", flush=True)

    payload = {"embedding_dim": 256, "classes": classes, "checkpoint": str(CHECKPOINT), "extraction_seconds": time.time() - started}
    payload.update(export)
    torch.save(payload, OUT_PATH)
    print(f"[FUSION] saved face embeddings -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()



