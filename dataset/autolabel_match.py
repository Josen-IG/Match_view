"""Auto-etiquetar los fotogramas de UN partido (multi-partido).

Mismo pipeline que los 210 originales y los 58 de la 2ª mitad:
GemsLabs custom_model.pt (jugador+balón) + YOLO11s para reforzar el balón.
Clases: 0=football, 1=player. El árbitro (2) sigue pendiente de etiquetar a mano.

Uso:
    python3 autolabel_match.py --frames dataset_frames_sanmiguel \
        --labels dataset_frames_sanmiguel_labels

Usar el venv de match_view_analysis (tiene cv2 + ultralytics):
    ~/workspace/match_view_analysis/venv/bin/python autolabel_match.py ...
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
IMGSZ = 640
CONF_GEMS = 0.25
CONF_YOLO_BALL = 0.25


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="carpeta con los .jpg del partido")
    ap.add_argument("--labels", required=True, help="carpeta de salida para los .txt YOLO")
    args = ap.parse_args()
    os.makedirs(args.labels, exist_ok=True)

    gems = YOLO(os.path.join(BASE, "gemslabs", "custom_model.pt"))
    yolo = YOLO(os.path.join(BASE, "yolo11s.pt"))

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    n_ball = n_players = 0
    frames_with_ball = 0
    for i, fp in enumerate(files):
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        boxes = []
        r = gems.predict(img, imgsz=IMGSZ, conf=CONF_GEMS, verbose=False)[0]
        for b in r.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            boxes.append((cls, x1, y1, x2, y2, conf))
        r2 = yolo.predict(img, imgsz=IMGSZ, conf=CONF_YOLO_BALL, verbose=False)[0]
        for b in r2.boxes:
            if int(b.cls[0]) != 32:  # COCO 32 = sports ball
                continue
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dup = False
            for j, (c, a1, a2, a3, a4, cf) in enumerate(boxes):
                if c == 0 and iou((x1, y1, x2, y2), (a1, a2, a3, a4)) > 0.3:
                    dup = True
                    if conf > cf:
                        boxes[j] = (0, x1, y1, x2, y2, conf)
                    break
            if not dup:
                boxes.append((0, x1, y1, x2, y2, conf))
        stem = os.path.splitext(os.path.basename(fp))[0]
        lines = []
        has_ball = False
        for (cls, x1, y1, x2, y2, conf) in boxes:
            xc = ((x1 + x2) / 2) / w; yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w; bh = (y2 - y1) / h
            lines.append(f"{cls} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}")
            if cls == 0:
                n_ball += 1; has_ball = True
            else:
                n_players += 1
        frames_with_ball += 1 if has_ball else 0
        with open(os.path.join(args.labels, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)

    print(f"frames={len(files)} jugadores={n_players} balones={n_ball} "
          f"(balón en {frames_with_ball}/{len(files)} frames)")
    print(f"etiquetas en {args.labels}")


if __name__ == "__main__":
    main()
