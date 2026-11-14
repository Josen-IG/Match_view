"""Extraer fotogramas de un partido para el dataset propio (multi-partido).

Uso:
    python3 extract_frames_match.py --video /ruta/partido.mp4 --match sanmiguel
    python3 extract_frames_match.py --video /ruta/partido.mp4 --match alc_elc --every 45 --start 600

- 1 fotograma cada --every segundos (30 por defecto, como el Cirbonero).
- Nombres con prefijo del partido (<match>_0000.jpg, ...) para que NUNCA
  choquen con los de otro partido al fusionar (merge_dataset_multi.py lo exige).
- --start salta los primeros segundos (calentamiento, por ejemplo).
- Salida: dataset_frames_<match>/ junto a este script.
"""
import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="ruta al vídeo del partido")
    ap.add_argument("--match", required=True,
                    help="nombre corto del partido, ej. sanmiguel, alc_elc (sin espacios)")
    ap.add_argument("--every", type=int, default=30,
                    help="1 fotograma cada N segundos (defecto 30)")
    ap.add_argument("--start", type=int, default=0,
                    help="saltar los primeros N segundos (defecto 0)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"no existe el vídeo: {args.video}")
        sys.exit(1)
    if " " in args.match or "/" in args.match:
        print("el nombre del partido no puede tener espacios ni /")
        sys.exit(1)

    out = os.path.join(BASE, f"dataset_frames_{args.match}")
    os.makedirs(out, exist_ok=True)
    existing = len([f for f in os.listdir(out) if f.endswith(".jpg")])
    if existing:
        print(f"AVISO: {out} ya tiene {existing} fotogramas; se sobrescriben los del mismo nombre")

    cmd = ["ffmpeg", "-y", "-ss", str(args.start), "-i", args.video,
           "-vf", f"fps=1/{args.every}",
           os.path.join(out, f"{args.match}_%04d.jpg")]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)
    n = len([f for f in os.listdir(out) if f.endswith(".jpg")])
    print(f"listo: {n} fotogramas en {out} (1 cada {args.every} s, desde {args.start} s)")


if __name__ == "__main__":
    main()
