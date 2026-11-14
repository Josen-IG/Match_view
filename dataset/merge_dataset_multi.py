"""Fusionar los fotogramas de VARIOS partidos en el dataset propio (multi-partido).

Cada partido es una fuente: (nombre, carpeta de fotogramas, carpetas de etiquetas).
Para añadir un partido nuevo, añade UNA línea a SOURCES y ejecuta el script:
los fotogramas nuevos entran con split 80/20 estratificado POR PARTIDO
(cada partido aporta a train y a val), sin fuga (cada fotograma va a un solo split).

Reglas para no romper lo que ya hay:
- Los 268 fotogramas que ya están en el repo CONSERVAN su split train/val
  actual (se lee de dataset/images/train|val), así que re-ejecutar el script
  sin partidos nuevos no cambia nada.
- El nombre del fotograma (stem) debe ser ÚNICO en todo el dataset: el script
  falla si dos partidos traen el mismo nombre. extract_frames_match.py ya
  pone prefijo de partido (<match>_0000.jpg), así que no debería pasar.
- Clases: 0=football, 1=player (2=referee pendiente de etiquetar a mano).

Uso:
    python3 merge_dataset_multi.py --dry-run        # solo comprobar y ver la tabla
    python3 merge_dataset_multi.py                  # reescribir dataset/ del repo
    python3 merge_dataset_multi.py --out /tmp/ds    # probar en otra carpeta
"""
import argparse
import glob
import os
import random
import shutil
from collections import Counter, defaultdict

ANAL = "/home/hatch/workspace/match_view_analysis"
REPO = "/home/hatch/workspace/match_view_video_v2/dataset"
SEED = 42

# Añadir aquí cada partido nuevo: (nombre, carpeta de imágenes, [carpetas de etiquetas])
SOURCES = [
    ("cirbonero_210", os.path.join(ANAL, "dataset_frames"),
     [os.path.join(ANAL, "dataset", "labels", "train"),
      os.path.join(ANAL, "dataset", "labels", "val")]),
    ("cirbonero_2mitad", os.path.join(ANAL, "dataset_frames_2mitad"),
     [os.path.join(ANAL, "dataset_frames_2mitad_labels")]),
    # Palencia (palencia_bembibre) SUSTITUIDO 2026-11-14: su subida a Drive
    # falló repetidamente (error de Google). Lo reemplazaba selaya_cayon
    # (Selaya FC vs CD Cayón, noche, 3ª Fed, Grupo III), aprobado por el
    # usuario — pero Selaya TAMBIÉN se descartó el 2026-11-14 por decisión
    # del usuario: su subida a Drive se atasca (mecanismo roto para archivos
    # grandes desde ~18:45) y con 16 vídeos ya en Drive hay de sobra.
    # (alcala_intercity fue un candidato propio no aprobado: su vídeo y 260
    # fotogramas quedan en disco pero NO entran.)
    # ("selaya_cayon", os.path.join(ANAL, "dataset_frames_selaya_cayon"),
    #  [os.path.join(ANAL, "dataset_frames_selaya_cayon_labels")]),
    # Fase 2 (2026-11-14): 16 partidos de YouTube aprobados por el usuario.
    # Palencia y Selaya NO entran (ver comentario arriba).
    ("sevillaat_betisdep", os.path.join(ANAL, "dataset_frames_sevillaat_betisdep"),
     [os.path.join(ANAL, "dataset_frames_sevillaat_betisdep_labels")]),
    ("talavera_zamora", os.path.join(ANAL, "dataset_frames_talavera_zamora"),
     [os.path.join(ANAL, "dataset_frames_talavera_zamora_labels")]),
    ("antequera_castilla", os.path.join(ANAL, "dataset_frames_antequera_castilla"),
     [os.path.join(ANAL, "dataset_frames_antequera_castilla_labels")]),
    ("castilla_aviles", os.path.join(ANAL, "dataset_frames_castilla_aviles"),
     [os.path.join(ANAL, "dataset_frames_castilla_aviles_labels")]),
    ("alcala_fuenlabrada", os.path.join(ANAL, "dataset_frames_alcala_fuenlabrada"),
     [os.path.join(ANAL, "dataset_frames_alcala_fuenlabrada_labels")]),
    ("alcala_madridc", os.path.join(ANAL, "dataset_frames_alcala_madridc"),
     [os.path.join(ANAL, "dataset_frames_alcala_madridc_labels")]),
    ("badajoz_tenerifeb", os.path.join(ANAL, "dataset_frames_badajoz_tenerifeb"),
     [os.path.join(ANAL, "dataset_frames_badajoz_tenerifeb_labels")]),
    ("recreativo_antoniano", os.path.join(ANAL, "dataset_frames_recreativo_antoniano"),
     [os.path.join(ANAL, "dataset_frames_recreativo_antoniano_labels")]),
    ("betisc_conil", os.path.join(ANAL, "dataset_frames_betisc_conil"),
     [os.path.join(ANAL, "dataset_frames_betisc_conil_labels")]),
    ("betisc_sevillac", os.path.join(ANAL, "dataset_frames_betisc_sevillac"),
     [os.path.join(ANAL, "dataset_frames_betisc_sevillac_labels")]),
    ("espanol_villarrealc", os.path.join(ANAL, "dataset_frames_espanol_villarrealc"),
     [os.path.join(ANAL, "dataset_frames_espanol_villarrealc_labels")]),
    ("galapagar_atleticoc", os.path.join(ANAL, "dataset_frames_galapagar_atleticoc"),
     [os.path.join(ANAL, "dataset_frames_galapagar_atleticoc_labels")]),
    ("cornella_mollerussa", os.path.join(ANAL, "dataset_frames_cornella_mollerussa"),
     [os.path.join(ANAL, "dataset_frames_cornella_mollerussa_labels")]),
    ("sanjose_tomelloso", os.path.join(ANAL, "dataset_frames_sanjose_tomelloso"),
     [os.path.join(ANAL, "dataset_frames_sanjose_tomelloso_labels")]),
    ("hellin_tomelloso", os.path.join(ANAL, "dataset_frames_hellin_tomelloso"),
     [os.path.join(ANAL, "dataset_frames_hellin_tomelloso_labels")]),
    ("casetas_ontinena", os.path.join(ANAL, "dataset_frames_casetas_ontinena"),
     [os.path.join(ANAL, "dataset_frames_casetas_ontinena_labels")]),
    # ("sanmiguel", os.path.join(ANAL, "dataset_frames_sanmiguel"),
    #  [os.path.join(ANAL, "dataset_frames_sanmiguel_labels")]),
    # ("alc_elc", os.path.join(ANAL, "dataset_frames_alc_elc"),
    #  [os.path.join(ANAL, "dataset_frames_alc_elc_labels")]),
]


def count_classes(lbl_path):
    nb = np_ = 0
    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if int(line.split()[0]) == 0:
                nb += 1
            else:
                np_ += 1
    return nb, np_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=REPO, help="carpeta dataset de destino")
    args = ap.parse_args()
    random.seed(SEED)

    # Split actual del repo: los stems que ya existen conservan su split.
    current_split = {}
    for split in ("train", "val"):
        d = os.path.join(REPO, "images", split)
        for f in glob.glob(os.path.join(d, "*.jpg")):
            current_split[os.path.splitext(os.path.basename(f))[0]] = split

    items = []  # (stem, img, lbl, match)
    seen = {}
    for name, frames_dir, label_dirs in SOURCES:
        stems = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(frames_dir, "*.jpg")))
        assert stems, f"sin fotogramas: {frames_dir}"
        for stem in stems:
            if stem in seen:
                raise SystemExit(
                    f"COLISIÓN de nombre: '{stem}' está en '{seen[stem]}' y en '{name}'. "
                    "Renombra los fotogramas del partido nuevo con su prefijo.")
            seen[stem] = name
            lbl = None
            for d in label_dirs:
                cand = os.path.join(d, stem + ".txt")
                if os.path.exists(cand):
                    lbl = cand
                    break
            assert lbl, f"sin etiqueta para {stem} ({name})"
            items.append((stem, os.path.join(frames_dir, stem + ".jpg"), lbl, name))
    print(f"total imágenes: {len(items)}")
    print("por partido:", dict(Counter(m for _, _, _, m in items)))

    by_match = defaultdict(list)
    for it in items:
        by_match[it[3]].append(it)
    train, val = [], []
    for match, lst in sorted(by_match.items()):
        old = [it for it in lst if it[0] in current_split]
        new = [it for it in lst if it[0] not in current_split]
        for it in old:
            (train if current_split[it[0]] == "train" else val).append(it)
        if new:
            random.shuffle(new)
            n_val = max(1, round(0.2 * len(new))) if len(new) >= 5 else 0
            val.extend(new[:n_val])
            train.extend(new[n_val:])
            print(f"{match}: {len(new)} nuevos -> train {len(new) - n_val}, val {n_val} "
                  f"(+ {len(old)} ya en repo, split conservado)")
        else:
            print(f"{match}: {len(old)} en repo, split conservado "
                  f"(train {sum(1 for it in old if current_split[it[0]] == 'train')}, "
                  f"val {sum(1 for it in old if current_split[it[0]] == 'val')})")

    # Sin fuga: ningún stem en ambos splits
    assert not (set(it[0] for it in train) & set(it[0] for it in val)), "FUGA train/val"
    print(f"TOTAL: train {len(train)}, val {len(val)}")

    # Tabla por partido (para el README del dataset)
    print("\n| Partido | Imágenes | Train | Val | Jugadores | Balones | Frames con balón |")
    print("|---|---|---|---|---|---|---|")
    for match in sorted(by_match):
        tb = tp = fb = 0
        for split in (train, val):
            for (stem, img, lbl, m) in split:
                if m != match:
                    continue
                b, p = count_classes(lbl)
                tb += b; tp += p
                fb += 1 if b > 0 else 0
        n = len(by_match[match])
        ntr = sum(1 for it in train if it[3] == match)
        nva = sum(1 for it in val if it[3] == match)
        print(f"| {match} | {n} | {ntr} | {nva} | {tp} | {tb} | {fb} |")
    tb = sum(count_classes(lbl)[0] for _, _, lbl, _ in train + val)
    tp = sum(count_classes(lbl)[1] for _, _, lbl, _ in train + val)
    fb = sum(1 for _, _, lbl, _ in train + val if count_classes(lbl)[0] > 0)
    print(f"| **Total** | **{len(items)}** | **{len(train)}** | **{len(val)}** | "
          f"**{tp}** | **{tb}** | **{fb}** |")

    if args.dry_run:
        print("\ndry-run: no se ha escrito nada.")
        return
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = os.path.join(args.out, sub, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        for (stem, img, lbl, src) in split:
            shutil.copy(img, os.path.join(args.out, "images", name, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(args.out, "labels", name, os.path.basename(lbl)))
    print(f"\ndataset reescrito en {args.out}")
    print("Después: rehacer el zip de imágenes para Drive y subir labels + data.yaml al repo.")


if __name__ == "__main__":
    main()
