"""Delimitación conservadora del césped para cámaras VEO móviles.

Se calcula una máscara por fotograma con OpenCV (no depende de coordenadas
fijas ni de la posición de la grada). Nunca se usa para seleccionar balón.
"""
from __future__ import annotations


def build_pitch_mask(frame):
    """Devuelve máscara uint8 (255 = superficie de juego) o None.

    Acepta hierba artificial oscura. Selecciona la mayor región verde
    conectada que ocupa el centro inferior de la imagen, para evitar
    confundirse con árboles u objetos verdes en la grada.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    if h < 40 or w < 40:
        return None
    small_w = min(w, 640)
    small_h = max(1, round(h * small_w / w))
    reduced = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    # Gama de verde (césped natural o sintético), excluyendo pavimento,
    # espectadores y los colores habituales de las fachadas.
    mask = cv2.inRange(hsv, (30, 22, 19), (100, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    candidates = []
    for label in range(1, n):
        x, y, width, height, area = stats[label]
        if area < small_h * small_w * 0.12:
            continue
        # La superficie debe tocar el cuarto inferior y ocupar el centro
        # de la imagen. Sin esas evidencias se declara desconocida.
        if y + height < small_h * .88:
            continue
        roi = labels[int(.72 * small_h):, int(.2 * small_w):int(.8 * small_w)]
        if not (roi == label).any():
            continue
        candidates.append((int(area), label))
    if not candidates:
        return None
    label = max(candidates)[1]
    pitch = np.uint8(labels == label) * 255
    # La base de los jugadores pisa el césped; cerrar huecos pequeños,
    # no dilatar fuera del borde (entrenadores en la banda).
    pitch = cv2.morphologyEx(pitch, cv2.MORPH_CLOSE, kernel, iterations=2)
    return pitch


def feet_on_pitch(mask, foot_x, foot_y, frame_width, frame_height):
    """Comprueba un pequeño vecindario del contacto de los pies.

    Se permite una tolerancia de ~0,6% del ancho para las líneas y
    detecciones con cajas desajustadas, pero no toda la altura de una
    persona (eso incluiría gente detrás de la portería).
    """
    if mask is None:
        return None
    h, w = mask.shape[:2]
    x = round(float(foot_x) * w / frame_width)
    y = round(float(foot_y) * h / frame_height)
    radius = max(1, round(w * .006))
    # Solo muestrear a la altura de los pies y un poco por debajo.
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - 1), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool((mask[y0:y1, x0:x1] > 0).any())
