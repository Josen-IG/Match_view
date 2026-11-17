"""Delimitación visual del césped en vídeos de cámara móvil.

La segmentación se usa como una barrera de seguridad junto con la homografía,
no como prueba de que una persona sea futbolista. No filtra el balón.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldMask:
    mask: object | None
    width: int
    height: int
    coverage: float = 0.0
    reason: str = "no_disponible"

    @property
    def reliable(self) -> bool:
        return self.mask is not None

    def contains(self, x: float, y: float) -> bool:
        if not self.reliable:
            return False
        import cv2
        px = int(round(x * self.width / max(1, self._original_width)))
        py = int(round(y * self.height / max(1, self._original_height)))
        if not 0 <= px < self.width or not 0 <= py < self.height:
            return False
        # Un punto de pie puede caer exactamente sobre una línea blanca.
        # Se acepta un pequeño vecindario que todavía sea césped.
        radius = max(2, self.width // 180)
        x0, x1 = max(0, px-radius), min(self.width, px+radius+1)
        y0, y1 = max(0, py-radius), min(self.height, py+radius+1)
        roi = self.mask[y0:y1, x0:x1]
        return bool(roi.size and cv2.countNonZero(roi) > roi.size * 0.20)

    _original_width: int = 0
    _original_height: int = 0


def field_mask_from_frame(frame, target_width: int = 480) -> FieldMask:
    """Encuentra la componente de césped conectada a la zona inferior.

    Funciona con césped natural o artificial verde. Si la confianza es
    insuficiente devuelve una máscara no fiable: no se inventa el campo.
    """
    import cv2
    import numpy as np

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return FieldMask(None, 0, 0, reason="imagen_invalida")
    original_h, original_w = frame.shape[:2]
    width = min(original_w, max(100, target_width))
    height = max(1, round(original_h * width / original_w))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Banda amplia de verdes. La selección por componente descarta los
    # árboles y carteles separados físicamente del terreno de juego.
    green = cv2.inRange(hsv, (28, 25, 20), (98, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    green = cv2.morphologyEx(
        green, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        green, connectivity=8)
    if count <= 1:
        return FieldMask(None, width, height, reason="sin_cesped")
    # Se exige conexión con la mitad inferior: esta región debe ser el
    # campo, no un árbol o una valla del fondo.
    lower_y = int(0.78 * height)
    valid = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 0.08 * width * height:
            continue
        bottom_presence = int(np.count_nonzero(labels[lower_y:, :] == component))
        if bottom_presence < 0.07 * width * (height - lower_y):
            continue
        valid.append((area, component))
    if not valid:
        return FieldMask(None, width, height, reason="cesped_no_confirmado")
    _, component = max(valid)
    selected = np.uint8(labels == component) * 255

    # Rellenar huecos de camisetas/porterías sin extender la máscara
    # fuera del perímetro del campo.
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return FieldMask(None, width, height, reason="sin_contorno")
    result = np.zeros_like(selected)
    cv2.drawContours(result, [max(contours, key=cv2.contourArea)],
                     -1, 255, thickness=cv2.FILLED)
    coverage = cv2.countNonZero(result) / (width * height)
    if coverage < 0.08 or coverage > 0.93:
        return FieldMask(None, width, height, coverage,
                         reason="cobertura_inverosimil")
    return FieldMask(result, width, height, round(coverage, 3),
                     "cesped_detectado", original_w, original_h)


def filter_pitch_people(detections, mapper, pitch_ok: bool, field_mask: FieldMask,
                        pitch_length=105.0, pitch_width=68.0,
                        strict: bool = True):
    """Acepta únicamente pies dentro del campo estimado, antes del tracking.

    Cuando la homografía y la máscara están disponibles, ambas deben
    coincidir. Si solo hay una fuente válida, se utiliza esa fuente.
    En modo estricto y sin ninguna fuente se omite la detección.
    """
    accepted = []
    counts = {"detected": len(detections), "accepted": 0,
              "rejected": 0, "unverified": 0,
              "homography_active": bool(pitch_ok),
              "mask_active": bool(field_mask.reliable)}
    for detection in detections:
        x0, _y0, x1, y1 = detection["caja"]
        x = (x0 + x1) / 2.0
        on_mask = field_mask.contains(x, y1) if field_mask.reliable else None
        pos = mapper.to_pitch(x, float(y1)) if pitch_ok else None
        on_pitch = (pos is not None and
                    0.0 <= pos[0] <= pitch_length and
                    0.0 <= pos[1] <= pitch_width) if pitch_ok else None
        checks = [c for c in (on_mask, on_pitch) if c is not None]
        if checks and all(checks):
            accepted.append(detection)
        elif not checks:
            counts["unverified"] += 1
            if not strict:
                accepted.append(detection)
        else:
            counts["rejected"] += 1
    counts["accepted"] = len(accepted)
    return accepted, counts
