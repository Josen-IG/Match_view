"""Availability and fallback rules, not unvalidated accuracy rankings."""
import importlib.util


def available(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def record_status(status, name, state, detail, progress=None):
    status[name] = {"state": state, "detail": str(detail)}
    if progress:
        progress(f"{name}: {detail}", 0.02)


def automatic_plan(progress=None):
    status = {"policy": {"state": "rules", "detail":
              "Selección por disponibilidad y respaldo. Sin porcentajes de acierto validados."}}
    hub = available("huggingface_hub")
    settings = {"ball_model": "hf" if hub else None,
                "rfdetr_model": "hf" if hub and available("rfdetr") else None,
                "pitch_model": "hf32" if hub else None,
                "actions_enabled": False, "gamestate_enabled": False}
    for name, enabled in (("yolov9", settings["ball_model"]),
                          ("rfdetr", settings["rfdetr_model"]),
                          ("pitch32", settings["pitch_model"])):
        record_status(status, name, "planned" if enabled else "unavailable",
                      "Disponible; se intentará cargar." if enabled else
                      "Faltan dependencias; se continúa con los componentes disponibles.")
    from model_jobs import validate_model_environment
    for name, option in (("actions", "actions_enabled"), ("gamestate", "gamestate_enabled")):
        try:
            validate_model_environment(name)
        except Exception as error:
            record_status(status, name, "unavailable", str(error))
        else:
            settings[option] = True
            record_status(status, name, "planned", "Entorno independiente disponible.")
    if progress:
        missing = [name for name, item in status.items() if item["state"] == "unavailable"]
        progress("Modo automático: " + ("no disponibles: " + ", ".join(missing) if missing
                 else "componentes disponibles; preparando análisis."), 0.01)
    return settings, status
