"""32-point Roboflow field topology for the martinjolif YOLO checkpoint."""
from pathlib import Path

PITCH_REPO = "martinjolif/yolo-football-pitch-detection"
PITCH_REVISION = "bdb4687"
PITCH_FILE = "yolo-football-pitch-detection.pt"


def pitch_points_32(length=105.0, width=68.0):
    # Topology: sports.configs.soccer.SoccerPitchConfiguration.vertices.
    # Use physical 105 x 68 m geometry, not the diagram's default 120 x 70.
    penalty_top, penalty_bottom = (width-40.32)/2, (width+40.32)/2
    goal_top, goal_bottom = (width-18.32)/2, (width+18.32)/2
    vertices = [
        (0, 0), (0, penalty_top), (0, goal_top), (0, goal_bottom),
        (0, penalty_bottom), (0, width), (5.5, goal_top), (5.5, goal_bottom),
        (11, width/2), (16.5, penalty_top), (16.5, goal_top),
        (16.5, goal_bottom), (16.5, penalty_bottom), (length/2, 0),
        (length/2, width/2-9.15), (length/2, width/2+9.15), (length/2, width),
        (length-16.5, penalty_top), (length-16.5, goal_top),
        (length-16.5, goal_bottom), (length-16.5, penalty_bottom),
        (length-11, width/2), (length-5.5, goal_top), (length-5.5, goal_bottom),
        (length, 0), (length, penalty_top), (length, goal_top),
        (length, goal_bottom), (length, penalty_bottom), (length, width),
        (length/2-9.15, width/2), (length/2+9.15, width/2),
    ]
    return dict(enumerate(vertices))


def load_pitch_detector():
    from ultralytics import YOLO
    path = Path(__file__).resolve().parent / PITCH_FILE
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(PITCH_REPO, PITCH_FILE,
                                   revision=PITCH_REVISION, token=False))
    model = YOLO(str(path))
    network = getattr(model, "model", None)
    shape = getattr(network, "kpt_shape", None)
    if shape is None:
        shape = (getattr(network, "yaml", None) or {}).get("kpt_shape")
    if shape is None:
        layers = getattr(network, "model", [])
        shape = getattr(layers[-1], "kpt_shape", None) if len(layers) else None
    if shape is None or list(shape) != [32, 3]:
        raise ValueError("El modelo del campo debe tener exactamente 32 keypoints (x, y, confianza).")
    return model
