"""Create optional isolated model environments. Never modifies app packages."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess

GSR_REVISION = "1c958345067218297d221e45e1a6405f975f83e0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["actions", "gamestate"])
    args = parser.parse_args()
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Instala uv primero: python -m pip install uv")
    root = Path(__file__).resolve().parent
    base = root / ".model-envs"
    env = base / args.model
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run = lambda command, cwd=None: subprocess.run(command, check=True, cwd=cwd)
    if not python.is_file():
        run([uv, "venv", "--python", "3.12" if args.model == "actions" else "3.9", str(env)])
    if args.model == "actions":
        run([uv, "pip", "install", "--python", str(python), "-r", str(root / "requirements-actions.txt")])
        key = "MATCHVIEW_OSL_PYTHON"
    else:
        checkout = base / "sn-gamestate"
        if not checkout.exists():
            run(["git", "clone", "https://github.com/SoccerNet/sn-gamestate.git", str(checkout)])
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                                check=True, capture_output=True, text=True)
        if result.stdout.strip() != GSR_REVISION:
            # Do not switch an existing checkout with user changes.
            status = subprocess.run(["git", "status", "--porcelain"], cwd=checkout,
                                    check=True, capture_output=True, text=True)
            if status.stdout.strip():
                raise SystemExit("El checkout sn-gamestate tiene cambios locales. Consérvalos y revisa su versión.")
            run(["git", "checkout", "--detach", GSR_REVISION], cwd=checkout)
        run([uv, "pip", "install", "--python", str(python), "-e", str(checkout / "plugins/calibration")])
        run([uv, "pip", "install", "--python", str(python), "-e", str(checkout)])
        run([str(python), "-m", "mim", "install", "mmcv==2.0.1"])
        key = "MATCHVIEW_GSR_PYTHON"
        print(f'MATCHVIEW_GSR_ROOT={checkout}')
    # Environment variables remain confined to the shell that launches Streamlit.
    print(f'{key}={python}')
    print("Configura esas variables en la terminal desde la que lanzas Streamlit.")


if __name__ == "__main__":
    main()
