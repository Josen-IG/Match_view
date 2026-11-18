"""Create optional isolated model environments. Never modifies app packages."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

GSR_REVISION = "1c958345067218297d221e45e1a6405f975f83e0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["actions", "gamestate"])
    args = parser.parse_args()
    uv = shutil.which("uv")
    root = Path(__file__).resolve().parent
    base = root / ".model-envs"
    env = base / args.model
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run = lambda command, cwd=None: subprocess.run(command, check=True, cwd=cwd)
    if not python.is_file():
        if uv:
            run([uv, "venv", "--python", "3.12" if args.model == "actions" else "3.9", str(env)])
        elif args.model == "actions" and sys.version_info >= (3, 12):
            run([sys.executable, "-m", "venv", str(env)])
        else:
            raise SystemExit("Instala uv o ejecuta este script con Python 3.12 para actions.")
    if args.model == "actions":
        install = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install"]
        run(install + ["-r", str(root / "requirements-actions.txt")])
        subprocess.run([str(python), "-c", "from opensportslib.apis import LocalizationModel; import torch; import torchvision"], check=True)
        print("Entorno de acciones preparado. Match View lo detectará automáticamente.")
        key = "MATCHVIEW_OSL_PYTHON"
    else:
        if not uv:
            raise SystemExit("Instala uv primero: python -m pip install uv")
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
