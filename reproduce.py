"""Run all five frozen methods and three seeds: python reproduce.py."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
REQUIRED = {'torch': '2.10.0+cpu', 'numpy': '2.4.3', 'scipy': '1.17.1',
            'scikit-learn': '1.9.0', 'joblib': '1.5.3', 'threadpoolctl': '3.6.0', 'tqdm': '4.70.0'}


def prepare_environment():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Install Python 3.12, then run this command again.')
    ready = True
    for package, version in REQUIRED.items():
        try:
            ready = ready and importlib.metadata.version(package) == version
        except importlib.metadata.PackageNotFoundError:
            ready = False
    if ready:
        return
    environment = ROOT / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.prefix).resolve() == environment.resolve():
        raise RuntimeError('The local environment has incompatible dependencies. Remove .venv and retry.')
    print('Preparing the local Python environment; the first run needs internet access.', flush=True)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True)
    raise SystemExit(subprocess.call([str(python), '-B', str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('tad', 'lm', 'ce', 'gm', 'idk'), help='Optional single-method diagnostic.')
    parser.add_argument('--seed', type=int, choices=(17, 29, 43), help='Optional single-seed diagnostic.')
    parser.add_argument('--output', type=Path, help='Optional new output directory.')
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    prepare_environment()
    sys.path.insert(0, str(ROOT))
    from gertad.experiments import main as run
    run(args.method, args.seed, args.output)


if __name__ == '__main__':
    main()
