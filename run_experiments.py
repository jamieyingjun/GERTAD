"""Prepare local inputs and run all 15 frozen TEST evaluation jobs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Install Python 3.12, then run this command again.')
    for script in ('prepare_external.py', 'reproduce.py'):
        result = subprocess.run([sys.executable, '-B', str(ROOT / script)], cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
