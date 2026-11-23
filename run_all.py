"""Execute each experiment's script, in M0--M3 order.

The individual scripts default to saved results: no fitting or FEM is launched.
Inspect MODE in each script before changing it or running this file.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    for model in ['M0','M1','M2','M3']:
        subprocess.run([sys.executable,str(ROOT/model/f'run_{model}.py')],check=True)
