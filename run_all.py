import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
SCRIPTS = ["exp_folding.py", "exp_correctness.py", "exp_bench.py",
           "exp_params.py", "exp_keygen.py", "exp_sizes.py", "make_figures.py"]

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=HERE, check=True)
    for s in SCRIPTS:
        print("=" * 70)
        print(s)
        print("=" * 70)
        subprocess.run([sys.executable, s], cwd=EXP, check=True)
