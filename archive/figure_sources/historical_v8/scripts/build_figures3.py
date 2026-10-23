from pathlib import Path
import subprocess
import sys

master = Path(__file__).with_name("build_bioengineering_data_figures.py")
raise SystemExit(subprocess.call([sys.executable, str(master), "--figure", "FigureS3"]))
