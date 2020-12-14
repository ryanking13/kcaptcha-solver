import random
import pathlib
import shutil
import sys

cnt = int(sys.argv[1])
files = list(pathlib.Path(".data").glob("*.png"))
selected_files = random.choices(files, k=20)

for f in selected_files:
    shutil.copy(f, f"{f.name}")