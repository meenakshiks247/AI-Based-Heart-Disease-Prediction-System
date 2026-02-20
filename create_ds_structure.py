#!/usr/bin/env python3
"""
create_ds_structure.py

Creates a standard data-science folder layout using the `os` library.

Folders: data, notebooks, models, src
Files: requirements.txt (empty if missing), .gitignore (created with sensible defaults if missing)

Usage:
    python create_ds_structure.py [base_path] [--gitignore-default]

If no base_path is provided the current working directory is used.
"""
import os
import argparse
import sys

DEFAULT_DIRS = ["data", "notebooks", "models", "src"]

GITIGNORE_CONTENT = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.env

# Jupyter
.ipynb_checkpoints

# IDEs
.vscode/
.idea/

# OS
.DS_Store
"""

def create_dirs(base_path, dir_list):
    for d in dir_list:
        path = os.path.join(base_path, d)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Failed to create directory {path}: {e}", file=sys.stderr)
        else:
            if os.path.isdir(path):
                print(f"Created or exists: {path}")

def create_file_if_missing(path, content=None):
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                if content:
                    f.write(content)
        except OSError as e:
            print(f"Failed to create file {path}: {e}", file=sys.stderr)
        else:
            print(f"Created file: {path}")
    else:
        print(f"Already exists: {path}")

def parse_args():
    p = argparse.ArgumentParser(description="Create a professional data-science folder structure.")
    p.add_argument("base", nargs="?", default=".", help="Base directory to create the structure in (default: current directory)")
    p.add_argument("--gitignore-default", action="store_true", help="Populate .gitignore with sensible defaults when creating it")
    return p.parse_args()

def main():
    args = parse_args()
    base = os.path.abspath(args.base)

    if not os.path.exists(base):
        try:
            os.makedirs(base, exist_ok=True)
        except OSError as e:
            print(f"Cannot create base directory {base}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Using base directory: {base}")
    create_dirs(base, DEFAULT_DIRS)

    req_path = os.path.join(base, "requirements.txt")
    gitignore_path = os.path.join(base, ".gitignore")

    create_file_if_missing(req_path, content="")

    if args.gitignore_default:
        create_file_if_missing(gitignore_path, content=GITIGNORE_CONTENT)
    else:
        create_file_if_missing(gitignore_path, content=None)

    print("Done.")

if __name__ == "__main__":
    main()
