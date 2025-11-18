#!/usr/bin/env python3

import os
import sys

try:
    import uproot
except ImportError:
    print("Error: uproot is not installed. Install with `pip install uproot`.")
    sys.exit(1)


def check_root_file(path):
    """
    Try to open a ROOT file with uproot.
    Returns (is_ok, error_message or None).
    """
    try:
        # Open file and force reading of top-level keys (meta-structure)
        with uproot.open(path) as f:
            _ = f.keys()
        return True, None
    except Exception as e:
        return False, str(e)


def scan_main_directory(main_dir):
    corrupted = []
    ok = []

    # Walk through all subdirectories of main_dir
    for dirpath, dirnames, filenames in os.walk(main_dir):
        root_files = [f for f in filenames if f.endswith(".root")]
        if not root_files:
            continue

        print(f"\nChecking directory: {dirpath}")
        for fname in root_files:
            fpath = os.path.join(dirpath, fname)
            is_ok, err = check_root_file(fpath)
            if is_ok:
                print(f"  [OK]      {fname}")
                ok.append(fpath)
            else:
                print(f"  [CORRUPT] {fname}")
                print(f"           -> {err}")
                corrupted.append(fpath)

    print("\n=== Summary ===")
    print(f"Good files:     {len(ok)}")
    print(f"Corrupted files:{len(corrupted)}")

    if corrupted:
        print("\nList of corrupted files:")
        for f in corrupted:
            print(f"  {f}")


if __name__ == "__main__":
    main_dir =  "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251114"
    if not os.path.isdir(main_dir):
        print(f"Error: {main_dir} is not a directory")
        sys.exit(1)

    scan_main_directory(main_dir)
