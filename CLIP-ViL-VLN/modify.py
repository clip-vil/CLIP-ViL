import os
import shutil
import argparse

ORIG_FEAT = 'ResNet-152-imagenet'

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="ResNet-152-imagenet")
parser.add_argument("--dim", default="2048")
parser.add_argument("--src-dir", default="r2r_src")

args = parser.parse_args()

print("Get ORIG files:")
orig_py_files = []
for fname in os.listdir(args.src_dir):
    stem, ext = os.path.splitext(fname)
    full_path = os.path.join(args.src_dir, fname)
    if ext == '.py':
        if 'ORIG' in stem:
            print('Ignore %s' % full_path)
            orig_py_files.append(full_path)
        else:
            new_full_path = os.path.join(args.src_dir, stem + "_ORIG" + ".py")
            if os.path.exists(new_full_path):
                pass
            else:
                print("Move %s to %s" % (full_path, new_full_path))
                shutil.copy(full_path, new_full_path)
                orig_py_files.append(new_full_path)

print("\nModify ORIG files:")
for path in orig_py_files:
    stem, ext = os.path.splitext(path)
    new_stem = stem.replace("_ORIG", '')
    new_path = new_stem + ext

    print(new_path)
    lines = open(path).readlines()
    new_lines = []
    for line in lines:
        new_line = (line.replace('2048', str(args.dim))
                        .replace(ORIG_FEAT, args.name))
        if new_line != line:
            print("OLD: %s" % line[:-1])
            print("NEW: %s" % new_line[:-1])
        new_lines.append(new_line)
    
    with open(new_path, 'w') as f:
        f.writelines(new_lines)

