import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile


def do_preprocessing(outpath, binary_path, binary_name):
    binary_full_path = os.path.join(binary_path, binary_name)
    out_path = os.path.join(outpath, binary_name + '.ida')

    # binary already analyized
    if os.path.exists(outpath) and os.path.isfile(out_path):
        return out_path

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # calling ida pro shell command
    # cwd = os.getcwd()
    # cmd = 'QT_X11_NO_MITSHM=1 ./ida/idaq -A -S\"' + cwd + '/raw-feature-extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path
    cmd = './idat64 -c -A -S"' + os.path.split(os.path.abspath(__file__))[
        0] + '/preprocessing_ida.py ' + outpath + '" ' + binary_full_path
    print(cmd)
    subprocess.call(shlex.split(cmd, posix=True))

    if os.path.isfile(out_path):
        return out_path

    return
