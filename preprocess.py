import os
import argparse
import hashlib
import subprocess
import shlex
import sys
import shutil
import glob

def _get_binary_hash(file_path):
    fd = open(file_path, 'rb')
    md5 = hashlib.sha256()
    data = fd.read()
    md5.update(data)
    return md5.hexdigest()


def gen_callgraph(outpath, binary_path, binary_name, script_path):
    binary_full_path = os.path.join(binary_path, binary_name)
    i64_out_path = binary_full_path + '.i64'
    out_path = os.path.join(outpath, binary_name + '.gdl')
    #out_path = os.path.join(outpath, binary_name)

    # binary already analyized
    if os.path.exists(outpath) and os.path.isfile(out_path):
        return out_path

    if not os.path.exists(outpath):
        print("mkdir " + outpath)
        os.mkdir(outpath)
    # calling ida pro shell command
    cwd = os.getcwd()
    print("test4")

    cmd = '/opt/ida-7.1/' + 'idat64 -B ' + binary_full_path
    print cmd
    subprocess.call(shlex.split(cmd))
    if os.path.isfile(i64_out_path):
        #cmd = 'QT_X11_NO_MITSHM=1 ./ida/idaq -A -S\"' + cwd + '/raw-feature-extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path
        #cmd = '/home/yijiufly/Downloads/codesearch/ida/idal64 -c -A -S\"' + cwd + '/raw_feature_extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path
        #cmd = '/opt/ida-7.1/' + 'idat64 -c -A -S\"' + '/home/yijiufly/Downloads/codesearch/rawfeatureextractor' + '/extractor/gen_cfg.py ' + outpath + '\" ' + binary_full_path
        cmd = '/opt/ida-7.1/' + 'idat64 -A -S\"' + \
            script_path + ' ' + out_path + '\" ' + i64_out_path
        print cmd
        subprocess.call(shlex.split(cmd))
        if os.path.isfile(out_path):
            return out_path

    return


def do_preprocessing(outpath, binary_path, binary_name, script_path):
    binary_full_path = os.path.join(binary_path, binary_name)
    out_path = os.path.join(outpath, binary_name + '.ida')
    #out_path = os.path.join(outpath, binary_name)

    # binary already analyized
    if os.path.exists(outpath) and os.path.isfile(out_path):
        return out_path

    if not os.path.exists(outpath):
        print("mkdir " + outpath)
        os.mkdir(outpath)
    # calling ida pro shell command
    cwd = os.getcwd()
    print("test4")

    #cmd = 'QT_X11_NO_MITSHM=1 ./ida/idaq -A -S\"' + cwd + '/raw-feature-extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path
    #cmd = '/home/yijiufly/Downloads/codesearch/ida/idal64 -c -A -S\"' + cwd + '/raw_feature_extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path
    cmd = '/opt/ida-7.1/' + 'idat64 -c -A -S\"' + '/home/yijiufly/Downloads/codesearch/rawfeatureextractor' + '/extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path

    print cmd
    subprocess.call(shlex.split(cmd))
    if os.path.isfile(out_path):
        return out_path

    return


def get_dot(outpath, binary_path, binary_name):
    binary_full_path = os.path.join(binary_path, binary_name)
    in_path = os.path.join(outpath, binary_name + '.gdl')
    out_path = os.path.join(outpath, binary_name + '.dot')

    # binary already analyized
    if os.path.exists(outpath) and os.path.isfile(out_path):
        return out_path

    if not os.path.exists(outpath):
        print("mkdir " + outpath)
        os.mkdir(outpath)

    cmd = 'graph-easy' + ' ' + in_path + ' --dot --output=' + out_path

    print cmd
    subprocess.call(shlex.split(cmd))
    if os.path.isfile(out_path):
        return out_path

    return


def parse_command():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--path", type=str, help=",The complete path of folder from which binary file to be preprocessed", required=True)
    parser.add_argument(
        "--name", type=str, help="The name of binary file to be preprocessed", required=True)
    parser.add_argument("--out", type=str,
                        help="The output path of IDA file", required=True)
    args = parser.parse_args()
    return args


# pass binary dir and binary name as argument
if __name__ == '__main__':
    args = parse_command()
    # get as an argument
    bin_path = args.path
    bin_name = args.name

    bin_full_path = os.path.join(bin_path, bin_name)
    print("test1")
    bhash = _get_binary_hash(bin_full_path)

    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #out_dir = os.path.join(dir_path, 'out_analysis')
    out_dir = args.out

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # copy to test3
    # out_path = os.path.join(out_dir, bin_name)
    # os.mkdir(out_path)
    # shutil.copy(bin_full_path, out_path)
    # for file in glob.iglob(r'/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/'+bhash+'/*.gdl'):
    #     shutil.copy(file, out_path)
    # for file in glob.iglob(r'/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/'+bhash+'/*.dot'):
    #     shutil.copy(file, out_path)

    print("test2")
    out_path = os.path.join(out_dir, bhash)
    print(out_path + '\n' + bin_path + '\n' + bin_name)

    #ida_path = do_preprocessing(out_dir, bin_path, bin_name, 'rawfeatureextractor' + '/extractor/preprocessing_ida.py')
    ida_path = gen_callgraph(out_dir, bin_path, bin_name, 'rawfeatureextractor/extractor/gen_callgraph.py')
    ida_path = get_dot(out_dir, bin_path, bin_name)
    print("test3")

    # check wheter ida processing done or not
    if ida_path is None:
        print("no .dot output")
    else:
        print(ida_path)

    #generate .dot
