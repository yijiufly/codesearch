import binaryninja as binja
import pdb
from collections import defaultdict
import pickle as p


def build_string_table(bv, db=None):
    addr = []
    string_dict = defaultdict(list)
    for key in bv.sections.keys():
        if 'rodata' in key:
            start = bv.sections[key].start
            end = bv.sections[key].end
            for string in bv.get_strings(start, end):
                addr.append(string.start)
    if addr == []:
        for string in bv.get_strings():
            addr.append(string.start)

    for func in bv.functions:
        print(func.name)
        string_list = []
        for block in func.low_level_il.basic_blocks:
            for instr in block:
                for oper in instr.operands:
                    if isinstance(oper, binja.LowLevelILInstruction):
                        if oper.operation == binja.LowLevelILOperation.LLIL_CONST_PTR or oper.operation == binja.LowLevelILOperation.LLIL_CONST:
                            if int(str(oper), 16) in addr:
                                string_list.append(bv.get_string_at(int(str(oper), 16)).value)
        if len(string_list) > 0:
            string_list_strip = list(set(string_list))
            my_dict = {"name": func.name, "strings": string_list_strip}
            string_dict[func.name] = string_list_strip
            if db is not None:
                db.save(my_dict)
    return string_dict


binaryview = binja.BinaryViewType.get_view_of_file('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/nginx-{openssl-1.0.1d}{zlib-1.2.11}')
string_table = build_string_table(binaryview)
pdb.set_trace()
