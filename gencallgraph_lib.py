import struct
import threading

import binaryninja as bn
import pdb
import os
import glob
import pickle as p
def get_indirect_address(view, func, instr, load_il):
    if view.address_size == 4:
        unpack_string = 'L'
    elif view.address_size == 8:
        unpack_string = 'Q'
    elif view.address_size == 2:
        unpack_string = 'H'
    else:
        # pretty sure this will never happen!
        unpack_string = 'B'

    if view.endianness == bn.Endianness.LittleEndian:
        unpack_string = '<' + unpack_string

    if load_il.operation == bn.LowLevelILOperation.LLIL_CONST_PTR:
        address = struct.unpack(
            unpack_string,
            view.read(load_il.value.value, view.address_size)
        )[0]

        if view.is_offset_executable(address):
            return address
        else:
            return load_il.value.value

    elif load_il.operation == bn.LowLevelILOperation.LLIL_REG:
        reg_value = func.get_reg_value_at(
            instr.address,
            load_il.src
        )

        if reg_value.type == bn.RegisterValueType.ConstantValue:
            const_value = view.read(reg_value.value, view.address_size)

            if const_value:
                if len(const_value) != view.address_size:
                    bn.log_info("const_value: {!r}".format(const_value))
                    return None
                address = struct.unpack(
                    unpack_string,
                    const_value
                )[0]
            else:
                return None

            if view.is_offset_executable(address):
                return address

    return None


def get_name(addr, view):
    symbol = view.get_symbol_at(addr)

    if symbol is not None:
        name = symbol.name.replace('!', '_').replace('@', '_')
    else:
        name = 'sub_{:x}'.format(abs(addr))

    return name


def gencallgraph(view, calledges):
    for func in view.functions:
        node = get_name(func.start, view)
        for block in func.low_level_il.basic_blocks:
            for instr in block:
                if instr.operation == bn.LowLevelILOperation.LLIL_CALL:
                    if instr.dest.operation == bn.LowLevelILOperation.LLIL_CONST_PTR:                        
                        calledges.append([node, get_name(instr.dest.value.value, view)])
                    elif instr.dest.operation == bn.LowLevelILOperation.LLIL_EXTERN_PTR:
                        calledges.append([node, get_name(int(str(instr.dest.tokens[0]), 16), view)])
                    elif instr.dest.operation == bn.LowLevelILOperation.LLIL_LOAD:
                        target = get_indirect_address(view, func, instr, instr.dest.src)

                        if target is not None:
                            calledges.append([node, get_name(target, view)])

                    elif instr.dest.operation == bn.LowLevelILOperation.LLIL_REG:
                        #pdb.set_trace()
                        reg_value = func.get_reg_value_at(
                            instr.address,
                            instr.dest.src
                        )

                        if (reg_value.type == bn.RegisterValueType.ConstantValue and
                                view.is_offset_executable(reg_value.value)):
                            calledges.append([node, get_name(reg_value.value, view)])

if __name__ == '__main__':
    libdir = '/home/yijiufly/Downloads/codesearch/data/mupdf/libjpeg-test'
    
    for folder in os.listdir(libdir):
        if os.path.exists(libdir + '/' + folder + '/edges.p'):
            continue
        calledges = []
        for path in glob.iglob(libdir + '/' + folder + '/objfiles/*.o'):
            print path
            view = bn.BinaryViewType.get_view_of_file(path)
            gencallgraph(view, calledges)

        p.dump(calledges, open(libdir + '/' + folder + '/edges.p', 'w'))
