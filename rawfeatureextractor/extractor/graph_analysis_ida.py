from idautils import *
from idaapi import *
from idc import *
import ida_ua
import ida_segment
import ida_bytes

def getfunc_consts(func):
	strings = []
	consts = []
	blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
	for bl in blocks:
		strs, conts = getBBconsts(bl)
		strings += strs
		consts += conts
	return strings, consts

def getConst(ea, offset):
	strings = []
	consts = []
	optype1 = get_operand_type(ea, offset)

	if optype1 == ida_ua.o_imm:
		imm_value = get_operand_value(ea, offset)
		if 0<= imm_value <= 10:
			consts.append(imm_value)
		else:
			if ida_bytes.is_loaded(imm_value) and ida_segment.getseg(imm_value):
				str_value = get_strlit_contents(imm_value)
				if str_value is None:
					str_value = get_strlit_contents(imm_value + get_imagebase())
					if str_value is None:
						consts.append(imm_value)
					else:
						re = all(40 <= ord(c) < 128 for c in str_value)
						if re:
							strings.append(str_value)
						else:
							consts.append(imm_value)
				else:
					re = all(40 <= ord(c) < 128 for c in str_value)
					if re:
						strings.append(str_value)
					else:
						consts.append(imm_value)
			else:
				consts.append(imm_value)
	return strings, consts

def getBBconsts(bl):
	strings = []
	consts = []
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = print_insn_mnem(inst_addr)
		if opcode in ['la','jalr','call', 'jal']:
			inst_addr = next_head(inst_addr)
			continue
		strings_src, consts_src = getConst(inst_addr, 0)
		strings_dst, consts_dst = getConst(inst_addr, 1)
		strings += strings_src
		strings += strings_dst
		consts += consts_src
		consts += consts_dst
		try:
			strings_dst, consts_dst = getConst(inst_addr, 2)
			consts += consts_dst
			strings += strings_dst
		except:
			pass

		inst_addr = next_head(inst_addr)
	return strings, consts

def getFuncCalls(func):
	blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calCalls(bl)
		sumcalls += callnum
	return sumcalls

def getLogicInsts(func):
	blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calLogicInstructions(bl)
		sumcalls += callnum
	return sumcalls

def getTransferInsts(func):
	blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calTransferIns(bl)
		sumcalls += callnum
	return sumcalls

def getIntrs(func):
	blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
	sumcalls = 0
	for bl in blocks:
		callnum = calInsts(bl)
		sumcalls += callnum
	return sumcalls	

def getLocalVariables(func):
	args_num = get_stackVariables(func.start_ea)
	return args_num

def getBasicBlocks(func):
	blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
	return len(blocks)

def getIncommingCalls(func):
	refs = CodeRefsTo(func.start_ea, 0)
	re = len([v for v in refs])
	return re


def get_stackVariables(func_addr):
    #print func_addr
    args = []
    stack = get_frame_id(func_addr)
    if not stack:
            return 0
    firstM = get_first_member(stack)
    lastM = get_last_member(stack)
    i = firstM
    while i <=lastM:
        mName = get_member_name(stack,i)
        mSize = get_member_size(stack,i)
        if mSize:
                i = i + mSize
        else:
                i = i+4
        if mName not in args and mName and 'var_' in mName:
            args.append(mName)
    return len(args)



def calArithmeticIns(bl):
	x86_AI = {'add':1, 'sub':1, 'div':1, 'imul':1, 'idiv':1, 'mul':1, 'shl':1, 'dec':1, 'inc':1}
	mips_AI = {'add':1, 'addu':1, 'addi':1, 'addiu':1, 'mult':1, 'multu':1, 'div':1, 'divu':1}
	arm_AI = {'AND':1, 'ADC':1, 'SUB':1, 'SBC':1, 'RSB':1, 'RSC':1}
	pc_AI = {'add':1, 'addc':1, 'adde':1, 'addi':1, 'addic':1, 'addme':1, 'addze':1, 'divw':1, 'divwu':1, 'mulhw':1, 'mulhwu':1, 'mulli':1, 'mullw':1}
	calls = {}
	calls.update(x86_AI)
	calls.update(mips_AI)
	calls.update(arm_AI)
	calls.update(pc_AI)
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = print_insn_mnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = next_head(inst_addr)
	return invoke_num

def calCalls(bl):
	calls = {'call':1, 'jal':1, 'jalr':1, 'BL':1, 'BLX':1}
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = print_insn_mnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = next_head(inst_addr)
	return invoke_num

def calInsts(bl):
	start = bl[0]
	end = bl[1]
	ea = start
	num = 0
	while ea < end:
		num += 1
		ea = next_head(ea)
	return num

def calLogicInstructions(bl):
	x86_LI = {'and':1, 'andn':1, 'andnpd':1, 'andpd':1, 'andps':1, 'andnps':1, 'test':1, 'xor':1, 'xorpd':1, 'pslld':1}
	mips_LI = {'and':1, 'andi':1, 'or':1, 'ori':1, 'xor':1, 'nor':1, 'slt':1, 'slti':1, 'sltu':1}
	arm_LI = {'AND':1, 'EOR':1, 'ORR':1, 'BIC':1, 'ORN':1, 'OR':1, 'NOT':1, 'TST':1, 'TEQ':1, 'EOR':1, 'BIC':1, 'CLZ':1, 'ASR':1, 'LSL':1, 'LSR':1, 'ROR':1, 'RRX':1}
	pc_LI = {'and':1, 'andc':1, 'andi.':1, 'andis.':1, 'crand':1, 'crandc':1, 'creqv':1, 'crnand':1, 'crnor':1, 'cror':1, 'crorc':1, 'crxor':1}
	calls = {}
	calls.update(x86_LI)
	calls.update(mips_LI)
	calls.update(arm_LI)
	calls.update(pc_LI)
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = print_insn_mnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = next_head(inst_addr)
	return invoke_num

def calSconstants(bl):
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = print_insn_mnem(inst_addr)
		if opcode in calls:
			invoke_num += 1
		inst_addr = next_head(inst_addr)
	return invoke_num


def calNconstants(bl):
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		optype1 = get_operand_type(inst_addr, 0)
		optype2 = get_operand_type(inst_addr, 1)
		if optype1 == 5 or optype2 == 5:
			invoke_num += 1
		inst_addr = next_head(inst_addr)
	return invoke_num

def retrieveExterns(bl, ea_externs):
	externs = []
	start = bl[0]
	end = bl[1]
	inst_addr = start
	while inst_addr < end:
		refs = CodeRefsFrom(inst_addr, 1)
		try:
			ea = [v for v in refs if v in ea_externs][0]
			externs.append(ea_externs[ea])
		except:
			pass
		inst_addr = next_head(inst_addr)
	return externs

def calTransferIns(bl):
	x86_TI = {'jmp':1, 'jz':1, 'jnz':1, 'js':1, 'je':1, 'jne':1, 'jg':1, 'jle':1, 'jge':1, 'ja':1, 'jnc':1, 'call':1}
	mips_TI = {'beq':1, 'bne':1, 'bgtz':1, "bltz":1, "bgez":1, "blez":1, 'j':1, 'jal':1, 'jr':1, 'jalr':1}
	arm_TI = {'CMP':1, 'CMN':1, 'B':1, 'BL':1, 'BX':1, 'BLX':1, 'BXJ':1, 'IT':1, 'CBZ':1, 'CBNZ':1}
	pc_TI= {'b':1, 'bc':1, 'bcctr':1, 'bclr':1}
	calls = {}
	calls.update(x86_TI)
	calls.update(mips_TI)
	calls.update(arm_TI)
	calls.update(pc_TI)
	start = bl[0]
	end = bl[1]
	invoke_num = 0
	inst_addr = start
	while inst_addr < end:
		opcode = print_insn_mnem(inst_addr)
		re = [v for v in calls if opcode in v]
		if len(re) > 0:
			invoke_num += 1
		inst_addr = next_head(inst_addr)
	return invoke_num