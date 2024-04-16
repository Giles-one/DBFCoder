import sark
import json
import idc, ida_nalt, ida_auto, ida_pro
from ida_hexrays import decompile, DecompilationFailure

ida_auto.auto_wait()
inputfile = ida_nalt.get_root_filename()

def noneStrippedFilePayload(saveTo: str):
    functionTable = []
    text = sark.Segment(name='.text')
    for fn in text.functions:
        fnName = fn.name
        fnAddr = fn.ea
        functionTable.append({
            'name': fnName,
            'ea': fnAddr
        })
    # print(functionTable)
    with open(saveTo, 'w', encoding='utf-8') as fp:
        json.dump(functionTable, fp, indent='\t')

def strippedFilePayload(saveTo: str):
    binaryInfo = []
    text = sark.Segment(name='.text')
    for fn in text.functions:
        fnAddr = fn.ea
        asmcode = []
        for ln in fn.lines:
            asmcode.append([ln.ea, ln.disasm])
        try:
            pseudocode = str(decompile(fn.ea))
        except DecompilationFailure:
            pseudocode = None # WTF, don't do like this.
            pseudocode = ''
        binaryInfo.append({
            'ea': fnAddr,
            'asmcode': asmcode,
            'pseudocode': pseudocode
        })
    # print(binaryInfo)
    with open(saveTo, 'w', encoding='utf-8') as fp:
        json.dump(binaryInfo, fp, indent='\t')

inputFile = ida_nalt.get_root_filename()
if inputFile.endswith('.stripped'):
    saveTo = inputFile + '.binaryInfo.json'
    strippedFilePayload(saveTo)
else:
    saveTo = inputFile + '.table.json'
    noneStrippedFilePayload(saveTo)

ida_pro.qexit(0)
