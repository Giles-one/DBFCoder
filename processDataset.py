import os
import json
import ijson
import time

from collections import defaultdict
from typing import List, Dict

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s]->[%(asctime)s]-> %(message)s'
)
logger = logging.getLogger(__name__)

TOP = "/data/lgy/GraduationProject/dataset/NEW"
MERGED_FUNCTION_INFO_SUFFIX = '.functionInfo.json'
BAN_LISTS = [
    '_start',
    '__x86.get_pc_thunk.bx',
    'deregister_tm_clones',
    'register_tm_clones',
    '__do_global_dtors_aux',
    'frame_dummy',
    '__libc_csu_init',
    '__libc_csu_fini',
    '__do_global_ctors_aux'
]

def _process_asm_to_clap_format(asmcode: List) -> Dict[str, str]:
    asmcode.sort(key=lambda x: x[0])
    baseAddr = asmcode[0][0]
    # TODO: loc_DEADBEEF -> INST23
    clap_asm = {
        str(ea - baseAddr): ins.split(';')[0] for ea, ins in asmcode
    }
    return clap_asm

def _process_pseudocode(pseudocode: str) -> str:
    pseudocode = pseudocode.replace(
        '// [COLLAPSED LOCAL DECLARATIONS. PRESS KEYPAD CTRL-"+" TO EXPAND]',
        ''
    )
    pseudocode = pseudocode.replace(
        '// positive sp value has been detected, the output may be wrong!',
        ''
    )
    return pseudocode

def get_dirs_by_depth(top, depth):
    '''
     Retrieves all subdirectories at a specified depth
     relative to the given top directory.
    '''
    normDepth = lambda d: os.path.normpath(d).count(os.path.sep)
    topDepth = normDepth(top)
    dirsWithDepth = []
    for root, dirs, files in os.walk(top):
        if normDepth(root) == topDepth + depth:
            dirsWithDepth.append(root)
    dirsWithDepth.sort(key=str.lower)
    return dirsWithDepth

def is_elf_file(filePath):
    try:
        with open(filePath, 'rb') as file:
            magic = file.read(4)
    except Exception as e:
        print(f'opening file error {e}')
        return False
    if len(magic) < 4:
        return False
    return magic == b'\x7fELF'

def get_vaild_elf_lists(TOP):
    ELFFilePathLists = []
    for root, dirs, files in os.walk(TOP):
        for file in files:
            if file.endswith(".stripped"):
                continue
            if file.endswith(".json"):
                continue
            filePath = os.path.join(root, file)
            if not is_elf_file(filePath):
                continue
            if not os.path.exists(filePath + '.table.json'):
                continue
            if not os.path.exists(filePath + '.stripped'):
                continue
            if not os.path.exists(filePath + '.stripped.binaryInfo.json'):
                continue
            ELFFilePathLists.append(filePath)
    ELFFilePathLists.sort(key=str.lower)
    return ELFFilePathLists

def get_misc_info(ELFPath, TopDepth, incDepth):
    ELFPath = os.path.normpath(ELFPath)
    depth = ELFPath.count(os.path.sep)
    assert depth == TopDepth + incDepth
    miscStuff = ELFPath.split(os.path.sep)
    return {
        'filePath': ELFPath,
        'project': miscStuff[-7],
        'file': miscStuff[-6],
        'compiler': miscStuff[-5],
        'arch': miscStuff[-4],
        'bitness': miscStuff[-3],
        'optimization': miscStuff[-2]
    }

def merge_stripped_and_none_stripped_to_one(top):
    ELFFileDirLists = get_dirs_by_depth(top, 3)
    logger.info(f'ELFFileDirLists: {len(ELFFileDirLists)}')

    for ELFFileIndex, ELFFileDir in enumerate(ELFFileDirLists):
        logger.debug(f'[{ELFFileIndex}/{len(ELFFileDirLists)}]: processing {ELFFileDir}')
        saveTo = os.path.normpath(ELFFileDir) + MERGED_FUNCTION_INFO_SUFFIX
        ELFFilePathLists = get_vaild_elf_lists(ELFFileDir)

        mergedFunctionInfo = []
        for ELFFile in ELFFilePathLists:
            try:
                ELFFileFunctionTable = ELFFile + ".table.json"
                with open(ELFFileFunctionTable, 'r') as fp:
                    ELFFileFunctionTable = json.load(fp)

                ELFFileStripped = ELFFile + ".stripped"
                ELFFileStrippedBinaryInfo = ELFFileStripped + ".binaryInfo.json"
                with open(ELFFileStrippedBinaryInfo, 'r') as fp:
                    ELFFileStrippedBinaryInfo = json.load(fp)
            except Exception as e:
                logger.error(f'Error {e} when processing {ELFFile}')
                continue

            ELFFileDir = os.path.normpath(ELFFileDir)
            ELFFileDirDepth = ELFFileDir.count(os.path.sep)
            miscInfo = get_misc_info(ELFFile, ELFFileDirDepth, 5)

            noneStrippedfunctionInfo = []
            for fn in ELFFileFunctionTable:
                functionName = fn['name']
                functionAddr = fn['ea']
                noneStrippedfunctionInfo.append({
                    'functionName': functionName,
                    'functionAddr': functionAddr
                })

            strippedFunctionInfo = []
            for strippedFunction in ELFFileStrippedBinaryInfo:
                asmCode: List = strippedFunction['asmcode']
                pseudoCode = strippedFunction['pseudocode']
                strippedFunctionAddr = strippedFunction['ea']
                asmCode.sort(key=lambda x: x[0])
                strippedFunctionInfo.append({
                    'strippedFunctionAddr': strippedFunctionAddr,
                    'strippedAsmCode': asmCode,
                    'strippedPseudoCode': pseudoCode
                })

            for fnInfo in noneStrippedfunctionInfo:
                '''filter not general function'''
                if fnInfo['functionName'] in BAN_LISTS:
                    continue

                strippedFnInfo = next(
                    (d for d in strippedFunctionInfo if d['strippedFunctionAddr'] == fnInfo['functionAddr']),
                    None
                )
                if strippedFnInfo is None:
                    continue

                if len(strippedFnInfo['strippedAsmCode']) <= 20:
                    continue

                # Process strippedAsmCode and strippedPseudoCode
                strippedFnInfo['strippedAsmCode'] = _process_asm_to_clap_format(strippedFnInfo['strippedAsmCode'])
                strippedFnInfo['strippedPseudoCode'] = _process_pseudocode(strippedFnInfo['strippedPseudoCode'])

                _mergedFunctionInfo: Dict = {
                    **miscInfo,
                    **fnInfo,
                    **strippedFnInfo,
                }
                _mergedFunctionInfo.pop('strippedFunctionAddr')
                mergedFunctionInfo.append(_mergedFunctionInfo)
        with open(saveTo, 'w') as fp:
            json.dump(mergedFunctionInfo, fp, indent='\t')
def summarize_dataset(top: str):
    logger.info(f'Summarizing {top}')
    functionInfoJsonLists = []
    for root, dirs, files in os.walk(top):
        for file in files:
            if not file.endswith(MERGED_FUNCTION_INFO_SUFFIX):
                continue
            filePath = os.path.join(root, file)
            functionInfoJsonLists.append(filePath)
    logger.info(f'Found {MERGED_FUNCTION_INFO_SUFFIX}: {len(functionInfoJsonLists)}')

    summary = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )
    globalIndex = 0
    for fnInfoJson in functionInfoJsonLists:
        with open(fnInfoJson, 'rb') as fp:
            fnInfo = ijson.items(fp, 'item')
            for fn in fnInfo:
                project, file, functionName = (
                    fn['project'],
                    fn['file'],
                    fn['functionName']
                )
                summary[project][file][functionName].append(globalIndex)
                globalIndex += 1
    '''
    summary to file
        @sum.log
            "project": 
                "file"
                    "function": [0, 45, ...]
        @project.sum.log
            count project
              ... ...
        @file.sum.log
            count file@project
              ... ...
        @function.sum.log
            count function:file@project
              ... ...
    '''
    projectSummary = []
    for project in summary:
        key = project
        count = sum(
            len(fnIndex)
            for file in summary[project].values()
            for fnIndex in file.values()
        )
        projectSummary.append((count, key))
    projectSummary.sort(key=lambda x: x[0])

    fileSummary = []
    for project, files in summary.items():
        for file, functions in files.items():
            key = '%s@%s' % (file, project)
            count = sum(len(fnIndex) for fnIndex in functions.values())
            fileSummary.append((count, key))
    fileSummary.sort(key=lambda x: x[0])

    functionSummary = []
    for project, files in summary.items():
        for file, functions in files.items():
            for fnName, fnIndex in functions.items():
                key = '%s:%s@%s' % (fnName, file, project)
                count = len(fnIndex)
                functionSummary.append((count, key))
    functionSummary.sort(key=lambda x: x[0])

    with open('summary/sum.log', 'w') as fp:
        json.dump(summary, fp, indent='\t')
    logger.info('dump summary to sum.log')

    with open('summary/project.sum.log', 'w') as fp:
        fp.write('%8s %s\n' % ('count', 'project'))
        for line in projectSummary:
            fp.write('%8d %s\n' % line)
    logger.info('dump summary to project.sum.log')

    with open('summary/file.sum.log', 'w') as fp:
        fp.write('%8s %s\n' % ('count', 'file@project'))
        for line in fileSummary:
            fp.write('%8d %s\n' % line)
    logger.info('dump summary to file.sum.log')

    with open('summary/function.sum.log', 'w') as fp:
        fp.write('%8s %s\n' % ('count', 'function:file@project'))
        for line in functionSummary:
            fp.write('%8d %s\n' % line)
    logger.info('dump summary to function.sum.log')


if __name__ == '__main__':
    merge_stripped_and_none_stripped_to_one(top=TOP)
    summarize_dataset(top=TOP)