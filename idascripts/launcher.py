import os
import json
import subprocess
import multiprocessing as mp

from typing import List

import logging
logging.basicConfig(
    level = logging.INFO,
    format='[%(process)d]->[%(levelname)s]->[%(asctime)s]-> %(message)s'
)
logger = logging.getLogger(__name__)

PAYLOAD = 'payload.py'
TARGET_DIR = "path/to/dataset"

if not os.path.exists(PAYLOAD):
    logger.error('WTF? Can not find {}'.format(PAYLOAD))
    exit(-1)
PAYLOAD = os.path.abspath(PAYLOAD)
logger.info('Load payload {}'.format(PAYLOAD))

def _is_ELF_file(ELFPath: str) -> bool:
    ELF = open(ELFPath, 'rb')
    e_ident = ELF.read(16); ELF.close()
    if len(e_ident) < 16:
        return False
    ELFMAG = e_ident[:4]
    EI_CLASS = e_ident[4]
    if ELFMAG != b'\x7fELF':
        return False
    return True

def ELFHeaderPaser(ELFPath: str) -> int:
    '''ELF Struct Information lies in /usr/include/elf.h.
    ALSO SEE: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html
    '''
    assert os.path.isfile(ELFPath), 'File {] not exits.'.format(ELFPath)
    ELF = open(ELFPath, 'rb')
    e_ident = ELF.read(16); ELF.close()
    ELFMAG = e_ident[:4]
    EI_CLASS = e_ident[4]
    assert ELFMAG == b'\x7fELF', 'Not a vaild ELF file {}'.format(ELFPath)
    if EI_CLASS == 1:
        return 32
    elif EI_CLASS == 2:
        return 64
    else:
        return -1

def getIDAHandler(ELFPath: str) -> str:
    bitness = ELFHeaderPaser(ELFPath)
    if bitness == 32:
        return 'ida.exe'
    elif bitness == 64:
        return 'ida64.exe'
    else:
        return None

def processTask(ELFPath: str):
    IDA = getIDAHandler(ELFPath)
    if IDA is None:
        raise Exception('IDA not found.')
    cmd = f'{IDA} -A -S{PAYLOAD} {ELFPath}'
    logger.debug(cmd)
    if ELFPath.endswith('.stripped'):
        saveTo = ELFPath + '.binaryInfo.json'
    else:
        saveTo = ELFPath + '.table.json'
    if os.path.exists(saveTo):
        return
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        timeout=20 * 60,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if not os.path.exists(saveTo):
        raise Exception('IDA Process Error')

def workerProcess(undoELF: mp.Queue, failELF: mp.Queue, processId: int):
   while True:
       if undoELF.empty():
           break
       ELFPath = undoELF.get()
       undoNum = undoELF.qsize()
       logger.info('[{}][{}][{}]'.format(processId, undoNum, ELFPath))
       try:
           processTask(ELFPath)
       except:
           logger.error('Fails {}'.format(ELFPath))
           failELF.put(ELFPath)

def getFileLists(root: str) -> List[str]:
    ELFPathLists = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.json'):
                continue
            if file.endswith('.idb'):
                continue
            if file.endswith('.i64'):
                continue
            filePath = os.path.join(root, file)
            if not _is_ELF_file(filePath):
                continue
            ELFPathLists.append(filePath)
    return ELFPathLists

def main():
    ELFPathLists = getFileLists(TARGET_DIR)
    logger.info('Get {} ELF files.'.format(len(ELFPathLists)))

    undoELF = mp.Queue()
    failELF = mp.Queue()
    for ELFPath in ELFPathLists:
        undoELF.put(ELFPath)
    # breakpoint()
    numProcess = mp.cpu_count()
    # numProcess = 2
    processes = []
    for processId in range(numProcess):
        p = mp.Process(target=workerProcess, args=(undoELF, failELF, processId))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    ELFFailList = []
    while not failELF.empty():
        ELFFailList.append(failELF.get())
    logger.error('fails {} times'.format(len(ELFFailList)))
    with open('fails.log', 'w') as fail:
        json.dump(ELFFailList, fail, indent='\t')

if __name__ == '__main__':
    main()
