import os
import ijson
import random

from typing import Callable
from collections import defaultdict
from torch.utils.data import Dataset

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]->[%(asctime)s]-> %(message)s'
)
logger = logging.getLogger(__name__)

MERGED_FUNCTION_INFO_SUFFIX = '.functionInfo.json'

def DBFCollate(batch):
    batchFunctions = []
    for sameSourceFunctionPair in batch:
        anchorFunction = sameSourceFunctionPair[0]
        positiveFunction = sameSourceFunctionPair[1]
        batchFunctions.append(anchorFunction)
        batchFunctions.append(positiveFunction)

    elem = batchFunctions[0]

    batchedFunction = {
        key: [fn[key] for fn in batchFunctions] for key in elem.keys()
    }
    return batchedFunction

class DBFCoderDatset(Dataset):
    def __init__(self, datasetPath: str, shuffle: bool = False, maxNumFunc: int = None, randomGroup: bool = True):
        self.maxNumFunc = maxNumFunc
        self.shuffle = shuffle
        self.randomGroup = randomGroup

        self.functionList = []
        self.jsonFileLists = []
        self.groupedFunctionIndex = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
        self.sameSourceFunctionPairs = []

        self._load(datasetPath, maxNumFunc)

    def __len__(self):
        return len(self.sameSourceFunctionPairs)

    def __getitem__(self, idx):
        anchorIndex, positiveIndex = self.sameSourceFunctionPairs[idx]
        anchorFunction = self.functionList[anchorIndex]
        positiveFunction = self.functionList[positiveIndex]

        anchorFeature = {
            'asmCode': anchorFunction['strippedAsmCode'],
            'pseudoCode': anchorFunction['strippedPseudoCode'] if anchorFunction['strippedPseudoCode'] != None else ''
        }
        positiveFeature = {
            'asmCode': positiveFunction['strippedAsmCode'],
            'pseudoCode': positiveFunction['strippedPseudoCode'] if positiveFunction['strippedPseudoCode'] != None else ''
        }

        return anchorFeature, positiveFeature

    def checker(self):
        for index, fn in enumerate(self.functionList):
            assert fn['index'] == index, f'Error {index}'

    def filter(self, function: Callable):
        self.sameSourceFunctionPairs = [
            fnPair
            for fnPair in self.sameSourceFunctionPairs
            if function(self, fnPair)
        ]
        logger.info(f'filtered sameSourceFunctionPairs: {len(self.sameSourceFunctionPairs)}')
        if self.shuffle:
            random.shuffle(self.sameSourceFunctionPairs)

    def _load(self, datasetPath: str, maxNumFunc: int):
        for root, dirs, files in os.walk(datasetPath):
            for file in files:
                if not file.endswith(MERGED_FUNCTION_INFO_SUFFIX):
                    continue
                filePath = os.path.join(root, file)
                self.jsonFileLists.append(filePath)
        logger.info(f'found {MERGED_FUNCTION_INFO_SUFFIX}: {len(self.jsonFileLists)} ')

        functionIdx = 0
        for jsonFile in self.jsonFileLists:
            jumpOut = False
            with open(jsonFile, 'rb') as fp:
                fnInfo = ijson.items(fp, 'item')
                for fn in fnInfo:
                    self.functionList.append({
                        'index': functionIdx,
                        **fn
                    })
                    project, file, functionName = (
                        fn['project'],
                        fn['file'],
                        fn['functionName']
                    )
                    self.groupedFunctionIndex[project][file][functionName].append(functionIdx)
                    functionIdx += 1
                    if self.maxNumFunc and functionIdx >= self.maxNumFunc:
                        jumpOut = True
                        break
                if jumpOut:
                    break
        logger.info(f'found {len(self.functionList)} functions.')
        self.group()

    def group(self):
        for project, files in self.groupedFunctionIndex.items():
            for file, functions in files.items():
                for fn, group in functions.items():
                    '''
                    So ugly code structure.
                    group means the same source functions with fn.
                    '''
                    if len(group) < 2:
                        '''
                        It mean no peer function.
                        '''
                        continue
                    if self.randomGroup:
                        for functionIndex in group:
                            peerFunctionIndex = random.choice(group)
                            while peerFunctionIndex == functionIndex:
                                peerFunctionIndex = random.choice(group)
                            self.sameSourceFunctionPairs.append(
                                (functionIndex, peerFunctionIndex)
                            )

                    else:
                        for functionIndex in group:
                            for peerFunctionIndex in group:
                                self.sameSourceFunctionPairs.append(
                                    (functionIndex, peerFunctionIndex)
                                )

                    logger.debug(f'handle {fn}:{file}@{project} done.')
        logger.info(f'Got {len(self.sameSourceFunctionPairs)} pairs same source function.')

