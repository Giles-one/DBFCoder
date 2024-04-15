import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_MODE"] = "online"

import wandb
import ijson
import torch
import random
from datetime import datetime

from collections import defaultdict
from DBFCoder import DBFCoder

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    def __init__(self, datasetPath: str, shuffle: bool = False, maxNumFunc: int = None):
        self.maxNumFunc = maxNumFunc
        self.shuffle = shuffle
        self.functionList = []
        self.jsonFileLists = []
        self.groupedFunctionIndex = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
        self.sameSourceFunctionPairs = []
        self._load(datasetPath, maxNumFunc)
        if self.shuffle:
            random.shuffle(self.sameSourceFunctionPairs)

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

    def _load(self, datasetPath: str, maxNumFunc: int):
        self.jsonFileLists = []
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
                    for functionIndex in group:
                        peerFunctionIndex = random.choice(group)
                        while peerFunctionIndex == functionIndex:
                            peerFunctionIndex = random.choice(group)
                        self.sameSourceFunctionPairs.append(
                            (functionIndex, peerFunctionIndex)
                        )
                    '''
                    (functionIndex - peerFunctionIndex) has the same source.
                    Algo:
                    1: group = [A, B, C, D]
                    2: sameSourceFunctionPair = []
                    3: for a in group:
                    4:     b = random.choice(group.remove(a))
                    5:     sameSourceFunctionPair.append((a, b)) 
                    So ugly code structure again.
                    '''
                    logger.debug(f'handle {fn}:{file}@{project} done.')
        logger.info(f'Got {len(self.sameSourceFunctionPairs)} pairs same source function.')

def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

def validate(model, asmTokenizer, srcTokenizer, validationDataloader, device):
    totalLoss = 0
    with torch.no_grad():
        predLists = []
        labelLists = []
        for batchIdx, batch in enumerate(validationDataloader):
            logger.debug(f'Validate steps: [{batchIdx}/{len(validationDataloader)}]')
            asmInput = asmTokenizer(
                batch['asmCode'],
                padding=True,
                return_tensors="pt"
            )
            srcInput = srcTokenizer(
                batch['pseudoCode'],
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            asmInput = {k: v.to(device) for k, v in asmInput.items()}
            srcInput = {k: v.to(device) for k, v in srcInput.items()}
            out = model(asmInput, srcInput, isTrain=False)

            loss = out['loss'].item()
            totalLoss += loss

            preds = out['preds'].tolist()
            labels = list(range(len(preds)))
            predLists += preds
            labelLists += labels
        avgLoss = totalLoss / len(validationDataloader)
        metrics = compute_metrics(labelLists, predLists)
        return {
            'loss': avgLoss,
            **metrics
        }

def train():
    '''super parameter'''
    numEpoch = 2
    saveModelPath = 'model/save'
    numStepsToValidate = 100
    numStepsToSaveModel = 100
    learningRate = 1e-5
    trainBatchSize = 48
    validationBatchSize = 48
    wandb.init(
        project='GraduationProject',
        name=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        config={
            "architecture": "DBFCoder",
            "learningRate": learningRate,
            "dataset": "GNU_OBFUS",
            "numEpoch": numEpoch,
        }
    )

    device = torch.device(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainDataset = DBFCoderDatset(
        'datasets/train',
        shuffle=True,
        maxNumFunc=400000,
    )
    trainDataset.checker()
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=trainBatchSize,
        shuffle=True,
        collate_fn=DBFCollate,
        drop_last=True
    )

    validationDataset = DBFCoderDatset(
        'datasets/validation',
        shuffle=True,
        maxNumFunc=1000,
    )
    validationDataset.checker()
    validationDataloader = DataLoader(
        dataset=validationDataset,
        batch_size=validationBatchSize,
        shuffle=True,
        collate_fn=DBFCollate,
        drop_last=True
    )

    config = {
        'asmEncoder': 'model/asmCodeEncoer/clap-asm/',
        'srcEncoder': 'model/sourceCodeEncoder/codet5p-110m-embedding/'
    }
    model = DBFCoder(config)
    model.to(device)
    asmTokenizer, srcTokenizer = model.tokenizer

    optimizer = AdamW(
        model.parameters(),
        lr=learningRate
    )
    totalSteps = len(trainDataloader) * numEpoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=totalSteps
    )

    steps = 0
    for epoch in range(numEpoch):
        model.train()
        for batchId, batch in enumerate(trainDataloader):
            logger.info(f'Epoch: [{epoch}/{numEpoch}], Batch: [{batchId}/{len(trainDataloader)}]')
            # exit(0)
            asmInput = asmTokenizer(
                batch['asmCode'],
                padding=True,
                return_tensors="pt"
            )
            srcInput = srcTokenizer(
                batch['pseudoCode'],
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            asmInput = {k: v.to(device) for k, v in asmInput.items()}
            srcInput = {k: v.to(device) for k, v in srcInput.items()}
            out = model(asmInput, srcInput, isTrain=True)

            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            steps += 1

            if steps % numStepsToValidate == 0:
                model.eval()
                metrics = validate(
                    model,
                    asmTokenizer,
                    srcTokenizer,
                    validationDataloader,
                    device
                )
                print(metrics)
                wandb.log(metrics, step=steps)
                model.train()

            if steps % numStepsToSaveModel == 0:
                saveTo = f'{saveModelPath}/model_{steps}_steps.pt'
                torch.save(model.state_dict(), saveTo)
                logger.info(f'Save to {saveTo}')


if __name__ == '__main__':
    train()

