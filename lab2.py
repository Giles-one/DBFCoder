import json
import torch
import argparse

from tqdm import tqdm
from typing import Dict, Callable

from DBFCoder import DBFCoder
from utils import DBFCoderDatset, DBFCollate
from torch.utils.data import DataLoader

import logging
logging.basicConfig(
    level=logging.ERROR
)

def createCrossOptimizationFilterFunction(anchorOption: Dict, positiveOption: Dict, interRelation: Dict) -> Callable:
    def filterFunction(self: DBFCoderDatset, fnPiar) -> bool:
        anchorFunctionIdx, positiveFunctionIdx = fnPiar
        if anchorFunctionIdx == positiveFunctionIdx:
            return False
        anchor, positive = (
            self.functionList[anchorFunctionIdx],
            self.functionList[positiveFunctionIdx]
        )

        for key in anchorOption.keys():
            if anchor[key] != anchorOption[key]:
                return False
        for key in positiveOption.keys():
            if positive[key] != positiveOption[key]:
                return False

        for k, v in interRelation.items():
            if v == 'equal' and anchor[k] != positive[k]:
                return False
            if v == 'differ' and anchor[k] == positive[k]:
                return False

        return True

    return filterFunction

def evaluateHandler(modelPath: str, anchorOption, positiveOption, interRelation, saveTo, **kwargs):
    y_score = []
    numBatch = 500
    batchSize = 2

    if 'numBatch' in kwargs.keys():
        numBatch = kwargs.get('numBatch')
    if 'batchSize' in kwargs.keys():
        batchSize = kwargs.get('batchSize')

    device = torch.device(
        "cuda" if torch.cuda.is_available() else 'cpu'
    )

    with open('model/config.json', 'r') as fp:
        config = json.load(fp)
    model = DBFCoder(config)

    modelWeight = torch.load(modelPath, map_location=device)
    model.load_state_dict(modelWeight)
    asmTokenizer, srcTokenizer = model.tokenizer

    evaluateDataset = DBFCoderDatset(
        'datasets/evaluation',
        shuffle=True,
        groupPattern='permutation'  # random | permutation | combination
    )
    evaluateDataset.checker()
    evaluateDataset.filter(
        function=createCrossOptimizationFilterFunction(
            anchorOption=anchorOption,
            positiveOption=positiveOption,
            interRelation=interRelation
        )
    )
    # breakpoint()
    # exit(-1)
    evaluateDataloader = DataLoader(
        dataset=evaluateDataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=DBFCollate,
        drop_last=True
    )
    if len(evaluateDataset) < numBatch:
        numBatch = len(evaluateDataset)

    model.eval()
    torch.no_grad()
    model.to(device)

    progressBar = tqdm(total=numBatch, desc='Evaluating...')
    for batchIndex, batch in enumerate(evaluateDataloader):
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
        # breakpoint()
        scores = out['scores'].tolist()
        y_score.append(scores)
        progressBar.update(1)
        if batchIndex >= numBatch:
            break

    recordObj = {
        'anchorOption': anchorOption,
        'positiveOption': positiveOption,
        'interRelation': interRelation,
        'y_score': y_score
    }
    with open(saveTo, 'w') as fp:
        json.dump(recordObj, fp, indent='\t')

def evaluate(args):
    anchorOption = {
        'compiler': args.anchorCompiler,
        'optimization': args.anchorOptimization
    }
    positiveOption = {
        'compiler': args.positiveCompiler,
        'optimization': args.positiveOptimization
    }
    interRelation = {
        'arch': 'equal',
        'bitness': 'equal'
    }
    saveTo = 'experiment/lab2_%s-%sVS%s-%s.json' % (
        anchorOption['compiler'],
        anchorOption['optimization'],
        positiveOption['compiler'],
        positiveOption['optimization']
    )
    print(f'anchorOption: {anchorOption}')
    print(f'positiveOption: {positiveOption}')
    print(f'interRelation: {interRelation}')
    print('saving to: ', saveTo)

    evaluateHandler(
        args.model,
        anchorOption,
        positiveOption,
        interRelation,
        saveTo,
        batchSize=100,
        numBatch=10
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DBFCoder lab2')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--anchorCompiler', type=str, default='clang-4.0')
    parser.add_argument('--anchorOptimization', type=str, default='O0')
    parser.add_argument('--positiveCompiler', type=str, default='clang-4.0')
    parser.add_argument('--positiveOptimization', type=str, default='O1')
    args = parser.parse_args()

    evaluate(args)




'''

'''

