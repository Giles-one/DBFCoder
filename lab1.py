import json
import torch
import argparse

from tqdm import tqdm
from typing import Dict, Callable

from DBFCoder import DBFCoder
from utils import DBFCoderDatset, DBFCollate
from torch.utils.data import DataLoader


def createCrossOptimationFilterFunction(anchorOption: Dict, positiveOption: Dict, interRelation: Dict) -> Callable:
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


def evaluathandler(anchorOption, positiveOption, interRelation, saveTo):
    y_true = []
    y_score = []
    numBatch = 30
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 'cpu'
    )

    with open('model/config.json', 'r') as fp:
        config = json.load(fp)
    model = DBFCoder(config)

    modelWeight = torch.load('../best_model.pt', map_location=device)
    model.load_state_dict(modelWeight)
    asmTokenizer, srcTokenizer = model.tokenizer

    evaluateDataset = DBFCoderDatset(
        'datasets/evaluation',
        shuffle=True,
        maxNumFunc=100000,
        groupPattern='permutation'  # random | permutation | combination
    )
    evaluateDataset.checker()
    evaluateDataset.filter(
        function=createCrossOptimationFilterFunction(
            anchorOption=anchorOption,
            positiveOption=positiveOption,
            interRelation=interRelation
        )
    )
    # breakpoint()
    # exit(-1)
    evaluateDataloader = DataLoader(
        dataset=evaluateDataset,
        batch_size=2,
        shuffle=True,
        collate_fn=DBFCollate,
        drop_last=True
    )

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
        scores = out['scores'].tolist()
        for i, row in enumerate(scores):
            for j, score in enumerate(row):
                label = 1 if i == j else 0
                y_true.append(label)
                y_score.append(score)
        progressBar.update(1)
        if batchIndex > numBatch:
            break

    recordObj = {
        'anchorOption': anchorOption,
        'positiveOption': positiveOption,
        'interRelation': interRelation,
        'y_true': y_true,
        'y_score': y_score
    }
    with open(saveTo, 'w') as fp:
        json.dump(recordObj, fp, indent='\t')

def evaluate(args):
    anchorOption = {
        'compiler': args.anchorCompiler,
        'optimation': args.anchorOptimization
    }
    positiveOption = {
        'compiler': args.positiveCompiler,
        'optimation': args.positiveOptimization
    }
    interRelation = {
        'arch': 'equal',
        'bitness': 'equal'
    }
    saveTo = 'experiment/lab1_%s-%sVS%s-%s.json' % (
        anchorOption['compiler'],
        anchorOption['optimation'],
        positiveOption['compiler'],
        positiveOption['optimation']
    )
    print(f'anchorOption: {anchorOption}')
    print(f'positiveOption: {positiveOption}')
    print(f'interRelation: {interRelation}')
    print('saving to: ', saveTo)
    evaluathandler(anchorOption, positiveOption, interRelation, saveTo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DBFCoder')
    parser.add_argument('--anchorCompiler', type=str, default='clang-4.0')
    parser.add_argument('--anchorOptimization', type=str, default='O0')
    parser.add_argument('--positiveCompiler', type=str, default='clang-4.0')
    parser.add_argument('--positiveOptimization', type=str, default='O1')
    args = parser.parse_args()
    evaluate(args)

