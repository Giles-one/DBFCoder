import json
import wandb
import torch

from datetime import datetime

from DBFCoder import DBFCoder
from utils import DBFCoderDatset, DBFCollate

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
    trainingSteps = 3
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
            "trainBatchSize": trainBatchSize,
            "validationBatchSize": validationBatchSize,
            "trainSteps": trainingSteps
        }
    )

    device = torch.device(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainDataset = DBFCoderDatset(
        'datasets/train',
        shuffle=True,
        maxNumFunc=400000,
        groupPattern='random'  # random | permutation | combination
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
        groupPattern='random'
    )
    validationDataset.checker()
    validationDataloader = DataLoader(
        dataset=validationDataset,
        batch_size=validationBatchSize,
        shuffle=True,
        collate_fn=DBFCollate,
        drop_last=True
    )

    with open('model/config.json', 'r') as fp:
        config = json.load(fp)
    model = DBFCoder(config)
    model.to(device)
    asmTokenizer, srcTokenizer = model.tokenizer

    optimizer = AdamW(
        model.parameters(),
        lr=learningRate
    )
    totalSteps = len(trainDataloader) * numEpoch
    logger.debug(f'totalSteps: {totalSteps}')
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

