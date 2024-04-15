import json
import logging

from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]->[%(asctime)s] -> %(message)s'
)
logger = logging.getLogger(__name__)
def diaplyROC(fpr, tpr, aucValue, saveTo: str) -> None:
    lw = 2
    bwith = 2
    aucValue = ('%.4f' % aucValue)
    plt.plot(fpr, tpr, color='tomato', lw=lw, label='DBFCoder (area = ' + str(aucValue) + ')')
    plt.legend(loc="lower right", prop={'size': 12})
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(color='silver', linestyle='-.', linewidth=1)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.xlabel('False Positive Rate', fontdict={'size': 14})
    plt.ylabel('True Positive Rate', fontdict={'size': 14})
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.tick_params(width=2, labelsize=12)
    plt.show()
    plt.savefig(saveTo)


def lab1_O0XO1():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-4.0-O1.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC O0xO1: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_O0XO1.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC O0xO1: {imagePath}')

def lab1_O0XO2():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-4.0-O2.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC O0xO2: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_O0XO2.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC O0xO2: {imagePath}')

def lab1_O0XO3():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-4.0-O3.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC O0xO3: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_O0XO3.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC O0xO3: {imagePath}')

def lab1_O1XO2():
    dataFile = 'experiment/lab1_clang-4.0-O1VSclang-4.0-O2.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC O1xO2: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_O1XO2.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC O1xO2: {imagePath}')

def lab1_O1XO3():
    dataFile = 'experiment/lab1_clang-4.0-O1VSclang-4.0-O3.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC O1xO3: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_O1XO3.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC O1xO3: {imagePath}')

def lab1_O2XO3():
    dataFile = 'experiment/lab1_clang-4.0-O2VSclang-4.0-O3.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC O2xO3: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_O2XO3.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC O2xO3: {imagePath}')

def lab1_sub():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-obfus-sub-O0.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC sub: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_sub.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC sub: {imagePath}')

def lab1_bcf():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-obfus-bcf-O0.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC bcf: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_bcf.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC bcf: {imagePath}')

def lab1_fla():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-obfus-fla-O0.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC fla: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_fla.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC fla: {imagePath}')

def lab1_all():
    dataFile = 'experiment/lab1_clang-4.0-O0VSclang-obfus-all-O0.json'
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yTrue = data['y_true']
    yScore = data['y_score']
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    aucValue = auc(fpr, tpr)
    logger.info(f'AUC all: {aucValue}')

    imagePath = 'experiment/images/lab1_ROC_all.png'
    diaplyROC(fpr, tpr, aucValue, imagePath)
    logger.info(f'ROC all: {imagePath}')


def getrRecallAtK(labels, scores, atK):
    socresIndex = list(range(len(scores)))
    sortedSocresIndex = sorted(
        socresIndex,
        key=lambda i: scores[i],
        reverse=True
    )
    TP = sum(labels[i] for i in sortedSocresIndex[:atK])
    TPAndFN = sum(labels)
    return TP / TPAndFN

def getMRRAtK(labels, scores, atK):
    socresIndex = list(range(len(scores)))
    sortedSocresIndex = sorted(
        socresIndex,
        key=lambda i: scores[i],
        reverse=True
    )
    MRRAtK = sum(
        1 / (i + 1)
        for i, index in enumerate(sortedSocresIndex[:atK])
        if labels[index]
    )
    return MRRAtK

def lab2_tmplate(banner, dataFile):
    with open(dataFile, 'r') as fp:
        data = json.load(fp)
    yScore = data['y_score']

    numSample = 0
    MRR = defaultdict(int)
    recall = defaultdict(int)
    for batch in yScore:
        for positiveIdx, scores in enumerate(batch):
            sampleLength = len(scores)
            labels = [1 if idx==positiveIdx else 0 for idx in range(sampleLength)]
            for topK in (1, 3, 5):
                MRRAtK = getMRRAtK(labels, scores, topK)
                MRR[f'MRR@{topK}'] += MRRAtK

                recallAtK = getrRecallAtK(labels, scores, topK)
                recall[f'recall@{topK}'] += recallAtK

            numSample += 1

    recall = {k: v / numSample for k, v in recall.items()}
    MRR = {k: v / numSample for k, v in MRR.items()}
    for k, v in MRR.items():
        logger.info(f'{banner} {k}: {v}')
    for k, v in recall.items():
        logger.info(f'{banner} {k}: {v}')

def lab1():
    lab1_O0XO1()
    lab1_O0XO2()
    lab1_O0XO3()

    lab1_O1XO2()
    lab1_O1XO3()

    lab1_O2XO3()

    lab1_sub()
    lab1_bcf()
    lab1_fla()
    lab1_all()

def lab2():
    lab2_tmplate('lab2_O0XO1', 'experiment/lab2_clang-4.0-O0VSclang-4.0-O1.json')
    lab2_tmplate('lab2_O0XO2', 'experiment/lab2_clang-4.0-O0VSclang-4.0-O2.json')
    lab2_tmplate('lab2_O0XO3', 'experiment/lab2_clang-4.0-O0VSclang-4.0-O3.json')
    lab2_tmplate('lab2_O1XO2', 'experiment/lab2_clang-4.0-O1VSclang-4.0-O2.json')
    lab2_tmplate('lab2_O1XO3', 'experiment/lab2_clang-4.0-O1VSclang-4.0-O3.json')
    lab2_tmplate('lab2_O2XO3', 'experiment/lab2_clang-4.0-O2VSclang-4.0-O3.json')

    lab2_tmplate('lab2_obfus_sub', 'experiment/lab2_clang-4.0-O0VSclang-obfus-sub-O0.json')
    lab2_tmplate('lab2_obfus_bcf', 'experiment/lab2_clang-4.0-O0VSclang-obfus-bcf-O0.json')
    lab2_tmplate('lab2_obfus_fla', 'experiment/lab2_clang-4.0-O0VSclang-obfus-fla-O0.json')
    lab2_tmplate('lab2_obfus_all', 'experiment/lab2_clang-4.0-O0VSclang-obfus-all-O0.json')


if __name__ == '__main__':
    lab1()
    lab2()
