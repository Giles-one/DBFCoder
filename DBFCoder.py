import torch

from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]->[%(asctime)s]-> %(message)s'
)
logger = logging.getLogger(__name__)

class MultipleNegativesRankingLoss:
    def __init__(self):
        super()

    def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def __call__(self, anchor_embs: List[torch.Tensor], pos_embs: List[torch.Tensor], scale: float = 20.0):
        scores = self.cos_sim(anchor_embs, pos_embs) * scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return torch.nn.functional.cross_entropy(scores, labels)

class DBFCoder(torch.nn.Module):
    def __init__(self, config: Dict[str, str]):
        super(DBFCoder, self).__init__()
        self.config = config
        self.embeddings = None
        self.asmTokenizer = None
        self.srcTokenizer = None
        self.asmEncoder = AutoModel.from_pretrained(
            self.config['asmEncoder'],
            trust_remote_code=True,
            local_files_only=True
        )
        self.srcEncoder = AutoModel.from_pretrained(
            self.config['srcEncoder'],
            trust_remote_code = True,
            local_files_only = True
        )
        self.asmEncoder.requires_grad_(False)
        self.srcEncoder.requires_grad_(False)
        self.srcEncoder.proj.requires_grad_(True)
        self.asmEncoder.projection.requires_grad_(True)

        self.linear = torch.nn.Linear(in_features=1024, out_features=256, bias=True)

    @property
    def tokenizer(self):
        self.asmTokenizer = AutoTokenizer.from_pretrained(
            self.config['asmEncoder'],
            trust_remote_code = True,
            local_files_only=True
        )
        self.srcTokenizer = AutoTokenizer.from_pretrained(
            self.config['srcEncoder'],
            trust_remote_code = True,
            local_files_only=True
        )
        return self.asmTokenizer, self.srcTokenizer

    @staticmethod
    def reshape_and_split_tensor(tensor, n_splits):
        feature_dim = tensor.shape[-1]
        tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
        tensor_split = []
        for i in range(n_splits):
            tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
        return tensor_split

    def forward(self, asmInput, srcInput, isTrain=True):
        asmEncoderOutput = self.asmEncoder(**asmInput)
        srcEncoderOutput = self.srcEncoder(**srcInput)
        concatOutput = torch.cat(
            tensors=(asmEncoderOutput, srcEncoderOutput),
            dim=-1
        )

        embeddings = self.linear(concatOutput)
        anchorEmbeddings, positiveEmbeddings = self.reshape_and_split_tensor(
            embeddings, 2
        )
        # check result
        '''
        for i in range(len(embeddings)):
            if i % 2 == 0:   # even number
                assert embeddings[i].equal(anchorEmbeddings[i//2])
            else:            # odd number
                assert embeddings[i].equal(queryEmbeddings[(i-1) // 2])
        '''
        lossFunction = MultipleNegativesRankingLoss()
        if isTrain:
            multi_neg_ranking_loss = lossFunction(anchorEmbeddings, positiveEmbeddings)
            scores = None
            preds = None
        else:
            multi_neg_ranking_loss = lossFunction(anchorEmbeddings, positiveEmbeddings)
            scores = lossFunction.cos_sim(anchorEmbeddings, positiveEmbeddings)
            preds = scores.argmax(dim=-1)
        return {
            'loss': multi_neg_ranking_loss,
            'scores': scores,
            'preds': preds
        }
