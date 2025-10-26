from torch import softmax,argmax
import torch.nn.functional as F

def compute_metrics(logits):
    return softmax(logits, dim= 1)

def accuracy_metric(logits, labels):

    preds = argmax(logits, dim =1)
    return (preds==labels).float().mean()


