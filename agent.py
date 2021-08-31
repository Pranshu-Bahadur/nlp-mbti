from preprocess import generate_dataset
from transformers import AutoModelForSequenceClassification, Trainer, EvalPrediction
import numpy as np
from torch import nn
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader as Loader

def init_agent(name : str, path : str, num_labels : int, train_split_factor : int, **kwargs) -> dict:
    agent = {"model": AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels).cuda()}
    agent['train_dataset'],  agent['eval_dataset'], _ , _= _splitter(generate_dataset(path, kwargs['dataset_config']), train_split_factor)
    return agent

def run(mode : str, agent : dict, **kwargs) -> dict:
    kwargs = {**agent, **kwargs}
    print(kwargs)
    trainer = MultilabelTrainer(**kwargs, compute_metrics=_multilabel_accuracy) if kwargs.pop('multilabel') else SinglelabelTrainer(**kwargs, compute_metrics=_accuracy)
    trainer.train()

#TODO add the following methods to utils


class SinglelabelTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loader = Loader(self.train_dataset, self.args.train_batch_size, shuffle=False, num_workers=4)
        labels = torch.stack([x['labels'] for x in loader][:-1])
        weights = torch.tensor([(labels==i).cpu().sum().item() for i in range(16)]).float()
        weights /= weights.sum().item()
        weights =  1 - weights
        #weights[2:] = 0.5
        #weights=torch.tensor([0.5, 0.5, 0.9, 0.9]).float()
        #print(weights)
        self.loss_fct = kwargs["loss"].weights=weights.cuda()#pos_weight=weights
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(labels.size(0),-1), labels)
        return (loss, outputs) if return_outputs else loss
     """



class MultilabelTrainer(Trainer):
    def __init__(self, **kwargs):
        
        self.loss_fct = kwargs.pop("loss")
        weights=torch.tensor([0.75, 0.75, 0.5, 0.5]).float()
        #print(weights)
        self.loss_fct.weight = weights.cuda()#pos_weight=weights

        super().__init__(**kwargs)
        """
        loader = Loader(self.train_dataset, self.args.train_batch_size, shuffle=False, num_workers=4)
        labels = torch.stack([x['labels'] for x in loader][:-1])
        weights = torch.tensor([labels[:,i].cpu().sum().item() for i in range(4)]).float()
        weights /= weights.sum().item()
        weights =  1 - weights
        weights[2:] = 0.5
        """

        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.float())
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 1e-5, betas=(0.999, 0.9), weight_decay=1e5)
        return self.optimizer


def _splitter(dataset, *args):
    train_split = int(args[0] * len(dataset))
    eval_split = int(len(dataset) - train_split) // 2
    splits = [train_split, *list(eval_split for _ in range(2))]
    splits.append(len(dataset) - sum(splits))
    return torch.utils.data.dataset.random_split(dataset, splits)

def _multilabel_accuracy(outputs : EvalPrediction):
    temp = 0
    h, y_true = outputs.predictions, outputs.label_ids
    h = torch.sigmoid(torch.from_numpy(h))
    print(h[0])
    #h[:,:-2] = torch.where(h[:,:-2] >= 0.6, 1, 0)
    h = torch.where(h >= 0.5, 1, 0)
    y_pred = h.long().cpu().detach().numpy()#np.vectorize(f)(h.unsqueeze(-1).cpu().detach().numpy())
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    print(y_true.shape)
    return {"accuracy" : temp/(y_true.shape[0])}

def _accuracy(outputs : EvalPrediction):
    y_pred, y_true = outputs.predictions, outputs.label_ids
    accuracy = np.sum(np.argmax(y_pred, axis=-1) == y_true) / y_true.shape[0]
    return {"accuracy" : accuracy}

