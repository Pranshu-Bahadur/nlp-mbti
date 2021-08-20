from preprocess import generate_dataset
from transformers import AutoModelForSequenceClassification, Trainer, EvalPrediction
import numpy as np
from torch import nn
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader as Loader

<<<<<<< HEAD
def init_agent(name : str, path : str, num_labels : int, train_split_factor : int, **kwargs) -> dict:
    agent = {"model": AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels).cuda()}
    agent['train_dataset'],  agent['eval_dataset'], _ , _= _splitter(generate_dataset(path, kwargs['dataset_config']), train_split_factor)
    return agent

def run(mode : str, agent : dict, **kwargs) -> dict:
    kwargs = {**agent, **kwargs}
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
        self.loss_fct = nn.CrossEntropyLoss(weight=weights).cuda()#pos_weight=weights
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
        super().__init__(**kwargs)
        """
        loader = Loader(self.train_dataset, self.args.train_batch_size, shuffle=False, num_workers=4)
        labels = torch.stack([x['labels'] for x in loader][:-1])
        weights = torch.tensor([labels[:,i].cpu().sum().item() for i in range(4)]).float()
        weights /= weights.sum().item()
        weights =  1 - weights
        weights[2:] = 0.5
        """
        weights=torch.tensor([0.75, 0.75, 0.5, 0.5]).float()
        #print(weights)
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights).cuda()#pos_weight=weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.float())
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 1e-5, betas=(0.999, 0.9), weight_decay=1e5)
        return self.optimizer

=======
class MultilabelTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fct = nn.BCEWithLogitsLoss()
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss
""" 
    def get_train_dataloader(self):
        def distribution():
            loader = Loader(self.train_dataset, 256*4, shuffle=False, num_workers=4)
            train_Y = torch.cat([data["labels"] for data in loader])
            train_split_dist = [(train_Y==i).sum().cpu().item()/train_Y.size(0) for i in range(4)]
            return train_split_dist
        class_weights = distribution()
        sampler_loader = Loader(self.train_dataset, 256*4, shuffle=False, num_workers=4)
        target = torch.cat([data["labels"] for data in sampler_loader])
        samples_weight = np.array([class_weights[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return torch.utils.data.DataLoader(self.train_dataset, 256*4, shuffle=False, sampler=sampler, num_workers=4)
"""


def init_agent(name : str, path : str, num_labels : int, **kwargs) -> dict:
    agent = {}
    agent['model'] = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels).cuda()
    agent['train_dataset'],  agent['eval_dataset'], _ = _splitter(generate_dataset(path, kwargs['dataset_config']), 0.6)
    return agent

def run(mode : str, agent : dict, **kwargs) -> dict:
    kwargs['args'] = kwargs.pop('train_args')
    kwargs = {**agent, **kwargs}
    trainer = MultilabelTrainer(**kwargs, compute_metrics=_multilabel_accuracy)
    trainer.train()

>>>>>>> c27c6baff1969c1a12452103378a5b9e21bf5d93

def _splitter(dataset, *args):
    train_split = int(args[0] * len(dataset))
    eval_split = int(len(dataset) - train_split) // 2
    splits = [train_split, *list(eval_split for _ in range(2))]
<<<<<<< HEAD
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
=======
    return torch.utils.data.dataset.random_split(dataset, splits)


def _multilabel_accuracy(outputs : EvalPrediction):
    temp = 0
    y_pred, y_true = outputs.predictions, outputs.label_ids
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return {"accuracy" : temp / y_true.shape[0]}

def _accuracy(outputs : EvalPrediction):
    y_pred, y_true = outputs.predictions, outputs.label_ids
    accuracy = sum(np.argmax(y_pred, axis=1) == y_true) / y_true.shape[0]
    return {"accuracy" : accuracy}
>>>>>>> c27c6baff1969c1a12452103378a5b9e21bf5d93
