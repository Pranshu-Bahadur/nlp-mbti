from preprocess import generate_dataset
from transformers import AutoModelForSequenceClassification, Trainer, EvalPrediction
import numpy as np
from torch import nn
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader as Loader

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


def init_agent(agent_config, **kwargs) -> dict:
    agent = {}
    agent['model'] = AutoModelForSequenceClassification.from_pretrained(agent_config['name'], num_labels=agent_config['labels']).cuda()
    agent['train_dataset'],  agent['eval_dataset'], _ = _splitter(generate_dataset(agent_config['dataset_path'], agent_config['dataset_config']), 0.6)
    return agent

def run(mode : str, agent : dict, **kwargs) -> dict:
    kwargs['args'] = kwargs.pop('train_args')
    kwargs = {**agent, **kwargs}
    trainer = MultilabelTrainer(**kwargs, compute_metrics=_multilabel_accuracy)
    trainer.train()


def _splitter(dataset, *args):
    train_split = int(args[0] * len(dataset))
    eval_split = int(len(dataset) - train_split) // 2
    splits = [train_split, *list(eval_split for _ in range(2))]
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