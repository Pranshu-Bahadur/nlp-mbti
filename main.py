import argparse
from torch import nn
import agent
from transformers import AutoTokenizer, TrainingArguments
#import tensorflow as tf
import torch


def create_optimizer(name: str, learning_rate: float, agent):
    optimizer = {
        "ADAM": torch.optim.Adam(agent['model'].parameters(), learning_rate, betas=(0.9, 0.999), eps=1e-8),
        "SGD":  torch.optim.SGD(agent['model'].parameters(), learning_rate, weight_decay=1e-5, momentum=0.9, nesterov=True),
        "ADAMW": torch.optim.AdamW(agent['model'].parameters(), lr=learning_rate,betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8, amsgrad=True)
    }
    return optimizer[name]

def loss_criterion(name: str):
    loss = {
        "CCE": nn.CrossEntropyLoss().cuda(),
        "MML": nn.MultiMarginLoss().cuda(),
        "BCE": nn.BCEWithLogitsLoss().cuda()
    }
    return loss[name]

'''def create_metrics(name: str):
    metrics = {
        "CCE": nn.CrossEntropyLoss().cuda(),
        "MML": nn.MultiMarginLoss().cuda(),
        "MSE": nn.MSELoss().cuda(),
        "BCE": nn.BCELoss().cuda(),
        "SCA": tf.metrics.SparseCategoricalAccuracy(name="train_accuracy"),
        "BA": tf.metrics.BinaryAccuracy(name="train_accuracy"),
        "CA": tf.metrics.CategoricalAccuracy(name="train_accuracy"),
        "STK": tf.metrics.SparseTopKCategoricalAccuracy(name="train_accuracy"),
        "TKC": tf.metrics.TopKCategoricalAccuracy(name="train_accuracy")
    }
    return metrics[name]'''

def configure_model(args):
    model_config = {
        "model_name": args.model_name,
        "dataset_directory": args.dataset_directory,
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "train_split": float(args.train_split),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "classes": int(args.num_classes),
        "multilabel": True if args.multilabel else False,
        "epochs": int(args.epochs),
        #"optimizer": create_optimizer(args.optimizer, float(args.learning_rate)),
        "loss": loss_criterion(args.loss),
        "train": True if args.train else False,
        "dataset_config": [ [int(args.word_limit),args.delimiter,True], {
                            "tokenizer": AutoTokenizer.from_pretrained(args.model_name, normalize=True),
                            "max_length": 40,
                            "truncation": True,
                            "padding": "max_length" }, set()],
        #"metrics": create_metrics(args.metrics),
        "output_directory": args.output_directory     
    }
    return model_config

def configure_agent(config):

    agent_config = {
        "model": config["model_name"],
        "dataset_path": config["dataset_directory"],
        "labels": 4,
        "dataset_config": config["dataset_config"],
        "train_split": config["train_split"],
        "classes": config['classes'],
        "loss": config["loss"]
    }
    
    return agent_config

if __name__ == "__main__":
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    parse = argparse.ArgumentParser()

    # Passing the command arguments.
    parse.add_argument("--model_name", "-m", help="Pick a model name")
    parse.add_argument("--dataset_directory", "-d", help="Set dataset directory path")
    parse.add_argument("--delimiter", "-dl", help="Enter a delimiter")
    parse.add_argument("--word_limit", "-w", help="Enter minimum words per post")
    parse.add_argument("--train_batch_size", "-tb", help="Set batch size")
    parse.add_argument("--eval_batch_size", "-eb", help="Set batch size")
    parse.add_argument("--train_split", "-r", help="Set the train, test split ratio")
    parse.add_argument("--learning_rate", "-l", help="set initial learning rate")
    parse.add_argument("--weight_decay", "-wd", help="Set weight decay")
    parse.add_argument("--num_classes", "-n", help="set num classes")
    parse.add_argument("--multilabel", "-ml")
    parse.add_argument("--epochs", "-f", help="Train for these many more epochs")
    #parse.add_argument("--metrics", "-mt", help="Set metrics")
    parse.add_argument("--optimizer", help="Choose an optimizer")
    parse.add_argument("--loss", help="Choose a loss criterion")
    parse.add_argument("--train", help="Set this model to train mode", action="store_true")
    parse.add_argument("--library")
    parse.add_argument("--output_directory", "-o", help="Enter the path of directory to save the output")
    parse.add_argument("--save_interval", help="# of epochs to save checkpoints at.")

    # Retrieving the model configuration from the passed arguments.
    args = parse.parse_args()
    model_config = configure_model(args)

    # Retrieve the training arguments.
    train_args = {

        "do_train": model_config["train"],

        "per_device_train_batch_size": model_config["train_batch_size"],

        "per_device_eval_batch_size": model_config["eval_batch_size"],

        "learning_rate": model_config["learning_rate"],

        "weight_decay": model_config["weight_decay"],

        "num_train_epochs": model_config["epochs"],

        "logging_strategy": "epoch",

        "seed": 420,

        "output_dir": model_config["output_directory"],

        "do_eval": True,
        
        "dataloader_num_workers": 4,

        "evaluation_strategy": "steps",

        "logging_dir": "./logs",

        "logging_strategy": "steps",

        "logging_steps": 1000

    }
    
    train_args = TrainingArguments(**train_args)

    # Retrieve agent configuration.
    agent_config = configure_agent(model_config)

    # Call the agent to initialize the model and run it.
    _agent = agent.init_agent(agent_config['model'], agent_config['dataset_path'], agent_config['classes'], agent_config['train_split'], dataset_config= agent_config['dataset_config'])

    # Since we need the model to pass its parameters to create optimizer.
    #model_config['optimizer'] = create_optimizer(args.optimizer, float(args.learning_rate), _agent)

    agent.run("train", _agent, args=train_args, multilabel=True, loss=agent_config["loss"])
