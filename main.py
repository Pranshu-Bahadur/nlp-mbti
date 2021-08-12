import argparse

def model_config(args):
    pass

def create_optimizer():
    pass

def loss_criterion():
    pass

def config_agent():
    pass

if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()

    # Passing the command arguments
    parse.add_argument("--model_name", "-m", help="Pick a model name")
    parse.add_argument("--dataset_directory", "-d", help="Set dataset directory path")
    parse.add_argument("--delimiter", "-dl", help="Enter a delimiter")
    parse.add_argument("--word_limit", "-w", help="Enter minimum words per post")
    parse.add_argument("--batch_size", "-b", help="Set batch size")
    parse.add_argument("--learning_rate", "-l", help="set initial learning rate")
    parse.add_argument("--num_classes", "-n", help="set num classes")
    parse.add_argument("--epochs", "-f", help="Train for these many more epochs")
    parse.add_argument("--optimizer", help="Choose an optimizer")
    parse.add_argument("--loss", help="Choose a loss criterion")
    parse.add_argument("--train", help="Set this model to train mode", action="store_true")
    parse.add_argument("--library")
    parse.add_argument("--save_directory", "-s")
    parse.add_argument("--save_interval", help="# of epochs to save checkpoints at.")

    