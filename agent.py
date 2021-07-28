import torch
import pandas as pd
import numpy as np

"""

Controller : (Basically _run function in experiment.py in a loop ran based on some logic)
Agent : Run the model on the dataset based on controller's conditions for N epochs (controller defined)



run controller main function for until we get desired:
    rewards = []
    agent is a BertTweetModel
    while condition_is_true:
        agent.run(config_dict) #For 8 epochs or something
        rewards.append(agent.val_acc_avg*(1./len(rewards)))
        current_config_expectation = sum(rewards)
        if decresing_increasing:
            decision


"""



"""
    Represents an instance of a nlp text classification model, that can do the following:
    - Instantiate a Dataset object based on user given file path.
    - Call preprocessing dataframe object (object because does too much stuff)
    - Instantiate a torch.utils.data.Dataset object child.
    - Tokenize using user given/ model's tokenizer. 
    - Handle Imbalance iff given Dataset is imbalanced.
    - Split a Dataset, based on user's given split description: For example:(train, val, test) (60, 20, 20), 3
        - First split is *always* the train split.
    - 

======

Text-Classification DS .csv [Single Class per input]
            Label, Data 

"""
class Agent():
    pass

""" Represents a Pandas DF Preprocessor for Text Classification.
    - Preprocess a Dataset according to user input.
        - Examples Args (Explode?, filter_out_word (should be atleast 3 chars), filter_by_lowest_wc, filter_by_highest_wc, user input de-limitter if needed)
    -have/return some how the dataframe.
"""
class DataFramePreprocessor():
    """
        Constructor for DataFramePreprocessor Object.
        Args:
            - path : string or os.PathLike
            - **kwargs:
                - explode: None or str
    """
    def __init__(self, path : str, **kwargs: dict):
        self.df = pd.read_csv(path)
        self.kwargs = kwargs
        #self.choices = {"explode": self.explode()}
        #[self.choices[k] for k,v in kwargs.items() if v != None]

    # Only two column support.
    def _explode(self):
        # https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
        col_names = self.df.columns.values.tolist()
        gen_names = ['target','input']
        df = self.df.rename(columns={col_names[i]: gen_names[i] for i in range(2)})
        col_names.reverse()
        df = pd.concat([Series(row['target'], row['input'].split(self.kwargs["explode"]))for _, row in a.iterrows()]).reset_index()
        self.df = self.df.rename(columns={cols: col_names[i] for i, cols in enumerate(df.columns.values.tolist())})

        








