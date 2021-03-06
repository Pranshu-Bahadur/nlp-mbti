import re
from pandas import read_csv, DataFrame, concat, Series
from functools import singledispatch, reduce
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

#======================================================================
#       Dataset preprocessing functions
#======================================================================

def generate_dataset(path : str, configs : list) -> Dataset:
    """
    Given a dictionary of instructions perform dynamic dispatch to pre-process
    .csv as a dataframe based on given strategy args, encode DataFrame and
    return tf.data.Dataset object.
    """
    return reduce(lambda x, y: _generate_dataset_helper(x, y), configs, path)

@singledispatch
def _generate_dataset_helper(path : str, args : list) ->  DataFrame:
    return nlp_tc_df_parser(path, *args)


@_generate_dataset_helper.register
def _tokenize(df : DataFrame, kwargs : dict) -> dict:
    tokenizer = kwargs.pop('tokenizer')
    encodings = dict(tokenizer(list(df['posts'].values), **kwargs))
    encodings['labels'] = list(df['type'].values)
    return encodings

class EncodedDataset(Dataset):
    def __init__(self, encodings):
        self._labels = encodings.pop('labels')
        self.encodings = encodings

    def __getitem__(self, idx):
        x = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        x['labels'] = torch.tensor(self._labels[idx])
        return x

    def __len__(self):
        return len(self._labels)
@_generate_dataset_helper.register
def _gen_tf_dataset(encodings : dict, kwargs : set) -> Dataset:
    return EncodedDataset(encodings)


#======================================================================
# Functions to preprocess a pandas.DataFrame from a csv
#======================================================================


def nlp_tc_df_parser(path : str, *args) -> DataFrame:
    """
    Given a path to an nlp text classification dataset (.csv),
    instantiate an instance of a DataFrame object and if needed perform
    cleaning procedures on it according to given kwargs.
    Parameters
    ----------
    path : str
         path to dataset
    *args : list
         additional user args
    Returns
    -------
    df : DataFrame
        parsed DataFrame of in given path
    """
    data_frame = reduce(lambda x, y: _parser(y, x), args, read_csv(path)) if len(args) > 0 else read_csv(path)
    return data_frame

#======================================
# Helper Functions for nlp_tc_df_parser:
#======================================
#TODO str labels to num
# Remove hyperlinks that end with .com

@singledispatch
def _parser(strategy, df) -> DataFrame:
    str_labels = pd.unique(df.type.values.tolist())
    labels_dict =  dict(zip(str_labels, list(range(len(str_labels)))))
    df['type'] = df['type'].apply(lambda x: labels_dict[x])
    return df


@_parser.register
def _hyper_link_cleaner(strategy : set, df)  -> DataFrame:
    df.posts = df.posts.str.replace(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', " ")
    return df

@_parser.register
def _remove_below_word_limit(strategy : int, df)  -> DataFrame:
    df["total_words"] = df.posts.str.split(" ").map(len)
    df = df[df["total_words"]>strategy]
    df = df.drop(columns=["total_words"])
    return df
# Splits input rows based on a given delimiter
@_parser.register
def _explode(strategy : str, df)  -> DataFrame:
    generic_col_names = ["labels", "x"]
    df_col_names = df.columns.values.tolist()
    df = df.rename(columns={df_col_names[i]: generic_col_names[i] for i in range(2)})
    df = DataFrame(concat([Series(row['labels'], row['x'].split(strategy)) for _, row in tqdm(df.iterrows())])).reset_index()#_splitter(row['x'], strategy, 128)
    df_col_names.reverse()
    df = df.rename(columns={k: df_col_names[i] for i,k in enumerate(df.columns.values.tolist())})
    df.to_csv("check.csv")
    return df

@_parser.register
def _add_separate_cols(strategy: bool, df) -> DataFrame:
    df['type'] = df['type'].str.split('')
    df['type'] = df['type'].apply(lambda x: list(map(lambda attr_type: 0 if attr_type in "ESFP" else 1, x[1:-1])))
    return df

def _splitter(string : str, delimiter : str, num_words : int) -> list:
    string.replace(f"{delimiter}", ' ')
    strings = string.split(' ')
    return [' '.join(strings[j-num_words-1:j]) for j in range(num_words-1, len(strings), num_words)]




"""
Retaining domain name: doesn't transform multiple links in a single post
@parser.register
def domain_retain(strategy : list, df) -> DataFrame:
    def transform_url(post):
        url = re.search(r'\bhttp.*[a-zA-Z0-9]\s',post)
        if url:
            regex = re.findall(r'^.*\.(.*)\.', post)
            post = post.replace(url.group(0),regex[0]+" ")

        return post
    df['posts'] = df['posts'].apply(lambda x: transform_url(x))
    return df
"""

