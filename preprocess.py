from pandas import read_csv, DataFrame, concat, Series
from functools import singledispatch, reduce
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from tensorflow import data
from tensorflow.data import Dataset
import numpy as np
import pandas as pd

#======================================================================
# 
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
    encodings = tokenizer(list(df['posts']), **kwargs)
    encodings['labels'] = df['type'].tolist()
    return encodings

@_generate_dataset_helper.register
def _gen_tf_dataset(encodings : dict, kwargs=None) -> Dataset:
    return Dataset.from_tensor_slices(encodings)





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
def _hyper_link_cleaner(strategy : None, df)  -> DataFrame:
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
    df = DataFrame(concat([Series(row['labels'], row['x'].split(strategy)) for _, row in tqdm(df.iterrows())])).reset_index()
    df_col_names.reverse()
    df = df.rename(columns={k: df_col_names[i] for i,k in enumerate(df.columns.values.tolist())})
    df.to_csv("check.csv")
    return df

@_parser.register
def _add_separate_cols(strategy: set, df) -> DataFrame:

#     df['IE'] = df['mbti'] 
#     df['NS'] = df['mbti']
#     df['FT'] = df['mbti']
#     df['PJ'] = df['mbti']

# #Assigning individual labels 
#     for i, label in enumerate(df.type):
#         if 'I' in label:
#             df.IE[i] = 'I'
#         elif 'E' in label:
#             df.IE[i] = 'E'
            
#         if 'N' in label:
#             df.NS[i] = 'N'
#         elif 'S' in label:
#             df.NS[i] = 'S'
            
#         if 'F' in label:
#             df.FT[i] = 'F'
#         elif 'T' in label:
#             df.FT[i] = 'T'

#         if 'P' in label:
#             df.PJ[i] = 'P'
#         elif 'J' in label:
#             df.PJ[i] = 'J'
#     df.to_csv("test.csv")

    if 'type' not in df.columns:
        return df   
    #storing the indexes manually and assigning labels since the string labels have been overwritten 
    ie_list = [0, 2, 3, 6, 8, 9, 10, 11]
    ns_list = [0,1,2,3,4,5,6,7]
    ft_list = [0, 5, 6, 8, 7, 10, 13, 15]
    pj_list = [1 ,2, 6, 7, 8, 9, 12, 13]

    df['IE'] = df.apply(lambda row: 'I' if row.type in ie_list else 'E', axis=1)
    df['NS'] = df.apply(lambda row: 'N' if row.type in ns_list else 'S', axis=1)
    df['FT'] = df.apply(lambda row: 'F' if row.type in ft_list else 'T', axis=1)
    df['PJ'] = df.apply(lambda row: 'P' if row.type in pj_list else 'J', axis=1)

    print(df.head()) 
    return df

  





"""
# Retaining domain name: doesn't transform multiple links in a single post
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
