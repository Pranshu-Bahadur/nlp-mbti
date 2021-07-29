from pandas import read_csv, DataFrame, concat, Series
from tld import is_tld
from functools import singledispatch
from functools import reduce
import re
def nlp_tc_df_parser(path : str, *args : list):
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
    df = reduce(lambda x, y: parser(y, x), args, read_csv(path)) if len(args) > 0 else read_csv(path)
    return df

#======================================
# Helper Functions for nlp_tc_df_parser:
#======================================

# Remove hyperlinks that end with .com
@singledispatch
def parser(df, strategy) -> DataFrame:
    return df

@parser.register
def hyper_link_cleaner(strategy : None, df)  -> DataFrame:
    df.posts = df.posts.str.replace(r'\bhttp.*[a-zA-Z0-9]\b', " ")
    return df

@parser.register
def remove_below_word_limit(strategy : int, df)  -> DataFrame:
    df["total_words"] = df.posts.str.split(" ").map(len)
    df = df[df["total_words"]>strategy]
    df = df.drop(columns=["total_words"])
    return df

# Splits input rows based on a given delimiter
@parser.register
def explode(strategy : str, df)  -> DataFrame:
    generic_col_names = ["labels", "x"]
    df_col_names = df.columns.values.tolist()
    df = df.rename(columns={df_col_names[i]: generic_col_names[i] for i in range(2)})
    df = DataFrame(concat([Series(row['labels'], row['x'].split(strategy)) for _, row in df.iterrows()])).reset_index()
    df_col_names.reverse()
    df = df.rename(columns={k: df_col_names[i] for i,k in enumerate(df.columns.values)})
    return df

# Retaining domain name: doesn't transform multiple links in a single post
@parser.register
def domain_retain(strategy : list, df) -> DataFrame:
    def transform_url(post):
        url = re.search(r'\bhttp.*[a-zA-Z0-9]\s',post)
        if url:                
            regex = re.findall(r'^.*\.(.*)\.', post)
            post = post.replace(url.group(0),regex[0]+" ")
            print(post)
        return post

    df['posts'] = df['posts'].apply(lambda x: transform_url(x))
    return df