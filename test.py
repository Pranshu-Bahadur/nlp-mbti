import unittest
from preprocess import _parser, nlp_tc_df_parser, generate_dataset
from pandas import read_csv
from transformers import AutoTokenizer, TrainingArguments
from agent import init_agent, run
import tensorflow as tf


PATH = 'mbti_1.csv'

"""
class TestDFParser(unittest.TestCase):
    def test__parser_helpers(self):
        path = PATH
        df = read_csv(path)
        #hyperlink remover check
        no_links_df = _parser(set(), df)
        self.assertFalse(no_links_df.posts.str.contains(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?').any())
        #min word limit check
        min_word_limit_df = _parser(18, no_links_df)
        self.assertTrue(len(df)>len(min_word_limit_df))
        #.posts explosion check
        e_df = _parser("|||", df)
        self.assertTrue(len(df)<len(e_df))
        #label splitting check
        ind_df = _parser(True, df)
        #self.assertTrue(len(e_df.columns < len(ind_df.columns)))
        print(ind_df.head())
        # TODO Domain check
        #d_df = _parser([], e_df)
        #self.assertTrue(len(d_df) == len(e_df))
    def test_nlp_tc_df_parser(self):
        path = PATH
        df = read_csv(path)
        test_df = nlp_tc_df_parser(path, set(), "|||", 2, True)
        self.assertFalse(test_df.posts.str.contains(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?').any())
        self.assertTrue(len(df)<len(test_df))
        self.assertTrue(len(nlp_tc_df_parser(path))==len(df))
 class TestGenDataset(unittest.TestCase):
    def test_gen_dataset(self):
        args = [[['', None, 2, "|||", True],
                {"tokenizer": AutoTokenizer.from_pretrained("distilbert-base-uncased"), "max_length" : 40, "truncation" : True}]
        generate_dataset(PATH, args)
"""

class TestAgent(unittest.TestCase):
    def test_run_agent(self):
        ds_config = [[set(), 2, "|||", True],
            {"tokenizer": AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True),
            "max_length" : 40,
            "truncation" : True,
            "padding" : "max_length"}, set()]
        train_args = {"do_train" : True,
                      "per_device_train_batch_size" : 128,
                      "per_device_eval_batch_size" : 128,
                      "learning_rate" : 1e-03,
                      "num_train_epochs" : 5,
                      "logging_strategy" : "epoch",
                      "seed" : 420,
                      "output_dir" : "./ops",
                      "do_eval" : True,
                      "dataloader_num_workers" : 4,
                      "evaluation_strategy" : "steps",
                      "logging_dir" : "./logs",
                      "logging_strategy" : "steps",
                      "logging_steps" : 1000,
                      }
        train_args = TrainingArguments(**train_args)
        #with train_args.strategy.scope():
        agent = init_agent("distilbert-base-uncased", PATH, 4, dataset_config=ds_config, train_split_factor=0.6)
        run("train", agent, args=train_args, multilabel=True)







if __name__ == '__main__':
  unittest.main()