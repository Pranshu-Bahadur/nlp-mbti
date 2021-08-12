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
                {"tokenizer": AutoTokenizer.from_pretrained("vinai/bertweet-base"), "max_length" : 40, "truncation" : True}]
        generate_dataset(PATH, args)
"""

class TestAgent(unittest.TestCase):
    def test_run_agent(self):
        ds_config = [[2, "|||", True], 
            {"tokenizer": AutoTokenizer.from_pretrained("vinai/bertweet-base", normalize=True),
            "max_length" : 40,
            "truncation" : True,
            "padding" : "max_length"}, set()]

        agent_config = {"optimizer" : tf.keras.optimizers.Adam(learning_rate=176e-4),
                        "loss" : tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='train_loss'),
                        "metrics" : tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')}
        train_args = {"do_train" : True,
                      "per_device_train_batch_size" : 256,
                      "per_device_eval_batch_size" : 256,
                      "learning_rate" : 176e-06,
                      "weight_decay" : 1e-05,
                      "num_train_epochs" : 15,
                      "logging_strategy" : "epoch",
                      "seed" : 420,
                      "output_dir" : "./ops",
                      "do_eval" : True,
                      "dataloader_num_workers" : 4,
                      "evaluation_strategy" : "steps",
                      "logging_dir" : "./logs",
                      "logging_strategy" : "steps",
                      "logging_steps" : 100
                      }
        train_args = TrainingArguments(**train_args)
        #with train_args.strategy.scope(): 
        agent = init_agent("vinai/bertweet-base", PATH, 4, dataset_config=ds_config)
        run("train", agent, train_args=train_args)







if __name__ == '__main__':
  unittest.main()
