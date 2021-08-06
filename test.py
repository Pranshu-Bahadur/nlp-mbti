import unittest
from preprocess import _parser, nlp_tc_df_parser, generate_dataset
from pandas import read_csv
from transformers import AutoTokenizer

PATH = 'mbti_1.csv'
class TestDFParser(unittest.TestCase):
    def test__parser_helpers(self):
        path = PATH
        df = read_csv(path)
        
        #hyperlink remover check
        no_links_df = _parser(None, df)
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
        test_df = nlp_tc_df_parser(path, ['bring me back pls.'], None, 2, "|||")
        self.assertFalse(test_df.posts.str.contains(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?').any())
        self.assertTrue(len(df)<len(test_df))
        self.assertTrue(len(nlp_tc_df_parser(path))==len(df))


class TestGenDataset(unittest.TestCase):
    def test_gen_dataset(self):
        args = [[['bring me back pls.'], None, 2, "|||",set()],
                {"tokenizer": AutoTokenizer.from_pretrained("vinai/bertweet-base"), "max_length" : 40, "truncation" : True}]
        generate_dataset(PATH, args)







if __name__ == '__main__':
  unittest.main()
