import unittest
from utils import parser, nlp_tc_df_parser
from pandas import read_csv

PATH = 'mbti_1.csv'
class TestDFParser(unittest.TestCase):
    def test_parser_helpers(self):
        path = PATH
        df = read_csv(path)
        
        #hyperlink remover check
        no_links_df = parser(None, df)
        self.assertFalse(no_links_df.posts.str.contains(r'\bhttp.*[a-zA-Z0-9]\b').any())

        #min word limit check
        min_word_limit_df = parser(18, no_links_df)
        self.assertTrue(len(df)>len(min_word_limit_df))

        #explosion check
        e_df = parser("|||", df)
        self.assertTrue(len(df)<len(e_df))

        # Domain check
        d_df = parser([], e_df)
        self.assertTrue(len(d_df) == len(e_df))

    def test_nlp_tc_df_parser(self):
        path = PATH
        df = read_csv(path)
        test_df = nlp_tc_df_parser(path, None, 2, "|||",[])
        self.assertFalse(test_df.posts.str.contains(r'\bhttp*[a-zA-Z0-9]\b').any())
        self.assertTrue(len(df)<len(test_df))
        self.assertTrue(len(nlp_tc_df_parser(path))==len(df))


#class Test







if __name__ == '__main__':
  unittest.main()
