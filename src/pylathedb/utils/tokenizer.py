import re
from string import punctuation,ascii_uppercase,ascii_lowercase
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Tokenizer:
    stop_words = set(stopwords.words('english'))-{'will'}
    re_spaces = re.compile(r'\s+')
    translate_table = str.maketrans(ascii_uppercase+punctuation,ascii_lowercase+' '*len(punctuation))
    del translate_table[ord('_')]
    del translate_table[ord('.')]

    def __init__(self, **kwargs):
        self.tokenize_method = kwargs.get('tokenize_method','simple')

    def compound_keywords(self,keyword_query):
        return [self.tokenize(segment) for segment in keyword_query.split('"')[1::2]]

    def keywords(self,keyword_query):
        return self.tokenize(keyword_query)

    def tokenize(self,text):
        if self.tokenize_method == 'simple':
            return [keyword
                    for token in re.split(self.re_spaces, text.translate(self.translate_table))
                    if ( keyword := token.strip('._') ) not in Tokenizer.stop_words
                    and len(keyword)>1
            ]

        if self.tokenize_method == 'nltk':
            word_tokens = word_tokenize(text)
            keywords = [w for w in word_tokens if not w in self.stop_words]
            return keywords