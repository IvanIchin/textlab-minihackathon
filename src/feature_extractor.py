import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Descarga segura de recursos NLTK en servidor
for resource in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type="N-gramas de palabras", ngram_size=1, representation="TF-IDF", language="spanish"):
        self.feature_type = feature_type
        self.ngram_size = ngram_size
        self.representation = representation
        self.language = language
        self.vectorizer_ = None

    def _preprocess_texts(self, X):
        stop_words = set(stopwords.words(self.language)) if self.language in stopwords.fileids() else set()
        processed = []
        for text in X:
            tokens = nltk.word_tokenize(str(text).lower())
            if self.feature_type == "N-gramas de palabras":
                processed.append(" ".join(tokens))
            elif self.feature_type == "Stopwords":
                sw = [t for t in tokens if t in stop_words]
                processed.append(" ".join(sw) if sw else "empty_sw")
            elif self.feature_type == "Content words":
                cw = [t for t in tokens if t.isalpha() and t not in stop_words]
                processed.append(" ".join(cw) if cw else "empty_cw")
            elif self.feature_type == "Etiquetas POS":
                tags = [tag for _, tag in nltk.pos_tag(tokens)]
                processed.append(" ".join(tags) if tags else "empty_pos")
        return processed

    def fit(self, X, y=None):
        processed_X = self._preprocess_texts(X)
        ngram_range = (self.ngram_size, self.ngram_size)
        
        if self.representation == "TF-IDF":
            self.vectorizer_ = TfidfVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
        elif self.representation == "Frecuencia":
            self.vectorizer_ = CountVectorizer(ngram_range=ngram_range, binary=False, token_pattern=r"(?u)\b\w+\b")
        elif self.representation == "Binaria":
            self.vectorizer_ = CountVectorizer(ngram_range=ngram_range, binary=True, token_pattern=r"(?u)\b\w+\b")
            
        self.vectorizer_.fit(processed_X)
        return self

    def transform(self, X):
        processed_X = self._preprocess_texts(X)
        return self.vectorizer_.transform(processed_X)