import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

class HezarfenAI:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.y_pred = None
        self.y_test = None
        self.y = None
        self.x = None
        self.stemmer = None
        self.stop_words = None
        self.df = None
        self.model = LogisticRegression()

    def download_dependencies(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def load_dataset(self):
        self.df = pd.read_csv("../hezarfen_ai/datasets/fake_or_real_news.csv")
        self.df.head()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]

        return ' '.join(tokens)

    def train_model(self):
        self.df['processed_text'] = self.df['text'].apply(self.preprocess_text)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.x = self.tfidf_vectorizer.fit_transform(self.df["processed_text"]).toarray()
        self.y = self.df["label"].values
        x_train, x_test, y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.model.fit(x_train, y_train)
        self.y_pred = self.model.predict(x_test)

        print("Guest: ", self.y_pred)

    def evaluate_model(self):
        # Doğruluk değerlendirmesi
        accuracy = accuracy_score(self.y_test, self.y_pred)

        print(f'Model Doğruluğu: {accuracy * 100:.2f}%')

    def save_model(self):
        joblib.dump((self.model, self.tfidf_vectorizer), '../web-api/hezarfen.pkl')

        return joblib.load('../web-api/hezarfen.pkl')

    def ask(self, text):
        processed_text = self.preprocess_text(text)
        vectorized_text = self.tfidf_vectorizer.transform([processed_text]).toarray()
        prediction = self.model.predict(vectorized_text)

        return prediction

    def run_random_test(self):
        import random
        num = random.randint(0, 100)
        test_text = self.df["text"][num]
        test_label = self.df["label"][num]
        result = self.ask(test_text)

        return {
            "text": test_text,
            "label": test_label,
            "prediction": result
        }

    #print(hezarfen.run_random_test())

class ModelLoader:
    def __init__(self):
        self.model = None
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.model, self.tfidf_vectorizer = joblib.load("../web-api/hezarfen.pkl")

    def run_model(self):
        hezarfen = HezarfenAI()
        hezarfen.download_dependencies()
        hezarfen.load_dataset()
        hezarfen.train_model()
        hezarfen.evaluate_model()
        hezarfen.save_model()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]

        return ' '.join(tokens)

    def ask(self, text):
        processed_text = self.preprocess_text(text)
        vectorized_text = self.tfidf_vectorizer.transform([processed_text]).toarray()
        prediction = self.model.predict(vectorized_text)

        return prediction