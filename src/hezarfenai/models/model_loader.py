from TurkishStemmer import TurkishStemmer
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from hezarfenai.models.news_checker import HezarfenAI
from sklearn.feature_extraction.text import TfidfVectorizer

defined_models = {
    "HezarfenAI": HezarfenAI,
    "HezarfenSearch": ""
}

class ModelLoader:
    def __init__(self, model_path, dataset_file, model_type: str):
        self.stop_words = set(stopwords.words("turkish"))
        self.stemmer = TurkishStemmer()
        self.model_path = model_path
        self.dataset_file = dataset_file
        self.model_type = model_type
        self.tfidf_vectorizer = TfidfVectorizer()

        hezarfen = HezarfenAI(model_path=self.model_path, dataset_path=self.dataset_file)
        hezarfen.download_dependencies()
        hezarfen.load_dataset()
        hezarfen.train_model()
        hezarfen.evaluate_model()
        hezarfen.save_model()

        self.model = joblib.load(model_path)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        print(tokens)

        return ' '.join(tokens)

    def ask(self, text):
        processed_text = self.preprocess_text(text)
        vectorized_text = self.tfidf_vectorizer.transform([processed_text]).toarray() # type: ignore
        prediction = self.model.predict_proba(vectorized_text)

        return prediction[0][1]
