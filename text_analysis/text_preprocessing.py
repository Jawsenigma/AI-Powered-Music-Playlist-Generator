import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)
