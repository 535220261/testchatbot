import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Tokenisasi
def tokenization(text):
    return word_tokenize(text)

# 2. Stemming dan Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

def lemmatization(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# 3. Part-of-Speech Tagging
def pos_tagging(tokens):
    return nltk.pos_tag(tokens)

# 4. Klasifikasi (dummy classifier for demonstration)
def classify(tokens):
    if 'pizza' in tokens:
        return 'pesan makanan'
    return 'lainnya'

# 5. Regresi
def predict_price(size):
    # Model sederhana untuk prediksi harga
    X = np.array([[20], [25], [30], [35], [40]])
    y = np.array([50, 60, 70, 80, 90])
    model = LinearRegression().fit(X, y)
    return model.predict(np.array([[size]]))[0]

# Data uji untuk setiap langkah
tokenization_test_data = [
    ("Saya ingin memesan pizza besar.", ['Saya', 'ingin', 'memesan', 'pizza', 'besar']),
    ("Halo, bagaimana kabarmu?", ['Halo', ',', 'bagaimana', 'kabarmu', '?'])
]

stemming_test_data = [
    (['memesan', 'pizza', 'besar'], ['pesan', 'pizza', 'besar']),
    (['berjalan', 'menuju', 'sekolah'], ['jalan', 'menuju', 'sekolah'])
]

lemmatization_test_data = [
    (['mice', 'better'], ['mouse', 'good']),
    (['running', 'quickly'], ['run', 'quickly'])
]

pos_tagging_test_data = [
    (['Saya', 'ingin', 'pesan', 'pizza', 'besar'], [('Saya', 'PRP'), ('ingin', 'VB'), ('pesan', 'VB'), ('pizza', 'NN'), ('besar', 'JJ')])
]

classification_test_data = [
    (['pizza', 'besar'], 'pesan makanan'),
    (['apa', 'kabar'], 'lainnya')
]

regression_test_data = [
    (30, 70),
    (35, 80)
]

# Menghitung akurasi untuk tokenisasi
tokenization_accuracy = sum([tokenization(text) == expected for text, expected in tokenization_test_data]) / len(tokenization_test_data)

# Menghitung akurasi untuk stemming
stemming_accuracy = sum([stemming(tokens) == expected for tokens, expected in stemming_test_data]) / len(stemming_test_data)

# Menghitung akurasi untuk lemmatization
lemmatization_accuracy = sum([lemmatization(tokens) == expected for tokens, expected in lemmatization_test_data]) / len(lemmatization_test_data)

# Menghitung akurasi untuk POS tagging
pos_tagging_accuracy = sum([pos_tagging(tokens) == expected for tokens, expected in pos_tagging_test_data]) / len(pos_tagging_test_data)

# Menghitung akurasi untuk klasifikasi
classification_accuracy = sum([classify(tokens) == expected for tokens, expected in classification_test_data]) / len(classification_test_data)

# Menghitung akurasi untuk regresi menggunakan Mean Squared Error (MSE)
mse = mean_squared_error([expected for size, expected in regression_test_data], [predict_price(size) for size, expected in regression_test_data])

# Output hasil akurasi
print(f"Tokenization Accuracy: {tokenization_accuracy:.2f}")
print(f"Stemming Accuracy: {stemming_accuracy:.2f}")
print(f"Lemmatization Accuracy: {lemmatization_accuracy:.2f}")
print(f"POS Tagging Accuracy: {pos_tagging_accuracy:.2f}")
print(f"Classification Accuracy: {classification_accuracy:.2f}")
print(f"Regression MSE: {mse:.2f}")
