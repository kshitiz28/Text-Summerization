import heapq
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
punctuation = punctuation + '\n' + "‘" + "’" + "\n\n \n"


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def header_generation(article):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(article.lower())
    filtered_tokens = [lemmatizer.lemmatize(
        token) for token in tokens if token not in stop_words and token not in punctuation]
    word_frequencies = {}
    for word in filtered_tokens:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    greatest_two = [item[0] for item in heapq.nlargest(
        2, word_frequencies.items(), key=lambda x: x[1])]
    header = [' '.join(greatest_two)]
    return {"heading" : header[0]}
