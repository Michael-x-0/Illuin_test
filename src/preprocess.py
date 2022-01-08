import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)

def preprocess(text):
    text = text.lower().replace('é','e').replace('è','e').replace('ê','e').replace('ï','i').replace('ü','u')
    
    #On retire les pontuation
    text_p = "".join([char for char in text if char not in string.punctuation+'ːˈ'])
    
    words = word_tokenize(text_p)
    
    #on retire les stopwords
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    
    #Stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words]

    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmetized = [lemmatizer.lemmatize(word) for word in stemmed]
    return " ".join(lemmetized)