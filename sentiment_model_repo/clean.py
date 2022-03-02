# import packages 
import nltk

# Download dependency
corpora_list = ["stopwords","names","brown","wordnet"]  

for dependency in corpora_list:
    try:
        nltk.data.find('corpora/{}'.format(dependency))
    except LookupError:
        nltk.download(dependency)

taggers_list = ["averaged_perceptron_tagger","universal_tagset"]

for dependency in taggers_list:
    try:
        nltk.data.find('taggers/{}'.format(dependency))
    except LookupError:
        nltk.download(dependency)

tokenizers_list = ["punkt"]

for dependency in tokenizers_list:
    try:
        nltk.data.find('tokenizers/{}'.format(dependency))
    except LookupError:
        nltk.download(dependency)
    
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re #regular expression
from string import punctuation 


stop_words =  stopwords.words('english')
    
# function to clean the text 
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)


