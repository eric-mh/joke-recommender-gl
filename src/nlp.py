from bs4 import BeautifulSoup
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

filename = "data/jokes.dat"
open_data_file = open(filename, "r")
soup = BeautifulSoup(open_data_file.read(), 'html.parser')

jokes = soup.text.split(':')
jokes.pop(0)

vectorizer = TfidfVectorizer(jokes, stop_words = stopwords.words('english'))

tfidfs = vectorizer.fit_transform(jokes)

vocabulary = vectorizer.vocabulary_




'''
tokenizer = RegexpTokenizer(r'\w+')
tokenized_jokes = []

for joke in jokes:
	tokenized_joke = tokenizer.tokenize(joke)
	toremove = lambda x: x not in stopwords.words('english')
	tokenized_joke = list(filter(toremove, tokenized_joke))
	tokenized_joke.pop()
	tokenized_jokes.append(tokenized_joke)
	import pdb
	pdb.set_trace()
'''