from bs4 import BeautifulSoup
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import graphlab


class Jokes(object):

	def __init__(self):
		pass
	
	def fit(self):
		filename = "data/jokes.dat"
		open_data_file = open(filename, "r")
		soup = BeautifulSoup(open_data_file.read(), 'html.parser')

		self.jokes = soup.text.split(':')
		self.jokes.pop(0)

		self.vectorizer = TfidfVectorizer(self.jokes, stop_words = stopwords.words('english'))

		self.tfidfs = self.vectorizer.fit_transform(self.jokes)
		self.feature_matrix = self.tfidfs

		self.vocabulary = self.vectorizer.vocabulary_

		self.jokes_with_ids = []

	def dim_reduced_cosine_similar(self, joke_number):
		'''
		Can only be called after self.fit() has been called.
		INPUTS: 
			joke_index: the index of the joke we'd like similar jokes to.  Corresponds to joke number, not index of joke
		OUTPUTS: 
			order_of_jokes: len(order_of_jokes) = len(jokes) and is a list of indices in order of similarity to jokes[joke_index]
		'''
		joke_number -= 1
		cosine_sims = cosine_similarity(self.tfidfs)
		import pdb
		pdb.set_trace()

	def reduce_dims(self, num_dims):
		'''
		INPUTS: 
			num_dims: the number of dimensions to be reduced to. 
		'''
		pass

	def item_item_similarity(self, joke_number, train_ratings, score_ratings):
		sf_train_ratings = graphlab.SFrame(train_ratings)
		sf_score_ratings = graphlab.SFrame(score_ratings)
		new_basic_ratings = score_ratings.copy()
		recommender = graphlab.recommender.item_similarity_recommender.create(user_id = 'user_id', 
										                                     item_id = 'joke_id', 
										                                     target = 'rating',
										                                     similarity_type = 'cosine')
		recommender.fit()
		return recommender.predict(sf_score_ratings)

if __name__ == '__main__':
	what_a_joke = Jokes()
	what_a_joke.fit()
	#what_a_joke.item_item_similarity(1)

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