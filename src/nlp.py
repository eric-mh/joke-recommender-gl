from bs4 import BeautifulSoup
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import graphlab
from sklearn.decomposition import PCA
from nmf import *


class Jokes(object):

	def __init__(self):
		pass

	def fit(self):
		filename = "../data/jokes.dat"
		open_data_file = open(filename, "r")
		soup = BeautifulSoup(open_data_file.read(), 'html.parser')

		self.jokes = soup.text.split(':')
		self.jokes.pop(0)

		self.vectorizer = TfidfVectorizer(self.jokes, stop_words = stopwords.words('english'))

		self.tfidfs = self.vectorizer.fit_transform(self.jokes)
		self.feature_matrix = self.tfidfs

		self.vocabulary = self.vectorizer.vocabulary_

		self.jokes_with_ids = []

	def get_cosine_sim_matrix(self):
		'''
		Can only be called after self.fit() has been called.
		INPUTS:
			joke_index: the index of the joke we'd like similar jokes to.  Corresponds to joke number, not index of joke
		OUTPUTS:
			order_of_jokes: len(order_of_jokes) = len(jokes) and is a list of indices in order of similarity to jokes[joke_index]
		'''
		design = np.dot(self.W, self.H)
		return cosine_similarity(design)

	def reduce_dims(self, num_dims):
		'''
		INPUTS:
			num_dims: the number of dimensions to be reduced to.
		OUTPUTS:
			feature_matrix: a len(jokes) x num_dims array
		'''
		features = self.tfidfs.toarray()
		features = np.nan_to_num(features)
		self.W, self.H, _ = calc_nmf(features, rank = 20)

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

	def joke_types(self):
		joke_order_per_cat = np.argsort(self.W.T)
		self.important_jokes_per_cat = []
		for order in joke_order_per_cat:
			mask = order[::-1][:10]
			import pdb
			pdb.set_trace()
			self.important_jokes_per_cat.append(self.jokes[mask])

if __name__ == '__main__':
	what_a_joke = Jokes()
	what_a_joke.fit()
	#what_a_joke.item_item_similarity(1)
	what_a_joke.reduce_dims(4)
	cosine_sims = what_a_joke.get_cosine_sim_matrix()
	what_a_joke.joke_types()

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
