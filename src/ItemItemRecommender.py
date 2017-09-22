import numpy as np
import pandas as pd
from scipy import sparse
from time import time
from nlp import Jokes

class ItemItemRecommender(object):


    def __init__(self, neighborhood_size):
        '''
        Initialize the parameters of the model.
        '''
        self.jokes = Jokes()
        self.jokes.fit()

        self.neighborhood_size = neighborhood_size
        self.mat = None

    def fit(self, mat):
        '''
        Implement the model and fit it to the data passed as an argument.

        Store objects for describing model fit as class attributes.
        '''

        self.mat = mat
        self._set_neighborhoods()


    def _set_neighborhoods(self):
        '''
        Get the items most similar to each other item.

        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.

        You will call this in your fit method.
        '''

        #self.items_cos_sim = cosine_similarity(self.mat.T)
        self.items_sim = self.jokes.get_cosine_sim_matrix()
        least_to_most_sim_indexes = np.argsort(self.items_sim, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]


    def pred_one_user(self, user_id, timing=False):
        '''
        Accept user id as arg. Return the predictions for a single user.

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''

        start_time = time()

        n_items = self.mat.shape[1]
        items_rated_by_this_user = self.mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        output = np.zeros(n_items)
        for item_to_rate in range(n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)
                                        # assume_unique speeds up intersection op
            output[item_to_rate] = self.mat[user_id, relevant_items] * \
                self.items_sim[item_to_rate, relevant_items] / \
                self.items_sim[item_to_rate, relevant_items].sum()

        if timing:
            return np.nan_to_num(output), (time() - start_time)
        else:
            return np.nan_to_num(output)


    def pred_all_users(self, timing=False):
        '''
        Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''

        start_time = time()

        res = []
        for user_id, ratings in enumerate(self.mat):
            res.append(self.pred_one_user(user_id))

        if timing:
            return res, (time() - start_time)
        else:
            return res


    def top_n_recs(self, user_id, number):
        '''
        Take user_id argument and number argument.

        Return that number of items with the highest predicted ratings, after
        removing items that user has already rated.
        '''
        res = []
        y_pred = self.pred_one_user(user_id)
        items_rated_by_this_user = self.mat[user_id]

        # Exclude items already rated
        #not_rated = y_pred[y_pred != items_rated_by_this_user]

        # Use np.argsort and number
        #top_items = np.argsort(not_rated)

        # return a list
        return y_pred, items_rated_by_this_user



# if __name__ == "__main__":


sample_ratings = pd.read_csv('../data/ratings.dat', sep="\t")

highest_user_id = sample_ratings.user_id.max()
highest_joke_id = sample_ratings.joke_id.max()

ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_joke_id))
for _, row in sample_ratings.iterrows():
    # subtract 1 from id's due to match 0 indexing
    ratings_as_mat[row.user_id - 1, row.joke_id - 1] = row.rating

model = ItemItemRecommender(50)
model.fit(sample_ratings)

model.pred_one_user(34888)



'''


model = ItemItemRecommender(75)
model.fit(ratings_mat)

print model.pred_one_user(1, timing=True)
print model.pred_all_users(timing=True)


y_pred, items = model.top_n_recs(1, 10)
'''
