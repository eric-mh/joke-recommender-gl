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
        self.jokes.reduce_dims(200)

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


    def pred_one_user(self, user_id, joke_id=True, timing=False):
        '''
        Accept user id as arg. Return the predictions for a single user.

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''

        start_time = time()

        n_items = self.mat.shape[1]
        items_rated_by_this_user = self.mat[user_id]
        items_rated_by_this_user = items_rated_by_this_user.nonzero()[1]
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
            return np.nan_to_num(output[joke_id]), (time() - start_time)
        else:
            return np.nan_to_num(output[joke_id])


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
sample_ratings = sample_ratings.sort_values(['user_id', 'joke_id'])

ratings_as_mat = sparse.csr_matrix((sample_ratings['rating'], (sample_ratings['user_id'], sample_ratings['joke_id'])))

model = ItemItemRecommender(50)
model.fit(ratings_as_mat)
res = model.pred_all_users()

test_data = pd.read_csv('../data/dont_use.csv')

predictions = []
for _, row in test_data.iterrows():
    user_id, joke_id = int(row['user_id']), int(row['joke_id'])
    pred_y = res[user_id, joke_id]
    predictions.append([user_id, joke_id, pred_y])


#save_csv



'''


model = ItemItemRecommender(75)
model.fit(ratings_mat)

print model.pred_one_user(1, timing=True)
print model.pred_all_users(timing=True)


y_pred, items = model.top_n_recs(1, 10)
'''
