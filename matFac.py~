import graphlab
from graphlab.toolkits.cross_validation import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_sframe(filepath, names = []):
    df = pd.read_table(filepath, names = names)
    return graphlab.SFrame(df)

class genericRecommender(object):
    ''' A Generic wrapper for graphlab's recommender, giving some extra features:
    Including automatically changing the Sframe into a feature matrix.
    Parameters
    ----------
    user : string
         The column in the SFrame that specifies the 'users'

    item : string
         The column in the SFrame that specifies the 'items'
    
    target : string
         The column in the SFrame that the recommender should target
         
    model : string, optional
         If none, use the vanilla collaborative filtering classifier.
         If 'similarity', use item-item similarity
         If 'factorization', use matrix factorization

    solver : string, optional
         Solver argument passed to graphlab's recommender. Default 'als'.
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.rec = graphlab.recommender.factorization_recommender.create
      
        # Fitted Attributes
        self.model = None
        self.cross_score = None
        self.pred_ratings = None
        self.sframe = None
  
    def fit(self, sframe):
        if self.model:
            return self
        self.model = self.rec(sframe, **self.kwargs)
        self.sframe = sframe

        self.item_v = self.model.coefficients[self.kwargs['item_id']]
        self.user_v = self.model.coefficients[self.kwargs['user_id']]
        return self

    def cross_val(self, sframe, folds = 3):
        if self.cross_score:
            return self.cross_score
        data_folds = graphlab.cross_validation.KFold(sframe, folds)
        cross_val = cross_val_score(data_folds, self.rec, self.kwargs).get_results()

        self.model = cross_val['models'][0]
        self.cross_score = cross_val['summary']['training_rmse'].mean()
        self.sframe = sframe

        self.item_v = self.model.coefficients[self.kwargs['item_id']]
        self.user_v = self.model.coefficients[self.kwargs['user_id']]
        return self.cross_score

    def predict(self):
        if self.pred_ratings:
            return self.pred_ratings
        self.pred_ratings = np.array(self.model.predict(sframe))
        return self.pred_ratings

    def plot_violin(self):
        pred_ratings = self.predict()
        rating = np.array(self.sframe['rating'])
        target = self.kwargs['target']
        data = []
        for i in sorted(self.sframe[target].unique()):
            mask = rating == i
            data.append(pred_ratings[mask])

        plt.violinplot(data)
        plt.show()

    def score(self):
        return self.model.training_rmse

    def get_latent(self, title_series, top = 10):
        model = self.model
        title = np.array(title_series)

        item = self.item_v
        items = np.array(item['factors'])
        items_lf = items.T

        sorted_items = np.argsort(items_lf)
        items_coll = []
        for i in sorted_items:
            mask_i = i[::-1][:top]
            items_coll.append(title[mask_i])

        df_latent = pd.DataFrame(items_coll)
        self.df_latent = graphlab.SFrame(df_latent.T)
        return self.df_latent


if __name__ == "__main__":
    sframe = load_sframe('u.data', names = ['user', 'movie', 'rating', 'timestamp'])
    
    rec = genericRecommender(user_id = 'user', item_id = 'movie', target = 'rating',
                             num_factors = 2)
    rec.fit(sframe)

    title_series = df_ec = pd.read_table('u.item', names=range(24),sep='|')[1]
    rec.get_latent(title_series)
