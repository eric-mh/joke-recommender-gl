import graphlab
import pandas as pd
import numpy as np
from matFac import *
train_ratings = pd.read_csv('data/ratings.dat', sep="\t")
score_ratings = pd.read_csv('data/dont_use.csv')

def baseline_predictions():
    new_ratings_worst =score_ratings.copy()
    new_ratings_best = score_ratings.copy()
    new_ratings_base = score_ratings.copy()

    # Create worst possible ratings                                                   
    new_ratings_worst['rating'] = -10
    new_ratings_worst.to_csv('predictions/worst_pred.csv')

    # Create best possible ratings 
    new_ratings_best.to_csv('predictions/best_pred.csv')

    # Create baseline ratings with predicted mean
    baseline = np.random.uniform(-10,10,score_ratings.shape[0])
    new_ratings_base['rating'] = baseline
    new_ratings_base.to_csv('predictions/base_pred.csv')

def factorize_only(factors=4):
    sf_train_ratings = graphlab.SFrame(train_ratings)
    sf_score_ratings = graphlab.SFrame(score_ratings)
    new_basic_ratings = score_ratings.copy()
    recommender = genericRecommender(user_id = 'user_id', 
                                     item_id = 'joke_id', 
                                     target = 'rating',
                                     num_factors = factors)
    recommender.fit(sf_train_ratings)
    new_ratings = recommender.predict(sf_score_ratings)
    new_basic_ratings['rating'] = new_ratings
    new_basic_ratings.to_csv('predictions/fac_pred.csv')
