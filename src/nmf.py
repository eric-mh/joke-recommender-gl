import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_nmf_k(V, maxrank, iter = 10, plot = False):
    ''' Calculate the nmf over a range of ranks '''
    ranks = range(1,maxrank + 1)
    f_dict = {}
    scores = []
    for rank in ranks:
        W, H, score = calc_nmf(V.copy(), rank, iter, False)
        f_dict[rank] = (W, H, score)
        scores.append(score)

    if plot:
        plt.plot(ranks, scores)
        plt.show()
    return f_dict

def calc_nmf(V, rank, iter = 10, plot = False):
    ''' Calculate the  nmf of a feature matrix
    Parameters
    ----------
    V : array
        Feature Matrix
    rank : int
        Rank to use for the nmf approximation
    iter : int, optional
        Number of iterations
    plot : boolean, optional
        Plot the RMSE over the iterations of the optimization
    
    Returns
    -------
    W : array
    H : array
    score : float
        The score of the final 
    '''
    W = np.random.rand(V.shape[0], rank)

    iterations = range(iter)
    RMSE = []
    for i in iterations:
        lstsq_r = np.linalg.lstsq(W, V)
        H = lstsq_r[0]

        H[H < 0] = 0
        RMSE.append(score(V, W, H))

        W = np.linalg.lstsq(H.T, V.T)[0].T
        W[W < 0] = 0

    fscore = score(V, W, H)
    if plot:
        plt.title('Final Score for rank {}: {}'.format(rank, fscore))
        plt.plot(iterations, RMSE)
        plt.show()

    return W, H, fscore

def score(V, W, H):
    ''' Score a nmf approximation V = WH
    Parameters
    ----------
    V : array
    W : array
    H : array '''
    return np.linalg.norm(V - np.dot(W,H))

### TESTING THINGS
movies = ['Matrix','Alien','StarWars','Casablanca','Titanic']
users = ['Alice','Bob','Cindy','Dan','Emily','Frank','Greg']
M = pd.DataFrame([[1, 2, 2, 0, 0],
                  [3, 5, 5, 0, 0],
                  [4, 4, 4, 0, 0],
                  [5, 5, 5, 0, 0],
                  [0, 2, 0, 4, 4],
                  [0, 0, 0, 5, 5],
                  [0, 1, 0, 2, 2]],
                 index=users, columns=movies)
M_values = M.values
