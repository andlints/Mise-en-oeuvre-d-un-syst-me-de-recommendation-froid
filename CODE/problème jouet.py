#!/usr/bin/env python
# coding: utf-8


import matplotlib.pylab as pl
import ot
import ot.plot
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from numpy.linalg import norm
from scipy.stats import norm
from scipy.stats import uniform


movies = pd.read_csv("../DATASETS/ml-20m/movies.csv")
ratings = pd.read_csv("../DATASETS/ml-20m/ratings.csv")
genome_scores = pd.read_csv("../DATASETS/ml-20m/genome-scores.csv")

## merge the data
movies_ratings_data = movies.merge(ratings, on = 'movieId', how='inner')

genome_scores.head(2)

movie1_geno = genome_scores[genome_scores.movieId == 1].relevance.values
movie2_geno = genome_scores[genome_scores.movieId == 2].relevance.values

movie1_geno


# ## Exemple sujet

## matrix loss
a = np.array([0.8, 0.2])
b = np.array([0.4, 0.5, 0.1])

M = np.array([[0.15, 0.1, 0.9],
             [0.8, 0.95, 0.005]])

## Transport plan
T = ot.emd(a,b, M)
T

## Wasserstein distance
W = round(ot.emd2(a,b,M), 2)
W 


# ## Exemple datasets

# ### Exemple 1
movies_ratings_data.head(3)

user3 = (movies_ratings_data[movies_ratings_data.userId == 9]).reset_index(drop=True)
user1 = (movies_ratings_data[movies_ratings_data.userId == 200]).reset_index(drop=True)

user1 = user1.head(3)
user3 = user3.head(2)

# normalisation des scores
def dist_norm(a):
    l = len(a)
    s = a.values
    s = s / sum(s)
    #s = uniform.cdf(s, loc = 1, scale = max(s)) / 5 * 2
    return s
dist_norm(user1.rating)

user1_rating = dist_norm(user1.rating)
user3_rating = dist_norm(user3.rating) 
user1_rating

## Matrix loss

#""""""""""""""""""""""""""#
#     Matrice de coût      #
#""""""""""""""""""""""""""#
#   bewteen users 1 and 3  #
#       preferences        #

def Matrix_loss(user1,user3, genome_scores):
    n = len(user1)
    s = len(user3)

    M = np.zeros((n,s))

    for i in range(n):
        for j in range(s):

            # pour comparer la distance entre deux films, on prend les tags genomes 
            # des deux, ici ce sont vi et vj
            # ce qui donne la matrice de coût M
            # où Mij = 1 - sim(vi,vj) (cosinue similtary)

            # user1
            us1_mvId = user1.iloc[i][0] # pour récuperer le movieId
            v1 = genome_scores[genome_scores.movieId == us1_mvId].relevance.values
            #print(us1_mvId)

            # user2
            us3_mvId = user3.iloc[j][0] # pour récuprer le movieId
            v3 = genome_scores[genome_scores.movieId == us3_mvId].relevance.values
            #print(us3_mvId)
            
            M[i, j] = round(cosine(v1, v3), 2)
            #M[i, j] = 1 - np.dot(v1, v3)/(norm(v1)*norm(v3))
            

    M_ = pd.DataFrame(data = M, index = list(user1.title.values), columns=list(user3.title.values))
    return M, M_

M, M_ = Matrix_loss(user3, user1, genome_scores)
## on remarque que user1 (film3) et user3 (film3) ont le même genre
## et que la distance est 0, qui est minimum
M

## Transport plan
T = ot.emd(user3_rating, user1_rating, M)
T

## Wasserstein distance
W = round(ot.emd2(user3_rating,user1_rating,M), 2)
W 


# ### Exemple 2

## for 3 users

ex2_user1 = (movies_ratings_data[movies_ratings_data.userId == 3]).reset_index(drop=True)
ex2_user2 = (movies_ratings_data[movies_ratings_data.userId == 8]).reset_index(drop=True)
ex2_user3 = (movies_ratings_data[movies_ratings_data.userId == 6]).reset_index(drop=True)


ex2_user1 = ex2_user1.head(3)
ex2_user2 = ex2_user2.head(3)
ex2_user3 = ex2_user3.head(3)


def Wasserstein_dist(a, b, genome):
    
    ## normalization of ratings values [0 , ... , 5]
    ## 5 max score
    a_norm = dist_norm(a.rating)
    b_norm = dist_norm(b.rating)
    print(a_norm)
    print(b_norm)
    
    ## Loss Matrix
    M, M_ = Matrix_loss(a, b, genome)
    print('\nla matrice de cout est : \n', M)
    
    ## Transport plan
    T = ot.emd(a_norm, b_norm, M)
    T_ = pd.DataFrame(data = T, index = list(a.title.values), columns=list(b.title.values))
    print('\nle plan de transport est : \n', T)
    
    ## Wasserstein distance
    W = round(ot.emd2(a_norm, b_norm, M), 2)
    print('\nla distance de wasserstein est : ', W)
    
    return M_, T_, W

M_, T_, W_ = Wasserstein_dist(ex2_user1, ex2_user2, genome_scores)

M_

## on remarque qu'on a 0 pour [toy story, Toy story] pour la matrice
## de coût car les deux films sont les mêmes donc il n'y a pas de 
## différence
## la matrice de coût se calcule avec les relevances de chaque films
## ie (tag genome)

T_

## pour le transport plan
## on a 0 entre deux films quand les genres sont très différents
## Grumpier old men (1995) : Comedy|Romance 
## Powder (1995) : Drama|Sci-Fi
## -> sans ressemblance


# #### comparaison entre les 3 utilisateurs

M12, T12, W12 = Wasserstein_dist(ex2_user1, ex2_user2, genome_scores)

M13, T13, T13 = Wasserstein_dist(ex2_user1, ex2_user3, genome_scores)

M23, T23, W23 = Wasserstein_dist(ex2_user2, ex2_user3, genome_scores)

ex2_user1

ex2_user2

ex2_user3

## user's 2 et user's 3 preferences ont les plus proches d = 0.14
## apres
## user's 1 et user's 2 preferences ont les plus proches d = 0.22
## apres
## user's 1 et user's 3 preferences ont les plus proches d = 0.25

