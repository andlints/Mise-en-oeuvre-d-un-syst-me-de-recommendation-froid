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



###################
###################
# Data processing #
###################
###################

# 1 -
# ratings >= 4 
# len(ratings) == 20000263
# length after 9995410
s = (ratings[ratings.rating.values >= 4]).reset_index(drop=True)
del s['timestamp']
print(s.isnull().any())

# 2 - 
# drop no tags genomes data
id_genome = genome_scores.movieId.unique().tolist()
notin_genome = set(movies.movieId.unique().tolist()) - set(id_genome)
movies = (movies[~movies.movieId.isin(notin_genome)]).reset_index(drop=True)

# 3 -
# take only 10 users, reduciton de données
all_users = ratings.userId.unique().tolist()
n_users = np.random.choice(all_users, 5)
notin_user = set(s.userId.unique().tolist()) - set(n_users)
s = (s[~s.userId.isin(notin_user)]).reset_index(drop=True)

# merging data
data = movies.merge(s, on = 'movieId', how='inner')

# 4 -
# all unique movies with rating >= 4
Idx_unique_movies = data['movieId'].unique().tolist()
movies = movies.loc[movies['movieId'].isin(Idx_unique_movies)]

# 5 -
# index of all users
Idx_uniques_users = ratings['userId'].unique().tolist()

# 6 -
# exeperimental setup based on items

# train:test = 3:1
rs = ShuffleSplit(n_splits = 1, test_size = 0.25, random_state = len(movies))
rs.get_n_splits(movies)

for train_index, test_index in rs.split(movies):
    print("TRAIN:", train_index, "TEST:", test_index)
    

mv_interact_V = (movies.iloc[train_index][:]).reset_index(drop=True)
mv_coldStrt_C = (movies.iloc[test_index][:]).reset_index(drop=True)

print('longueur training :', len(mv_interact_V), '=',round (len(mv_interact_V) / len(movies) * 100, 0), '%:')
print('longueur test :', len(mv_coldStrt_C), '=', round(len(mv_coldStrt_C) / len(movies) * 100, 0), '%')


###################
###################
##### program #####
###################
###################

# regularisation parameter
reg = 0.05


# Matrix loss M€R^(nxs)_+
def Matrix_loss_all(data1, data2, genome_scores):
    m = len(data1)
    n = len(data2)

    M = np.zeros((m,n))

    for i in range(m):
        for j in range(n):

            # pour comparer la distance entre deux films, on prend les tags genomes 
            # des deux, ici ce sont vi et vj
            # ce qui donne la matrice de coût M
            # où Mij = 1 - sim(vi,vj) (cosinue similtary)

            # movies 1
            mvId1 = data1.iloc[i][0] # pour récuperer le movieId
            #print(mvId1)
            v1 = genome_scores[genome_scores.movieId == mvId1].relevance.values
            #print(us1_mvId)

            # movies 2
            mvId2 = data2.iloc[j][0] # pour récuprer le movieId
            v2 = genome_scores[genome_scores.movieId == mvId2].relevance.values
            #print(mvId2)
            
            M[i, j] = round(cosine(v1, v2), 2)
            #M[i, j] = 1 - np.dot(v1, v3)/(norm(v1)*norm(v3))
            #print(M[i,j])

    #M_ = pd.DataFrame(data = M, index = list(data.title.values), columns=list(data.title.values))
    return M

M = Matrix_loss_all(mv_interact_V, mv_coldStrt_C, genome_scores)

# conjugate variable K
# K := e^(-M/reg)
#K = .......... 


# compute g where D.T * g = 0, g € R^s
# g_u* € argmin H*_pu(g) 

# alpha
# alpha := a^(g/reg)
