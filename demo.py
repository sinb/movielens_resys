# coding=utf-8 
'''
Created on Oct 6, 2016

@author: sinb
'''
from movielens_model import MovieLens
from sklearn.cluster import KMeans
import numpy as np


UDATA = r'/home/sinb/Programming/python/spark_python/ml-100k/u.data'
UITEM = r'/home/sinb/Programming/python/spark_python/ml-100k/u.item'
SIMILAR_SCORE_CACHE = r'/home/sinb/similar_score.pkl'

def kmeans_movielens(model):
    movie_genre = [map(int, movie_info['genre']) for movie_info in model.movies.values()]
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=0).fit(movie_genre)
    labels = kmeans.labels_
    for tag in range(k):
        print len(np.where(labels==tag)[0])
    print_title_from_cluster(model.movies, labels, tag=1)


def print_title_from_cluster(movie_info, labels, tag):
    labels_tag = np.where(labels==tag)[0]
    for i in labels_tag:
        idx = i + 1
        print movie_info[idx]['title']

def print_top_rated_movie(model, topn):
    for id, rating in topn:
        print model.movies[id]['title'], rating  

def print_user_rated_topn_movie(model, uid, n=10):
    lst = sorted(model.reviews[uid].values(), key=lambda x: x['rating'], reverse=True)
    for i in lst[:n]:
        print model.movies[i['movieid']]['title'], i['rating']  
        
if __name__ == '__main__':
    model = MovieLens(udata=UDATA, uitem=UITEM)
    model.load_similar_score_dict(SIMILAR_SCORE_CACHE)
    
    uid = 1
    topn = 100
    a = model.predict_all_rankings(user=uid, metric='pearson', n=topn)
    print_top_rated_movie(model=model, topn=a)
    print
    print_user_rated_topn_movie(model, uid=uid, n=topn)
    print
    b = model.predict_all_rankings_item_based(user=uid, metric='pearson', n=topn)
    print_top_rated_movie(model=model, topn=b)
    
    