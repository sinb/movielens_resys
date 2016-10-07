# coding=utf-8 

'''
Created on Oct 6, 2016

@author: sinb
'''
import os
import csv
from datetime import datetime
from collections import defaultdict
import heapq
from operator import itemgetter
from math import sqrt
import cPickle as pickle
import logging

def load_reviews(path, **kwargs):
    """
    Loads MovieLens reviews
    """
    options = {
        'fieldnames': ('userid', 'movieid', 'rating', 'timestamp'),
        'delimiter': '\t',
    }
    options.update(kwargs)
    
    parse_date = lambda r,k: datetime.fromtimestamp(float(r[k]))
    parse_int = lambda r,k: int(r[k])
    
    with open(path, 'rb') as reviews:
        reader = csv.DictReader(reviews, **options)
        for row in reader:
            row['userid'] = parse_int(row, 'userid')
            row['movieid'] = parse_int(row, 'movieid')
            row['rating'] = parse_int(row, 'rating')
            row['timestamp'] = parse_date(row, 'timestamp')
            yield row

def relative_path(path):
    """
    Returns a path relative from this code file
    """
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname, path)
    return os.path.normpath(path)

def load_movies(path, **kwargs):
    """
    Loads MovieLens movies
    """
    
    options = {
        'fieldnames': ('movieid', 'title', 'release', 'video', 'url'), 
        'delimiter': '|',
        'restkey': 'genre',
    }
    options.update(kwargs)
    
    parse_int = lambda r,k: int(r[k])
    parse_date = lambda r,k: datetime.strptime(r[k], '%d-%b-%Y') if r[k] else None
    
    with open(path, 'rb') as movies:
        reader = csv.DictReader(movies, **options)
        for row in reader:
            row['movieid'] = parse_int(row, 'movieid')
            row['release'] = parse_date(row, 'release')
            row['video']   = parse_date(row, 'video')
            yield row



class MovieLens(object):
    """
    Data structure to build our recommender model on.
    """
    
    def __init__(self, udata, uitem):
        """
        Instantiate with a path to u.data and u.item
        """
        self.udata   = udata
        self.uitem   = uitem
        self.movies = {}
        self.reviews = defaultdict(dict)
        self.load_dataset()
    
    def load_dataset(self):
        """
        Loads the two datasets into memory, indexed on the ID.
        """
        for movie in load_movies(self.uitem):
            self.movies[movie['movieid']] = movie
        
        for review in load_reviews(self.udata):
            self.reviews[review['userid']][review['movieid']] = review

    def reviews_for_movie(self, movieid):
        """
        Yields the reviews for a given movie
        """
        for review in self.reviews.values():
            if movieid in review:
                yield review[movieid]

    def average_reviews(self):
        """
        Averages the star rating for all movies.
        Yields a tuple of movieid,
        the average rating, and the number of reviews.
        """
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = sum(reviews) / float(len(reviews))
            yield (movieid, average, len(reviews))
    
    def bayesian_average(self, c=59, m=3):
        """
        Reports the Bayesian average with parameters c and m.
        """
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = ((c * m) + sum(reviews)) / float(c + len(reviews))
            yield (movieid, average, len(reviews))
    
    def top_rated(self, n=10):
        """
        Yields the n top rated movies
        """
        return heapq.nlargest(n, self.bayesian_average(), key=itemgetter(1))
    
    def shared_preferences(self, criticA, criticB):
        """
        Returns the intersection of ratings for two critics
        """
        if criticA not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data" % criticA)
        if criticB not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data" % criticB)
        
        moviesA = set(self.reviews[criticA].keys())
        moviesB = set(self.reviews[criticB].keys())
        shared   = moviesA & moviesB # Intersection operator
        
        # Create a reviews dictionary to return
        reviews = {}
        for movieid in shared:
            reviews[movieid] = (
                self.reviews[criticA][movieid]['rating'],
                self.reviews[criticB][movieid]['rating'],
            )
        return reviews

    def euclidean_distance(self, criticA, criticB, prefs='users'):
        """
        Reports the Euclidean distance of two critics, A&B by
        performing a J-dimensional Euclidean calculation of
        each of their preference vectors for the intersection
        of movies the critics have rated.
        if prefs='users', this function return the distance
        between two users by using all shared movies' ratings
        as vector space.
        if prefs='movies', this function return the distance
        between two movies by using all shared users' ratings
        as vector space.
        """
        # Get the intersection of the rated titles in the data.
        if prefs not in ['users', 'movies']:
            raise KeyError("Unknown type of preference, '%s'." % prefs)
        if prefs == 'users':
            preferences = self.shared_preferences(criticA, criticB)
        else:
            preferences = self.shared_critics(criticA, criticB)
        # If they have no rankings in common, return 0.
        if len(preferences) == 0: return 0
        
        # Sum the squares of the differences
        sum_of_squares = sum([pow(a-b, 2) for a, b in preferences.values()])
        
        # Return the inverse of the distance to give a higher score to
        # folks who are more similar (e.g. less distance) add 1 to prevent
        # division by zero errors and normalize ranks in [0, 1]
        return 1 / (1 + sqrt(sum_of_squares))

    def pearson_correlation(self, criticA, criticB, prefs='users'):
        """
        Returns the Pearson Correlation of two critics, A and B by
        performing the PPMC calculation on the scatter plot of (a, b)
        ratings on the shared set of critiqued titles.
        """
        
        # Get the set of mutually rated items
        if prefs not in ['users', 'movies']:
            raise KeyError("Unknown type of preference, '%s'." % prefs)
        if prefs == 'users':
            preferences = self.shared_preferences(criticA, criticB)
        else:
            preferences = self.shared_critics(criticA, criticB)
        # Store the length to save traversals of the len computation.
        # If they have no rankings in common, return 0.
        length = len(preferences)
        if length == 0: return 0
        
        # Loop through the preferences of each critic once and compute the
        # various summations that are required for our final calculation.
        sumA = sumB = sumSquareA = sumSquareB = sumProducts = 0
        for a, b in preferences.values():
            sumA += a
            sumB += b
            sumSquareA += pow(a, 2)
            sumSquareB += pow(b, 2)
            sumProducts += a*b
        
        # Calculate Pearson Score
        numerator   = (sumProducts*length) - (sumA*sumB)
        denominator = sqrt(((sumSquareA*length) - pow(sumA, 2)) * ((sumSquareB*length) - pow(sumB, 2)))
        
        # Prevent division by zero.
        if denominator == 0: return 0
        
        return abs(numerator / denominator)

    def similar_critics(self, user, metric='euclidean', n=None):
        """
        Finds, ranks similar critics for the user according to the
        specified distance metric. Returns the top n similar critics
        if n is specified.
        """
        # Metric jump table
        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation,
        }
        
        distance = metrics.get(metric, None)
        
        # Handle problems that might occur
        if user not in self.reviews:
            raise KeyError("Unknown user, '%s'." % user)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unprogrammed distance metric '%s'." % metric)
        
        # Compute user to critic similarities for all critics
        critics = {}
        for critic in self.reviews:
            # Don't compare against yourself!
            if critic == user:
                continue
            critics[critic] = distance(user, critic)
        
        if n:
            return heapq.nlargest(n, critics.items(), key=itemgetter(1))
        return critics
    
    def predict_ranking(self, user, movie, metric='euclidean', critics=None):
        """
        Predicts the ranking a user might give a movie based on the
        weighted average of the critics similar to the that user.
        """
     
        critics = critics or self.similar_critics(user, metric=metric)
        total   = 0.0
        simsum = 0.0
         
        for critic, similarity in critics.items():
            if movie in self.reviews[critic]:
                total += similarity * self.reviews[critic][movie]['rating']
                simsum += similarity
         
        if simsum == 0.0: return 0.0
        return total / simsum
    
    def predict_all_rankings(self, user, metric='euclidean', n=None, recommend=True):
        """
        Predicts all rankings for all movies, if n is
        specified returns
        the top n movies and their predicted ranking.
        对一个用户,估计他对所有影片的评分, 方法是先找到他与所有用户的相似度,存放在critics里;
        然后对每个电影, 在所有的(相似)用户里, 累计similar_score*rating,最后除以总的total_similar_score,
        也就是考虑所有用户,看这些用户对当前这部影片评分多岁,然后看这些用户和自己的用户相似度是多少,
        求个加权的分数. 
        """
        critics = self.similar_critics(user, metric=metric)
        movies = {
           movie: self.predict_ranking(user, movie, metric, critics) for movie in self.movies
        }
        if recommend:
            for movie in movies.keys():
                if movie in self.reviews[user]:
                    _ = movies.pop(movie)
        
        if n:
            return heapq.nlargest(n, movies.items(), key=itemgetter(1))
        return movies    

    def shared_critics(self, movieA, movieB):
        """
        Returns the intersection of critics for two items, A and B
        """

        if movieA not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" %movieA)
        if movieB not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" %movieB)

        criticsA = set(critic for critic in self.reviews if movieA in self.reviews[critic])
        criticsB = set(critic for critic in self.reviews if movieB in self.reviews[critic])
        shared = criticsA & criticsB # Intersection operator
        # Create the reviews dictionary to return
        reviews = {}
        for critic in shared:
            reviews[critic] = (
                self.reviews[critic][movieA]['rating'],
                self.reviews[critic][movieB]['rating'],
            )
        return reviews

    def similar_items(self, movie, metric='euclidean', n=None):
        # Metric jump table
        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation,
        }

        distance = metrics.get(metric, None)
        # Handle problems that might occur
        if movie not in self.movies:
            raise KeyError("Unknown movie, '%s'." % movie)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unprogrammed distance metric '%s'." % metric)

        items = {}
        for item in self.movies:
            if item == movie:
                continue
            items[item] = distance(item, movie, prefs='movies')

        if n:
            return heapq.nlargest(n, items.items(), key=itemgetter(1))
        return items

    def predict_ranking_item_based(self, user, movie, metric='euclidean'):
        """
        if user has already rated this movie, just return it's previous rating.
        :param user:
        :param movie:
        :param metric:
        :return:
        """
        if self.reviews[user].get(movie) is not None:
            return self.reviews[user][movie]['rating']
        # cache
        if self.item_similar_score is not None:
            movies = self.item_similar_score[movie]
        else:
            movies = self.similar_items(movie, metric=metric)
        total = 0.0
        simsum = 0.0

        for relmovie, similarity in movies.items():
            if relmovie in self.reviews[user]:
                total += similarity * self.reviews[user][relmovie]['rating']
                simsum += similarity

        if simsum == 0.0:
            return 0.0
        return total / simsum
    
    def predict_all_rankings_item_based(self, user, metric='pearson', n=10, recommend=True):
        """
        recommend top n best predicted items for an user
        this function need a pre-computed dict, which stores
        every related items and their similarity score
        使用item based CF推荐影片(也就是对一个用户,估计他对所有影片的评分,
        如果他本来就评价过某个影片,那么就直接使用他的评分(但是推荐的时候要把已经评分过的去掉))
        对所有的影片, 如果这个user以前看过它,那么累计其rating*similar_score,最后加权.
        如果是recommend=True,那么推荐列表中不应该出现用户已经看过的电影
        """
        movies = {
           movie: self.predict_ranking_item_based(user, movie, metric) for movie in self.movies
        }
        if recommend:
            for movie in movies.keys():
                if movie in self.reviews[user]:
                    _ = movies.pop(movie)
        if n:
            return heapq.nlargest(n, movies.items(), key=itemgetter(1))
        return movies    
    
    def get_similar_score_dict(self, metric='pearson', cache=True):
        """
        compute every pair's similar score based on similar_items
        """
        self.item_similar_score = {}
        total_count = len(self.movies)
        count = 0
        perc = 0.05
        for movie_id in self.movies.keys():
            count += 1
            if count / float(total_count) > perc:
                print("%.2f in progress" % perc)
                perc += 0.05
            self.item_similar_score[movie_id] = self.similar_items(movie=movie_id, metric=metric)
        print("compute item similarity score finished")
        if cache:
            with open("similar_score.pkl", "wb") as f:
                pickle.dump(self.item_similar_score, f)
            print "cached!"
    
    def load_similar_score_dict(self, filename):
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            self.item_similar_score = tmp 
        print("load similar score from %s" % filename)
            
            
            