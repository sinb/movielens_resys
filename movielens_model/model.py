import os
import csv
from datetime import datetime
from collections import defaultdict
import heapq
from operator import itemgetter
from math import sqrt


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

    def euclidean_distance(self, criticA, criticB):
        """
        Reports the Euclidean distance of two critics, A&B by
        performing a J-dimensional Euclidean calculation of
        each of their preference vectors for the intersection
        of movies the critics have rated.
        """
        # Get the intersection of the rated titles in the data.
        preferences = self.shared_preferences(criticA, criticB)
    
        # If they have no rankings in common, return 0.
        if len(preferences) == 0: return 0
        
        # Sum the squares of the differences
        sum_of_squares = sum([pow(a-b, 2) for a, b in preferences.values()])
        
        # Return the inverse of the distance to give a higher score to
        # folks who are more similar (e.g. less distance) add 1 to prevent
        # division by zero errors and normalize ranks in [0, 1]
        return 1 / (1 + sqrt(sum_of_squares))

    def pearson_correlation(self, criticA, criticB):
        """
        Returns the Pearson Correlation of two critics, A and B by
        performing the PPMC calculation on the scatter plot of (a, b)
        ratings on the shared set of critiqued titles.
        """
        
        # Get the set of mutually rated items
        preferences = self.shared_preferences(criticA, criticB)
        
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
    
    def predict_all_rankings(self, user, metric='euclidean', n=None):
        """
        Predicts all rankings for all movies, if n is
        specified returns
        the top n movies and their predicted ranking.
        """
        critics = self.similar_critics(user, metric=metric)
        movies = {
           movie: self.predict_ranking(user, movie, metric, critics) for movie in self.movies
        }
        
        if n:
            return heapq.nlargest(n, movies.items(), key=itemgetter(1))
        return movies    
