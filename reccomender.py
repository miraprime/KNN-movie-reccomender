import pandas

movies_dataframe = pandas.read_csv("movies.csv")
movies_dataframe.set_index("movieID", inplace = True)

ratings_dataframe = pandas.read_csv("ratings.csv")

total_counts = ratings_dataframe["movieID"].value_counts()

movies_dataframe["RatingsCount"] = total_counts

average_ratings = ratings_dataframe.groupby("movieID").mean()["rating"]

movies_dataframe["averageRating"] = average_ratings

movies_dataframe.sort_values(["RatingsCount", "averageRating"],
                             ascending = False)

MINIMUM_RATINGS_COUNT = 100

minimum_ratings_subset = movies_dataframe.query(f"RatingsCount >= {MINIMUM_RATINGS_COUNT}")

import numpy

def find_user_ratings(userID):

  user_ratings = ratings_dataframe.query(f"userID =={userID}")

  return user_ratings[["movieID", "rating"]].set_index("movieID")

def find_distance_between_real_users(userID_1, userID_2):
  ratings_user_1 = find_user_ratings(userID_1)

  ratings_user_2 = find_user_ratings(userID_2)

  ratings_comparison = ratings_user_1.join(ratings_user_2, 
                                           lsuffix="_1", 
                                           rsuffix="_2").dropna()

  user1_compared = ratings_comparison["rating_1"]

  user2_compared = ratings_comparison["rating_2"]

  distance_between_users = numpy.linalg.norm(user1_compared - user2_compared)

  return [userID_1, userID_2, distance_between_users]

def find_relative_disances(userID):

  users = ratings_dataframe["userID"].unique()

  users = users[users != userID]

  distances = [find_distance_between_real_users(userID, every_id) for every_id in users]

  return pandas.DataFrame(distances, columns = ["singleUserID", "userID", "distance"])

examples_distances = find_relative_disances(7)

def find_top_similar_users(userID):
  distances_to_user = find_relative_disances(userID)

  distances_to_user = distances_to_user.sort_values("distance")

  distances_to_user = distances_to_user.set_index("userID")

  return distances_to_user

SAMPLE_USER_ID = 9

#print(find_top_similar_users(SAMPLE_USER_ID).head(20))

def make_movie_reccomendation(userID):

  user_ratings = find_user_ratings(userID)

  top_similar_users = find_top_similar_users(userID)

  MOST_SIMILAR = 0

  most_similar_user = top_similar_users.iloc[MOST_SIMILAR]

  most_similar_user_ratings = find_user_ratings(most_similar_user.name)

  unwatched_movies = most_similar_user_ratings.drop(user_ratings.index,
                                                   errors = "ignore")
  unwatched_movies = unwatched_movies.sort_values("rating", ascending = False)

  reccomended_movies = unwatched_movies.join(movies_dataframe)

  return reccomended_movies

#print(make_movie_reccomendation(SAMPLE_USER_ID))

##K NEAREST

NUMBER_OF_NEIGHBORS = 5

def find_k_nearest_neighbors(userID, k=NUMBER_OF_NEIGHBORS):

  distances_to_user = find_relative_disances(userID)

  distances_to_user = distances_to_user.sort_values("distance")

  distances_to_user = distances_to_user.set_index("userID")

  return distances_to_user.head(k)

#print(find_k_nearest_neighbors(SAMPLE_USER_ID))

def make_reccomendation_with_knn(userID):

  top_k_neighbors = find_k_nearest_neighbors(userID)

  ratings_by_index = ratings_dataframe.set_index("userID")

  top_similar_ratings = ratings_by_index.loc[top_k_neighbors.index]

  top_similar_rating_average = top_similar_ratings.groupby("movieID").mean()[["rating"]]

  reccomended_movie = top_similar_rating_average.sort_values("rating", ascending=False)

  return reccomended_movie.join(movies_dataframe)

#print(make_reccomendation_with_knn(SAMPLE_USER_ID))

NUMBER_OF_MOVIES = 14

import random

MINIMUM_NUMBER = 1

ROWS_INDEX = 0

MAXIMUM_NUMBER = movies_dataframe.shape[ROWS_INDEX]

test_user_watched_movies = []


for i in range(0, NUMBER_OF_MOVIES):

  random_movie_index = random.randint(MINIMUM_NUMBER, MAXIMUM_NUMBER)

  test_user_watched_movies.append(random_movie_index)

MINIMUM_RATING = 0

MAXIMUM_RATING = 5

test_user_ratings = []

for index in range(0, NUMBER_OF_MOVIES):

  random_rating = random.randint(MINIMUM_RATING, MAXIMUM_RATING + 1)

  test_user_ratings.append(random_rating)


user_data = [list(index) for index in zip(test_user_watched_movies, test_user_ratings)]

def create_new_user(user_data):

  new_user_id = ratings_dataframe["userID"].max() + 1

  new_user_dataframe = pandas.DataFrame(user_data, columns = ["movieID", "rating"])

  new_user_dataframe["userID"] = new_user_id

  return pandas.concat([ratings_dataframe, new_user_dataframe])

NEW_USER_ID = 612

print(make_reccomendation_with_knn(NEW_USER_ID))