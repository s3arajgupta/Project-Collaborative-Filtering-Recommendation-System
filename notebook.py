#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering Recommendation System

# ## Task 1: Import Modules

# In[1]:


import pandas as pd 
import numpy as np 
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


# ## Task 2: Import the Dataset

# In[2]:


#Load the rating data into a DataFrame:
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv('Movie_data.csv', sep = ',', names = column_names)

#Load the move information in a DataFrame:
movies_title_df = pd.read_csv("Movie_Id_Titles.csv")
movies_title_df.rename(columns = {'item_id':'Movie_ID', 'title':'Movie_Title'}, inplace = True)

#Merge the DataFrames:
movies_df = pd.merge(movies_df,movies_title_df, on='Movie_ID')

#View the DataFrame:
print(movies_df.head())


# ## Task 3: Explore the Dataset

# In[3]:


print(f"\n Size of the movie_df dataset is {movies_df.shape}")


# In[4]:


movies_df.describe()


# In[5]:


movies_df.groupby('User_ID')['Rating'].count().sort_values(ascending = True).head()


# In[6]:


n_users = movies_df.User_ID.unique().shape[0]
n_movies = movies_df.Movie_ID.unique().shape[0]
print( str(n_users) + ' users')
print( str(n_movies) + ' movies')


# ## Task 4: Create an Interaction Matrix

# In[7]:


#This would be a 2D array matrix to display user-movie_rating relationship
#Rows represent users by IDs, columns represent movies by IDs
ratings = np.zeros((n_users, n_movies))
for row in movies_df.itertuples():
    ratings[row[1], row[3]-1] = row[4]

# View the matrix
print(ratings)


# ## Task 5: Explore the Interaction Matrix

# In[8]:


sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print(sparsity)


# ## Task 6 : Create a Similarity Matrix

# In[9]:


rating_cosine_similarity = cosine_similarity(ratings)


# ## Task 7: Provide Recommendations

# In[10]:


def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    # Get the location of the actual user in the User-Items matrix
    # Use it to index the User similarity matrix
    user_similarities = X_user[user]
    # obtain the indices of the top k most similar users
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    # Obtain the mean ratings of those users for all movies
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    # Discard already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    # return recommendations - top similar users rated movies
    rec_movies_a=rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: 'Movie_ID'}, inplace=True)
    return rec_movies_a


# ## Task 8: View the Provided Recommendations 

# In[11]:


#Converting the 2D array into a DataFrame as expected by the movie_recommender function
ratings_df=pd.DataFrame(ratings)


# In[12]:


user_ID=12
movie_recommender(ratings_df, rating_cosine_similarity,user_ID)


# ## Task 9: Create Wrapper Function

# In[13]:


def movie_recommender_run(user_Name):
    #Get ID from Name
    user_ID=movies_df.loc[movies_df['User_Names'] == user_Name].User_ID.values[0]
    #Call the function
    temp=movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec=temp.merge(movies_title_df, how='inner')
    return top_k_rec


# In[ ]:




