from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
CORS(app)

credits_dataset=pd.read_csv('credits.csv')
movies_dataset=pd.read_csv('movies.csv')
movies_dataset=movies_dataset.merge(credits_dataset,on='title')
movies_dataset=movies_dataset[['movie_id','title','overview','genres','keywords','cast','crew']]
movies_dataset.dropna(inplace=True)

def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

def convert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter!=3:
      L.append(i['name'])
      counter=counter+1
    else:
      break
  return L

def find_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      L.append(i['name'])
  return L

movies_dataset['genres']=movies_dataset['genres'].apply(convert)
movies_dataset['keywords']=movies_dataset['keywords'].apply(convert)
movies_dataset['cast']=movies_dataset['cast'].apply(convert3)
movies_dataset['crew']=movies_dataset['crew'].apply(find_director)
movies_dataset['overview']=movies_dataset['overview'].apply(lambda x:x.split())
movies_dataset['genres']=movies_dataset['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_dataset['keywords']=movies_dataset['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_dataset['cast']=movies_dataset['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_dataset['crew']=movies_dataset['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies_dataset['tags']=movies_dataset['overview']+movies_dataset['genres']+movies_dataset['keywords']+movies_dataset['cast']+movies_dataset['crew']

new_df=movies_dataset[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

cv=CountVectorizer(max_features=5000,stop_words='english')
# cv.fit_transform(new_df['tags']).toarray().shape
vectors=cv.fit_transform(new_df['tags']).toarray()

ps=PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

similarity=cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommended(movie):
  movie_index=new_df[new_df['title']==movie].index
  if not len(movie_index):
      return []
  else:
     movie_index=movie_index[0]
  distances=similarity[movie_index]
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  recommended_movies = [] 
  for i in movies_list:
    recommended_movies.append(new_df.iloc[i[0]].title)
  return recommended_movies
# Sample movie data (you can replace this with your dataset)


# Recommendation algorithm (you can replace this with your own)
# def recommend_movies(favorite_movie):
#     recommended_movies = []
#     for movie in movies:
#         if movie["title"] != favorite_movie:
#             recommended_movies.append(movie["title"])
#     return recommended_movies
@app.route('/')
def index():
    # Extract movie titles from the DataFrame
    movie_titles = credits['title'].tolist()
    # Pass movie titles to the HTML template
    return render_template('index.html', movie_titles=movie_titles)
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    favorite_movie = data.get('favorite_movie')

    if not favorite_movie:
        return jsonify({'error': 'Please provide a favorite_movie parameter'}), 400

    recommended_movies = recommended(favorite_movie)
    if not len(recommended_movies):
        return jsonify({'error': 'No recommended movies found'}), 200
    return jsonify({'recommended_movies': recommended_movies}), 200

if __name__ == '__main__':
    app.run(debug=True)
