import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity








# search movie recommendation
@st.cache(suppress_st_warning=True)
def search_movie(option):


    ###### helper functions. Use them when needed #######
    def get_title_from_index(index):
            return df[df.index == index]["title"].values[0]

    def get_index_from_title(title):
            return df[df.title == title]["index"].values[0]
    ##################################################

    ##Step 1: Read CSV File

    df = pd.read_csv("https://raw.githubusercontent.com/jawaluke/streamlit-myfirst-app/master/movie_dataset.csv")
    df["title"] = [i.lower() for i in df["title"]]

    features = "keywords cast genres director".split()
    ##Step 2: Select Features

    # this for the combining the features where some features is NAN 
    for feature in features:

        df[feature] =df[feature].fillna("")

    ##Step 3: Create a column in DF which combines all selected features



    def combine_features(row):

        try:
            return row["keywords"]+" "+row["cast"]+" "+row["genres"]+" "+row["director"]
        except:
            print("ERROR")


    df["combined_features"] = df.apply(combine_features ,axis=1)

    ##Step 4: Create count matrix from this new combined column

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer()

    count = cv.fit_transform(df["combined_features"])



    ##Step 5: Compute the Cosine Similarity based on the count_matrix

    movie_user_like = option


    ## Step 6: Get index of this movie from its title

    movie_index = get_index_from_title(movie_user_like)
    similarity_scores = cosine_similarity(count)


    ## Step 7: Get a list of similar movies in descending order of similarity score
    # cosine similarity
    # get the index of this movies from the title

    similar_movie = list(enumerate(similarity_scores[movie_index]))

    # need to sorted the movie result based on the user liked movie

    sorted_similar_movie = sorted(similar_movie ,key = lambda x:x[1], reverse = True)
    ## Step 8: Print titles of first 50 movies

    # let see the result of the user liked movie recommendation
    # assume that we need first 10 movie recommendation
    i=0

    movie_list = []
    
    for movie in sorted_similar_movie:
        
        movie_list.append(get_title_from_index(movie[0]))    #------> it gives the index from the sorted movie by the function we can get the movie name
        
        if i>10:
            break
        i+=1

    return movie_list












if True:
    st.write("# Movie Recommendation")
    st.write("""
Enter the **Movie Name** see the result of the movie recommendation""")
    df = st.cache(pd.read_csv)("https://raw.githubusercontent.com/jawaluke/streamlit-myfirst-app/master/movie_dataset.csv")


    option = st.selectbox(" which movie do you like the most? ", df["title"].unique())
    st.write("you selected : ",option)

    result_movie = search_movie(option.lower())


    st.write(result_movie)         



