#Core Pkg
import streamlit as st
import streamlit.components.v1 as stc

#Load EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
# from surprise import SVD, Reader, Dataset 
# from surprise.model_selection import cross_validate

#Load data
def load_data(data):
    df = pd.read_csv(data)
    return df
# Fxn
# Vectorize +Cosine Similarity Matrix
def vectorize_to_cosine_mat(data):
    tfidfv=TfidfVectorizer(analyzer='word', stop_words='english')
    tfidfv_matrix=tfidfv.fit_transform(data)
    # print(tfidfv_matrix.todense())
    # tfidfv_matrix.todense().shape
    # Computing Similarity Score based on movie overview
    cosine_sim1 = linear_kernel(tfidfv_matrix, tfidfv_matrix)
    return cosine_sim1

# Recommendation system

# Function that takes in movie title as input and outputs most similar movies

@st.cache
def content_recommendations(title, cosine_sim,df,num_of_rec = 10):
    indices = pd.Series(df.index, index= df['title']).drop_duplicates()
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    selected_movie_indices = [i[0] for i in sim_scores[1:]]
    selected_movie_scores = [i[1] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_movie_indices]
    result_df['similarity_score'] = selected_movie_scores
    final_recommended_movies = result_df[['title','similarity_score','homepage']]
    return final_recommended_movies.head(num_of_rec)

#CSS style
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
</div>
"""
    # sim_scores=sim_scores[1:11]
    # # Get the movie indices
    # ind=[]
    # for (x,y) in sim_scores:
    #     ind.append(x)
        
    # # Return the top 10 most similar movies
    # tit=[]
    # for x in ind:
    #     tit.append(df.iloc[x]['title'])
    # return pd.Series(data=tit, index=ind)

# Search for a movie
def search_not_found(term,df):
    result_df = df[df['title'].str.contains(term)]
    return result_df

def main():
    st.title("Movie Recommender App")

    menu = ['Home','Recommend','About']
    choice = st.sidebar.selectbox("Menu",menu)

    df = load_data('df_movie.csv')
    if choice =='Home':
        st.subheader("Home")
        st.dataframe(df.head(10))
    elif choice == 'Recommend':
        st.subheader("Recommend Movies")
        search_term = st.text_input('Search')
        cosine_sim_1 = vectorize_to_cosine_mat(df['overview'])

        num_of_rec = st.sidebar.number_input("Number",5,20,10)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    result = content_recommendations(search_term , cosine_sim_1,df,num_of_rec)
                    with st.beta_expander("Results as JSON"):
                        results_json = result.to_dict('index')
                        st.write(results_json)
                        
                    for row in result.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_url = row[1][2]
                        # st.write('Title', rec_title)
                        stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url))
                except:
                    result = "Not Found"
                    st.warning(result)
                    st.info('Here are some suggested options')
                    result_df = search_not_found(search_term,df)
                    st.dataframe(result_df)
                

    else:
        st.subheader('About')
        st.text('Built with Streamlit & Pandas')



if __name__ == '__main__':
    main()