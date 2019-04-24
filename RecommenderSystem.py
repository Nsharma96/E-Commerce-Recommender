#we can apply fuzzy clustering and with collab filter given a actual dataset for more personalized recommendation
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv("data.csv",encoding = "ISO-8859-1")
# dataset.info()
print(dataset.head())

dataset.drop_duplicates(['Description'],inplace = True)
dataset.dropna(axis = 0, subset =['CustomerID'], inplace = True)
dataset = dataset[(dataset.InvoiceNo).apply(lambda x:( 'C' not in x))]

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataset['Description'])
tfidf_matrix.shape
# print(dataset.index[(dataset['Description'] == 'WHITE METAL LANTERN')])

from sklearn.metrics.pairwise import linear_kernel
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(des, cosine_sim=cosine_sim):
    # Get the index of the product that matches the Description
    idx = dataset[(dataset['Description'] == des)].index.tolist()
    print(idx[0])

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx[0]]))

    # Sort the product based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar product
    sim_scores = sim_scores[1:11]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar product
    return dataset['Description'].iloc[product_indices]

print("\n Top recommendations are \n ")
print(get_recommendations('RED WOOLLY HOTTIE WHITE HEART.'))

