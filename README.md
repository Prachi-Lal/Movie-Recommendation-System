# Movie Recommendation System using Bag of Words

This repository contains the code for building a content-based movie recommendation system using the Bag of Words technique. The recommendation system suggests similar movies based on the textual features of the movies' plots.

## Summary

The content-based movie recommendation system leverages the concept of Bag of Words to analyze the textual content of movie plots and determine their similarity. The system suggests movies to users based on the similarity of their plots to movies they have previously liked or rated.

## Prerequisites

To run this code, ensure that you have the following dependencies installed:

- Python (version 3.9)
- pandas
- scikit-learn
- numpy
- nltk
- ast

## Dataset

The dataset used for building the recommendation system is stored in the `tmdb_5000_credits.csv`  and <code>tmdb_5000_credits.csv</code>  files.

## Usage

1. Clone this repository:
<code>git clone [https://github.com/Prachi-Lal/Movie-Recommendation-System.git](https://github.com/Prachi-Lal/Movie-Recommendation-System.git)</code>

2. Navigate to the project directory:
<code>cd Movie-Recommendation-System</code>


3. Install the required dependencies:
<code>pip install -r requirements.txt</code>


4. Run the recommendation system:
<code> python movie-recommender-system.py</code>


The system will prompt you to enter the title of a movie that you like. Based on the movie's plot, it will recommend similar movies from the dataset.

## Approach

The recommendation system follows these steps to suggest similar movies:

1. Preprocess the movie plots by removing stopwords, punctuation, and applying stemming or lemmatization techniques.
2. Represent each movie plot as a numerical vector using the Bag of Words approach.
3. Calculate the similarity between movies by comparing their plot vectors using cosine similarity.
4. Identify the top 5 most similar movies to the user's input movie.
5. Display the recommended movies to the user.

## Results

The recommendation system successfully suggests similar movies based on the input movie's plot. It maps the top 5 similar movies using [cosine similarities](https://towardsdatascience.com/using-cosine-similarity-to-build-a-movie-recommendation-system-ae7f20842599#:~:text=Using%20the%20Cosine%20Similarity,-We%20will%20use&text=Mathematically%2C%20it%20measures%20the%20cosine,the%20items%20are%20100%25%20similar.)

## Contributing

Contributions to this project are welcome. If you find any issues or want to enhance the recommendation system, feel free to submit a pull request.

## Acknowledgments

- The dataset used in this project is sourced from [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
- The Bag of Words approach is inspired by various research papers and online tutorials.


