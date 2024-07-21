
# Netflix movies recommendations (LightGBM)

The goal of this project is to create a model that predicts how a user will rate various movies based on their past movie ratings. The model will then provide the top 10 movie recommendations, highlighting those the user is most likely to rate highly.
## Data
❗ Download data and put it in `Data` folder:
- `combined_data_1.txt`
- `movie_titles.csv`
### Combined Data

The file `combined_data_1.txt` contains movie rating information. Each line corresponds to a rating from a customer in the following format:

```
CustomerID,Rating,Date
```

Key details include:

- **MovieIDs:** Range sequentially from 1 to 17,770.
- **CustomerIDs:** Range from 1 to 2,649,429, with gaps, representing 480,189 users.
- **Ratings:** Are on a five-star (integral) scale from 1 to 5.
- **Dates:** Follow the format YYYY-MM-DD.

### Movie Information

Movie details are provided in the `movie_titles.csv` file, formatted as:

```
MovieID,YearOfRelease,Title
```

- **MovieID:** Does not correspond to actual Netflix or IMDB IDs.
- **YearOfRelease:** Ranges from 1890 to 2005, possibly indicating DVD release rather than theatrical release.
- **Title:** Represents the Netflix movie title, which may differ from titles on other sites. Titles are in English.

Link to daata: [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data)
## Installation

To install the necessary packages, please ensure you have `pip` installed and run the following command:

```
pip install -r requirements.txt
```
    
## Recommendations model
### More data...
The file `Extra_data.ipynb` demonstrates how to fetch additional movie data from the TMDB (The Movie Database) API. This Jupyter Notebook includes code and examples to retrieve supplementary information such as movie genres, runtime, original language, popularity, and adult content rating. This additional data enriches the dataset and provides more features for model.

Key contents of `Extra_data.ipynb` include:
- Instructions for setting up and using the TMDB API.
- Code snippets to fetch the following features from the TMDB database:
  - `genres`
  - `runtime`
  - `original_language`
  - `popularity`
  - `adult`
- Examples of how to integrate the additional data with the existing dataset.

Link for API: [TMDB](https://developer.themoviedb.org/reference/intro/getting-started)

### Data Cleaning and Preparation

The file `Data_cleaning_and_preparation.ipynb` contains the steps and code necessary for cleaning and preparing the movie rating dataset. This Jupyter Notebook ensures that the data is in a suitable format for analysis and modeling by addressing any inconsistencies and performing essential preprocessing tasks.

Key contents of `Data_cleaning_and_preparation.ipynb` include:
- Handling missing values and duplicates.
- Formatting and transforming data types.
- Extracting and engineering features for analysis.
- Merging additional data fetched from the TMDB API with the primary dataset.

This notebook is crucial for ensuring the dataset is clean, consistent, and ready for model training.

### Model

The file `Model.ipynb` contains the code and processes for building, training, and evaluating the movie recommendation model. This Jupyter Notebook utilizes the cleaned and prepared dataset to develop a predictive model that can recommend movies to users based on their past ratings.

Key contents of `Model.ipynb` include:
1. Data splitting into training and test sets.
2. Feature selection and engineering.
3. Model selection and implementation (e.g., LightGBM).
4. Hyperparameter tuning and optimization.
5. Model evaluation and performance metrics.

**Achievements:**
1. Successfully built and trained a LightGBM model for movie recommendations.
2. Achieved high accuracy and relevance in movie predictions based on user history.
3. Implemented efficient data processing and feature engineering techniques.
4. Evaluated the model using appropriate performance metrics to ensure robustness.

**Why LightGBM was selected:**

LightGBM was chosen for its efficiency and high performance in handling large datasets with many features. Its ability to perform well with minimal tuning and the support for parallel and GPU learning makes it an excellent choice for recommendation systems. Feature engineering significantly enhanced the model's performance, making it more accurate and reliable.

- **Cross-validated Comparison of Surprise Algorithms**:

  ![Comparison of Algorithms](./Pictures/Screenshot%202024-07-21%20170609.png)

  This picture from the job by [MorrisB on Kaggle](https://www.kaggle.com/code/morrisb/how-to-recommend-anything-deep-recommender) shows the relatively poor performance of various surprise algorithms in terms of RMSE and MAE. Due to this, we chose a different approach with LightGBM, focusing on better feature engineering to improve our model's accuracy and relevance.

- **Model Performance**:
  - Final RMSE: 0.45372
  - Precision: 0.91937
  - Recall: 0.93061
  - F1-score: 0.92496
  - AUC-ROC: 0.91516

Feature engineering played a crucial role in improving the model's performance.

❗ Model is saved in [Model folder](Model)

### Recommendations Test

The file `Recommendations_test.py` contains the code to load the trained LightGBM model and the prepared data, and to generate movie recommendations for a given user. This script demonstrates how to utilize the model and data to provide personalized movie suggestions.

Key contents of `Recommendations_test.py` include:
- Error handling for file loading to ensure robustness.
- The `generate_recommendations` function, which:
  - Takes a user ID, the dataset, the trained model, and movie titles as input.
  - Generates a list of top N recommended movies for the specified user.
  - Merges additional movie information for better insights.
- A test section that selects a random user who has watched more than 20 movies and generates recommendations for them.

## Docker Setup and Running the Application

This section explains how to set up and run the movie recommendation system using Docker. The key files involved are `Dockerfile`, `app.py`, and `requirements_docker.txt`.

### Dockerfile

The `Dockerfile` contains the instructions to build the Docker image for the application. It sets up the environment, installs necessary dependencies, and copies the application code into the image.

### app.py

The `app.py` file contains the Flask application that serves the movie recommendations. It loads the trained LightGBM model and the necessary data, and defines an endpoint to generate recommendations for a given user ID.

Key contents of `app.py` include:
- Loading the trained LightGBM model.
- Loading the dataset and movie titles.
- Defining the `generate_recommendations` function to produce movie recommendations.
- Creating a Flask endpoint `/recommend` to handle GET requests for recommendations.

### requirements_docker.txt

The `requirements_docker.txt` file lists all the Python dependencies needed to run the application. It ensures that the correct versions of the libraries are installed.

Contents of `requirements_docker.txt`:
```
Flask
pandas
lightgbm
confluent_kafka
pyarrow
fastparquet
```

### How to Run the Application

1. **Build the Docker Image**

   Open a terminal in the directory containing the `Dockerfile` and run the following command to build the Docker image:

   ```sh
   docker build -t movie-recommendation-app .
   ```

2. **Run the Docker Container**

   After building the image, run the following command to start the Docker container:

   ```sh
   docker run -p 5000:5000 movie-recommendation-app
   ```

   This command maps port 5000 on your local machine to port 5000 in the Docker container, allowing you to access the Flask application.

3. **Test the Recommendation Endpoint**

   Once the container is running, you can test the recommendation endpoint by opening a web browser or using a tool like `curl` or Postman to send a GET request to:

   ```
   http://localhost:5000/recommend?user_id=<USER_ID>
   ```

   Replace `<USER_ID>` with the user ID for which you want to generate recommendations(e.g. 458888). The application will return a JSON response with the top recommended movies for that user.

## In summary,

the project highlights the importance of algorithm selection and feature engineering in developing an effective recommendation system. The LightGBM model, supported by well-crafted features, demonstrated excellent predictive capabilities, making it a valuable tool for movie recommendations. The use of Docker further facilitates easy deployment and scalability of the application.

Example of output:
![Output](./Pictures/Screenshot%202024-07-21%20183026.png)