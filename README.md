# IMDB Movie Taste Clustering Project

## Project Overview

This project uses the IMDB Movie Dataset to explore whether movies can be grouped into meaningful “taste clusters” based on metadata such as genre, ratings, popularity, revenue, runtime, release year, and critical reception. The project begins with exploratory data analysis and unsupervised clustering, then extends the clustering results into a supervised prediction task where several models attempt to predict a movie’s assigned cluster.

The core idea is simple: if movies naturally cluster into recognizable groups, those groups can be interpreted as rough viewer preference profiles. For example, one cluster may represent high-budget action and adventure blockbusters, while another may represent lower-revenue dramas or thrillers. The project tests whether these groupings are visible in the data and whether machine learning models can reliably reproduce them.

## Repository Contents

| Notebook | Purpose |
|---|---|
| `Explanatory_Analysis_(EDA)_HW3.ipynb` | Initial exploratory analysis, feature engineering, baseline k-means clustering, genre and actor exploration, and an early classification model. |
| `Colab_Final (1).ipynb` | Final modeling notebook that refines the clustering approach, uses grouped genres, compares supervised models, trains a neural network, and evaluates Gemini 2.5 Flash as an LLM-based classifier. |

## Dataset

The notebooks use the IMDB Movie Dataset from Kaggle:

**Dataset:** IMDB data of 1,000 movies from 2006 to 2016  
**Source:** [Kaggle - IMDB Data from 2006 to 2016](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data?resource=download)

The dataset includes the following columns:

- `Rank`
- `Title`
- `Genre`
- `Description`
- `Director`
- `Actors`
- `Year`
- `Runtime (Minutes)`
- `Rating`
- `Votes`
- `Revenue (Millions)`
- `Metascore`

## Main Research Goal

The main goal is to use movie metadata to discover natural movie groupings that could represent viewer taste profiles.

More specifically, the project asks:

1. Can k-means clustering identify meaningful movie groups?
2. Which features are most useful for clustering movies?
3. Do genres improve clustering quality, or do they add sparse noise?
4. Can supervised models learn to predict the clusters created by k-means?
5. How do traditional machine learning models compare to a neural network and an LLM classifier?

## Notebook 1: Exploratory Analysis and Baseline Clustering

### File

`Explanatory_Analysis_(EDA)_HW3.ipynb`

### Purpose

This notebook serves as the foundation of the project. It explores the IMDB dataset, identifies important patterns, creates clustering-ready features, and tests an initial k-means clustering approach.

### Main Steps

#### 1. Data Loading

The notebook loads `IMDB-Movie-Data.csv` from Google Drive in a Google Colab environment.

```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('IMDB-Movie-Data.csv')
```

#### 2. Data Overview

The notebook inspects the dataset structure using:

- `df.info()`
- `df.head()`
- numeric summaries with `describe()`
- missing value counts and percentages

Important missing values found:

| Column | Missing Count | Missing Percent |
|---|---:|---:|
| `Revenue (Millions)` | 128 | 12.8% |
| `Metascore` | 64 | 6.4% |

The notebook notes that missing revenue values do not appear to be obviously tied to a single year.

#### 3. Feature Engineering

Several new features are created to make the dataset usable for clustering:

- `Genre_List`: splits comma-separated genre strings into lists
- `Actors_List`: splits comma-separated actor strings into lists
- `Log_Votes`: log-transformed vote count using `np.log1p`
- `Revenue_Missing`: indicator for missing revenue values
- `Metascore_Missing`: indicator for missing Metascore values

The log transformation is important because vote counts are heavily skewed. A small number of highly popular movies receive extremely high vote totals, so using raw votes would allow popularity to dominate the clustering too strongly.

#### 4. Exploratory Data Analysis

The notebook explores both numeric and categorical variables.

Numeric EDA includes:

- Histograms of numeric variables
- Summary statistics
- Correlation matrix
- Scatter plots comparing variables such as:
  - Rating vs. Votes
  - Rating vs. Revenue
  - Runtime vs. Rating
  - Metascore vs. Rating
  - Year vs. Rating

Categorical EDA includes:

- Genre frequency analysis
- Director frequency analysis
- Lead actor frequency analysis
- Genre-level summaries for:
  - average rating
  - average votes
  - average revenue
  - average Metascore
  - average runtime
  - movie count

#### 5. Genre Encoding

The notebook uses `MultiLabelBinarizer` to one-hot encode genres. Because movies can belong to multiple genres, this approach is better than treating each full genre string as a separate category.

Example:

```text
Action,Adventure,Sci-Fi
```

becomes:

```text
Genre_Action = 1
Genre_Adventure = 1
Genre_Sci-Fi = 1
```

This produces a feature matrix with 26 total columns:

- 6 numeric features
- 20 genre dummy variables

#### 6. Baseline K-Means Clustering

The notebook builds a k-means clustering model using:

- `Rating`
- `Log_Votes`
- `Revenue (Millions)`
- `Metascore`
- `Runtime (Minutes)`
- `Year`
- one-hot encoded genre columns

Missing numeric values are filled with each column’s median before scaling.

The features are standardized using `StandardScaler`, then k-means clustering is tested for several values of `k`.

The initial model uses `k = 3`.

#### 7. Cluster Interpretation

The notebook profiles clusters by comparing average numeric values and top genres within each cluster.

For the three-cluster model, the clusters are roughly interpretable as:

| Cluster | Approximate Interpretation |
|---|---|
| Cluster 0 | Higher-rated, drama-heavy films with stronger Metascores and longer runtimes |
| Cluster 1 | Lower-rated, lower-vote, lower-revenue films with many dramas, comedies, horror, and thrillers |
| Cluster 2 | Higher-revenue, high-vote adventure/action-heavy movies |

The cluster profile output showed:

| Cluster | Avg. Rating | Avg. Votes | Avg. Revenue | Avg. Metascore | Avg. Runtime |
|---|---:|---:|---:|---:|---:|
| 0 | 7.36 | 188,942 | 48.56 | 69.28 | 122.81 |
| 1 | 6.10 | 61,092 | 34.45 | 50.33 | 101.59 |
| 2 | 6.81 | 289,610 | 167.11 | 58.36 | 117.23 |

#### 8. PCA and t-SNE Visualization

The notebook uses PCA and t-SNE to visualize the clusters in two dimensions.

These visualizations help show whether the clusters are visibly separated or heavily overlapping.

#### 9. Numeric-Only Baseline Comparison

The notebook compares the full model against a numeric-only k-means baseline.

| Model | Number of Features | k | Silhouette Score |
|---|---:|---:|---:|
| Numeric-only K-Means | 6 | 3 | 0.199 |
| Numeric + Genre K-Means | 26 | 3 | 0.087 |

This result suggests that adding sparse genre dummy variables decreased the silhouette score. The likely reason is that genre one-hot encoding adds many sparse binary columns, which can distort distance calculations in k-means.

The notebook identifies possible improvements:

1. Increase the weight of genre features.
2. Group similar genres together.
3. Reduce sparse genre categories.
4. Use only the most common genres.
5. Try alternative clustering methods.

#### 10. Actor Exploration

The notebook also tests whether actor features improve clustering. It creates actor dummy variables for actors appearing in more than five movies.

This expands the feature matrix to 221 columns.

However, the actor-based clustering is not very meaningful because most actors appear too infrequently in a 1,000-movie dataset. The actor matrix is very sparse, so it adds dimensionality without adding much stable signal.

#### 11. Initial Classification

At the end of the notebook, logistic regression is used to predict k-means cluster labels.

The model achieves:

| Model | Train Accuracy | Test Accuracy |
|---|---:|---:|
| Logistic Regression | 1.000 | 0.915 |

This shows that the cluster labels are learnable from the engineered features, although the perfect training accuracy suggests possible overfitting or a task that is mechanically easy because the classifier is learning labels generated from the same input features.

## Notebook 2: Final Modeling and Model Comparison

### File

`Colab_Final (1).ipynb`

### Purpose

This notebook expands the original EDA and clustering work into a final modeling pipeline. It refines the feature engineering, creates grouped genre features, chooses a larger number of clusters, and compares several methods for predicting the k-means cluster assignments.

The final notebook includes:

- grouped genre engineering
- k-means clustering with `k = 11`
- statistical tests across clusters
- random forest classification
- logistic regression classification
- neural network classification
- Gemini 2.5 Flash classification
- final model comparison

### Main Steps

#### 1. Data Loading and Setup

The notebook loads the same IMDB dataset from Google Drive and imports libraries for:

- data manipulation
- visualization
- scaling
- clustering
- model evaluation
- supervised machine learning
- neural networks
- Gemini API usage

Main libraries include:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `statsmodels`
- `tensorflow`
- `google-genai`

#### 2. Genre Grouping

The final notebook improves on the original sparse genre encoding problem by grouping related genres into broader categories.

Examples of grouped genre categories include:

- `Drama`
- `High_Energy`
- `Comedy`
- `Scary/Suspenseful`
- `Romance`
- `Sci-Fi/Fantasy`
- `Family/Animated`
- `Other`

This is a major improvement over using all raw genre dummy variables because it reduces sparsity and makes each genre-related feature more meaningful.

#### 3. Feature Matrix

The final clustering feature matrix includes numeric movie features and grouped genre features.

Numeric features include:

- `Rating`
- `Log_Votes`
- `Revenue (Millions)`
- `Metascore`
- `Runtime (Minutes)`
- `Year`

Missing numeric values are filled with median values.

The resulting feature matrix has:

| Matrix | Shape |
|---|---|
| `X` | `(1000, 26)` |
| `X_scaled` | `(1000, 26)` |

#### 4. K-Means Clustering

The notebook tests several values of `k` using inertia and silhouette score.

For `k = 2` through `k = 20`, the strongest silhouette score occurs around `k = 11`.

The final model uses:

```python
FINAL_K = 11
```

The final k-means model assigns each movie to one of 11 clusters.

The silhouette score for `k = 11` is approximately:

```text
0.215
```

This is higher than the earlier raw genre model and supports the decision to group genres before clustering.

#### 5. Cluster Profiling

The notebook profiles the 11 clusters using averages for:

- Rating
- Votes
- Revenue
- Metascore
- Runtime
- Year
- grouped genre features

This helps interpret each cluster as a different movie type or viewer taste profile.

For example, one cluster has very high average votes and revenue, suggesting a blockbuster-oriented cluster:

| Cluster | Avg. Rating | Avg. Votes | Avg. Revenue | Avg. Metascore | Avg. Runtime |
|---|---:|---:|---:|---:|---:|
| 5 | 7.24 | 476,255 | 261.02 | 64.39 | 133.77 |

Another cluster has higher ratings and Metascores but more moderate revenue, suggesting a more critically respected or drama-heavy group:

| Cluster | Avg. Rating | Avg. Votes | Avg. Revenue | Avg. Metascore | Avg. Runtime |
|---|---:|---:|---:|---:|---:|
| 9 | 7.25 | 160,711 | 51.65 | 69.87 | 125.33 |

#### 6. Statistical Testing

The notebook uses ANOVA to test whether features differ significantly across clusters.

It also uses Tukey HSD tests to compare cluster pairs for each feature.

This section helps determine whether the clusters are statistically distinguishable, not just visually or intuitively different.

#### 7. Supervised Cluster Prediction

Once k-means cluster labels are created, the project treats cluster prediction as a supervised classification task.

The target variable is:

```python
y_clf = movies_genregroup["Cluster"]
```

The model input is the same scaled feature matrix used for clustering.

The train/test split uses:

- 80% training data
- 20% test data
- stratification by cluster label
- random seed of 42

### Models Compared

The final notebook compares four approaches:

1. Logistic Regression
2. Random Forest
3. Feedforward Neural Network
4. Gemini 2.5 Flash

## Model Results

### Final Comparison

| Method | Category | Task | Accuracy | Evaluation Size |
|---|---|---|---:|---:|
| Logistic Regression | Baseline | Predict K-Means cluster | 0.965 | 200 |
| Random Forest | Ensemble | Predict K-Means cluster | 0.920 | 200 |
| Feedforward Neural Network | Neural Network | Predict K-Means cluster | 0.945 | 200 |
| Gemini 2.5 Flash | Large Language Model | Predict K-Means cluster | 0.635 | 200 |

### Logistic Regression

Logistic regression performed the best overall.

| Metric | Value |
|---|---:|
| Test Accuracy | 0.965 |
| Cross-Validation Mean Accuracy | 0.981 |
| Cross-Validation Std. Dev. | 0.0086 |

This result makes sense because the k-means cluster labels were generated from the same standardized numeric feature space. Logistic regression can learn clean linear boundaries between many of those cluster regions.

### Random Forest

The random forest model achieved strong performance but was slightly weaker than logistic regression.

Best parameters:

```python
{
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 200
}
```

| Metric | Value |
|---|---:|
| Test Accuracy | 0.920 |
| Cross-Validation Mean Accuracy | 0.926 |
| Cross-Validation Std. Dev. | 0.0193 |

The random forest performs well, but it may be less naturally suited than logistic regression for reproducing k-means clusters in a standardized distance-based feature space.

### Feedforward Neural Network

The neural network uses a simple dense architecture:

- Input layer with 26 features
- Dense layer with 64 ReLU units
- Dropout layer
- Dense layer with 32 ReLU units
- Dropout layer
- Softmax output layer with 11 classes

| Metric | Value |
|---|---:|
| Train Accuracy | 0.995 |
| Test Accuracy | 0.945 |
| Train-Test Gap | 0.050 |

The neural network performs very well, but it does not beat logistic regression. Given the small dataset size and structured tabular feature space, the neural network is probably more complex than necessary.

### Gemini 2.5 Flash

The notebook evaluates Gemini 2.5 Flash by giving it cluster profiles and asking it to classify movies into one of the k-means clusters.

| Metric | Value |
|---|---:|
| Accuracy | 0.635 |
| Evaluated Rows | 200 |

Gemini performs worse than the traditional supervised models, but its performance is still meaningful because it is classifying from text-based cluster descriptions rather than directly learning the original standardized feature boundaries.

This result suggests that LLMs may be useful for interpretability or qualitative labeling, but they are not the best tool for reproducing precise k-means cluster assignments.

## Key Findings

### 1. Movie clusters are partially meaningful

The clusters are not perfect, but they show interpretable patterns based on genre, popularity, rating, revenue, and runtime.

Some clusters appear to capture broad movie types such as:

- blockbuster action/adventure movies
- critically stronger dramas
- lower-rating comedies, horror, or thrillers
- family/animation-oriented movies
- smaller or more niche films

### 2. Raw genre one-hot encoding can hurt k-means performance

The EDA notebook showed that adding all genre dummy variables reduced the silhouette score compared with a numeric-only baseline.

This happens because one-hot genre features are sparse and can distort distance calculations.

### 3. Grouped genres are more useful than raw genres

The final notebook improves the feature design by grouping genres into broader categories. This reduces sparsity and makes the clustering more stable.

### 4. Logistic regression is the strongest supervised model

Logistic regression achieved the highest final test accuracy at 96.5%.

Because k-means creates clusters using distance-based boundaries, a simpler linear model can reproduce those assignments surprisingly well. Sometimes the humble model pulls up in a Honda Civic and beats the sports cars.

### 5. Neural networks are strong but unnecessary for this dataset

The neural network achieved 94.5% test accuracy, but it did not outperform logistic regression. For this small tabular dataset, the extra complexity does not clearly improve performance.

### 6. Gemini is better for explanation than exact classification

Gemini achieved 63.5% accuracy when predicting clusters from profiles and movie descriptions. This is lower than the supervised ML models, but the LLM may still be valuable for naming clusters, summarizing cluster personalities, or explaining why a movie may fit a particular viewer group.

## Recommended Notebook Run Order

Run the notebooks in this order:

1. `Explanatory_Analysis_(EDA)_HW3.ipynb`
2. `Colab_Final (1).ipynb`

The first notebook explains the dataset, early EDA decisions, and baseline clustering. The second notebook contains the final modeling pipeline and comparison.

## How to Run

### Option 1: Google Colab

These notebooks are designed to run in Google Colab.

1. Upload the notebooks to Colab.
2. Mount Google Drive.
3. Make sure `IMDB-Movie-Data.csv` is located in the expected project folder.
4. Run cells from top to bottom.

The current notebooks assume the following working directory:

```python
/content/drive/Shareddrives/DSCI303 Group Project/
```

If your dataset is somewhere else, update the `os.chdir()` line or provide the full path to the CSV file.

### Option 2: Local Jupyter Notebook

To run locally:

1. Download the notebooks.
2. Place `IMDB-Movie-Data.csv` in the same folder.
3. Remove or comment out the Google Drive mounting code.
4. Replace the file loading line with:

```python
df = pd.read_csv("IMDB-Movie-Data.csv")
```

5. Install the required packages.

## Dependencies

Core dependencies:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
statsmodels
tensorflow
google-genai
```

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels tensorflow google-genai
```

The Gemini section also requires:

- access to Google Cloud / Vertex AI
- a valid project ID
- authentication in Colab
- permissions to use the selected Gemini model

The notebook uses:

```python
PROJECT_ID = "dsci303-test"
LOCATION = "global"
MODEL_ID = "gemini-2.5-flash"
```

These values may need to be changed depending on the user’s Google Cloud setup.

## Important Security Note

The notebooks currently contain hard-coded GitHub credentials/tokens in early Git setup cells.

Before sharing, submitting, or pushing these notebooks publicly:

1. Remove all hard-coded credentials.
2. Rotate/revoke the exposed GitHub token.
3. Clear notebook outputs if they contain sensitive information.
4. Use environment variables, Colab secrets, or GitHub authentication prompts instead of writing tokens directly into code.

A safer approach is:

```python
import os
token = os.environ.get("GITHUB_TOKEN")
```

or to use Colab’s built-in secrets manager.

## Limitations

### 1. K-means assumes spherical clusters

K-means works best when clusters are compact and roughly spherical in the feature space. Movie taste may not naturally follow this structure.

### 2. Cluster labels are not ground truth

The supervised models are predicting k-means labels, not true human-labeled taste groups. High classification accuracy means the model can reproduce k-means, not necessarily that the clusters are objectively correct.

### 3. Dataset is relatively small

The dataset contains only 1,000 movies. This limits the usefulness of high-dimensional features such as actors and makes neural networks less necessary.

### 4. Revenue and Metascore have missing values

Missing values are filled with medians, which is practical but may reduce nuance.

### 5. Actor data is too sparse

Actors appear too infrequently to create stable actor-based clusters in this dataset.

### 6. LLM evaluation depends on prompting

Gemini performance depends heavily on the prompt format, cluster descriptions, output parser, and sample size. A different prompt or more detailed cluster profile could change results.

## Possible Future Improvements

1. Try alternative clustering algorithms:
   - DBSCAN
   - Gaussian Mixture Models
   - Agglomerative Clustering
   - Spectral Clustering

2. Improve feature engineering:
   - weight genre features more carefully
   - group genres more systematically
   - include director-level features
   - use text embeddings from movie descriptions
   - normalize revenue by year

3. Improve cluster interpretation:
   - assign descriptive names to clusters
   - use Gemini to summarize clusters after k-means
   - compare clusters to known movie categories

4. Improve evaluation:
   - compare multiple random seeds
   - use adjusted Rand index for cluster stability
   - test whether clusters remain stable under feature changes

5. Build a recommendation system:
   - allow users to select favorite movies
   - identify their preferred cluster
   - recommend similar movies from the same cluster

## Final Takeaway

The project shows that IMDB movies can be grouped into interpretable clusters using metadata, especially when numeric features are combined with thoughtfully grouped genre features. The final 11-cluster model produces labels that traditional machine learning models can predict very accurately.

The strongest model is logistic regression, which achieved 96.5% accuracy on the test set. The neural network also performs well at 94.5%, while Gemini 2.5 Flash performs more modestly at 63.5%. Overall, the project demonstrates that simple, well-designed feature engineering and traditional machine learning can outperform more complex methods on structured tabular data.
