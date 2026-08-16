# Bookly

Bookly predicts the average rating a book is likely to receive, based only on its catalog information such as page count, publisher, author, language, and how widely it has been read. It is an end to end machine learning project built for the Book Rating Prediction assignment at DSTI School of Engineering.

The project covers the full pipeline: exploratory data analysis, data cleaning, feature engineering, model training and evaluation, and a deployed web application that returns a prediction from user input.

## What the project does

Given a book's metadata, Bookly estimates its average rating on a scale of roughly 1 to 5. The final model is a tuned Gradient Boosting Regressor that reaches a test RMSE of about 0.257 and an R squared of about 0.26, comfortably beating a naive baseline that always predicts the mean rating (RMSE 0.30).

The web application offers two ways to get a prediction:

1. Single book mode. Fill in a short form and get one predicted rating with a confidence band.
2. Batch mode. Upload a CSV of books and download a table with a predicted rating for each row.

## Dataset

The dataset contains about 11,000 books with 12 attributes each, including title, authors, average rating, ISBN, language code, page count, ratings count, text reviews count, publication date, and publisher. The raw file is imperfect and needs cleaning before use, which the pipeline handles.

## Repository structure

```
bookly/
├── notebook/
│   ├── bookly_EDA.ipynb                  Exploratory data analysis
│   ├── bookly_cleaning.ipynb             Data cleaning
│   ├── bookly_feature_engineering.ipynb  Feature engineering and selection
│   ├── bookly_training.ipynb             Model training, comparison, evaluation
│   ├── analyses-draft.ipynb              Working notes
│   ├── *.pkl                             Saved encodings and model
│   ├── feature_columns.json              Column order expected by the model
│   └── plot_*.png                        Generated figures
│
├── app/
│   ├── app.py                            Streamlit web application
│   ├── predictor.py                      Shared prediction logic
│   ├── bookly_model.pkl                  Trained model pipeline
│   ├── publisher_encoding.pkl            Publisher target encodings
│   ├── author_encoding.pkl               Author target encodings
│   ├── global_mean.pkl                   Fallback value for unseen categories
│   └── feature_columns.json              Column order expected by the model
│
├── project_info/                         Assignment brief and group work sheet
├── detailed_explanation.md               In depth walkthrough of the harder steps
├── requirements.txt                      Python dependencies
└── README.md                             This file
```

## How to run

### 1. Set up the environment

Clone the repository and install the dependencies. Python 3.9 or newer is recommended.

```bash
git clone https://github.com/samitochi04/bookly.git
cd bookly
pip install -r requirements.txt
```

### 2. Reproduce the pipeline

Open the notebooks in order and run all cells. Each notebook writes the files the next one needs, so the order matters:

```
bookly_EDA.ipynb
bookly_cleaning.ipynb
bookly_feature_engineering.ipynb
bookly_training.ipynb
```

Running the full sequence regenerates the cleaned dataset, the train and test splits, the encoding maps, and the trained model.

### 3. Launch the web application

From the app folder, start Streamlit:

```bash
cd app
streamlit run app.py
```

Streamlit opens the app in your browser, usually at http://localhost:8501. Enter a book's details and press Predict rating to see the estimate, or switch to the Upload a CSV tab to score a whole list at once.

## The machine learning pipeline

The project follows the standard steps taught in class.

**Exploratory data analysis.** Inspect the raw data, study the distribution of the target rating, check for missing and impossible values, and measure how each column relates to the rating.

**Data cleaning.** Remove rows with an unreliable target (zero ratings), impute impossible page counts, parse publication dates into a usable year, normalize the language codes, and confirm that odd looking values such as the publisher named "10/18" are in fact real.

**Feature engineering and selection.** Turn text and categories into numbers. Extract series and title signals from the title, count authors, log transform the skewed count columns, one hot encode the language, and target encode the publisher and author using training data only. A feature importance check then drops the columns that carry no signal.

**Model training and evaluation.** Compare six regressors against a naive baseline, validate them with five fold cross validation, tune the two strongest with grid search, and select the best. Evaluate with RMSE, MAE, and R squared, supported by actual versus predicted plots, residual analysis, and a learning curve.

For a full, plain language explanation of feature engineering, model training, and evaluation, see `detailed_explanation.md`.

## Results summary

| Model | RMSE | MAE | R squared |
| --- | --- | --- | --- |
| Gradient Boosting (tuned) | 0.257 | 0.184 | 0.260 |
| Random Forest | 0.257 | 0.184 | 0.262 |
| Ridge Regression | 0.259 | 0.189 | 0.251 |
| Linear Regression | 0.259 | 0.189 | 0.251 |
| K Nearest Neighbors | 0.263 | 0.189 | 0.225 |
| Decision Tree | 0.267 | 0.191 | 0.201 |
| Naive baseline (predict mean) | 0.300 | | 0.000 |

The strongest predictors are author reputation and publisher reputation, both captured through target encoding, followed by how widely a book has been read and its page count. The model explains about a quarter of the variation in ratings, which is a fair result given that the most important drivers of a rating, such as writing quality and reader taste, are not present in catalog metadata.

## Limitations and possible improvements

Bookly predicts from metadata alone, so it cannot see the content of a book. The remaining variation in ratings comes from factors the dataset does not contain. Natural next steps would be to add genre information, incorporate review text through natural language processing, bring in author popularity signals, or move toward collaborative filtering using reader interaction data, which is how large book platforms achieve much higher accuracy.

## Team

See `project_info` for the group work distribution sheet.

## Course

Book Rating Prediction project, Applied MSc programs, DSTI School of Engineering. Instructor: Hanna Abi Akl.
