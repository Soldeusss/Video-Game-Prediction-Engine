# Video Game Metascore Prediction Engine

## Overview
This project is an end-to-end machine learning pipeline that predicts a video game’s Metacritic score using features such as user score, genre, platform, and playtime.

The dataset was constructed by cleaning and merging over 140,000 records from three sources. A Random Forest Regressor is used for prediction, with SHAP values providing interpretability into feature impact

## The Problem & Solution
* **The Problem:** Predicting the critical success of a video game is notoriously difficult due to subjective factors. Furthermore, gaming data is heavily fragmented across different websites with inconsistent naming conventions.
* **The Solution:** We engineered a consolidated dataset by cleaning and merging over 140,000 records from three distinct sources. We then applied a Random Forest Regressor and SHAP analysis to extract quantitative insights into what actually drives a game's metascore.

## Data Sources
The final dataset was engineered by joining three distinct sources:
* **Metacritic:** Critic scores, user scores and general game info. About ~11,000 records
* **HowLongToBeat (HLTB):** List of developers, release year and average player completion times (Main Story, Completionist, etc.). About ~51000 records, 25 features
* **GameFAQs:** Platform details, About ~82,000 records.

## Skills & Technologies Used
* **Languages:** Python
* **Data Cleaning & Manipulation:** `pandas`, `numpy` (Handling missing values, standardizing columns, complex table merges)
* **Feature Engineering:** One-Hot Encoding (converting 40+ overlapping text genres and multiple platforms into  ML-ready features).
* **Exploratory Data Analysis (EDA):** `matplotlib`, `seaborn` (Visualizing distributions and score vs. playtime correlations).
* **Machine Learning:** `scikit-learn` (For Predictive modeling).
* **Model Explainability:** `shap` (SHapley Additive exPlanations to extract feature importance).

## Data Engineering & Cleaning
Building the final dataset required significant preprocessing to ensure model accuracy:

* **Standardization:** Standardized column names across three sources (e.g., correcting source typos like `dificulty` in the GameFAQs set).
* **Multi-Label Genre:** The HLTB dataset stored genres as comma-separated strings (e.g., "Action, RPG, Open-World"). These were parsed and expanded into 40+ unique binary features.
* **Joining:** Performed complex inner-joins on `Title` and `Platform`. This required text normalization (removing special characters and standardizing case) to ensure that a game like "The Legend of Zelda" matched across all three databases.
* **Outlier Removal:** Filtered "Completionist" and "Main Story" hours to remove extreme outliers (e.g., games listed with 1,000+ hours) that would have skewed the model's perception of playtime vs. quality.

## Key Insights & Project Highlights
* **Complex Data Consolidation:** Successfully mapped distinct, messy sub-genres into a clean, validated matrix of primary genre features.
* **Explainable AI:** Rather than a "black box" prediction, the model outputs a breakdown of exactly which features pushed a specific game's score up or down using SHAP analysis. 
* **Reality:** While the Random Forest Regressor improved our baseline, the final $R^2$ score plateaued around ~0.55. A review of the residual plots showed significant noise, which led to an important realization about the domain: **human opinions are inherently subjective.**

Our model explains roughly 55% of the variance in a game's score based on quantitative metrics like Genre, Platform, and Playtime. However, the remaining 45% is likely driven by features that cannot be accurately measured, such as:

   *  Narrative Impact: The emotional weight of a story.

   *  Artistic Direction: How a game looks and feels beyond just "graphics."

   *  Nostalgia & Brand Loyalty: The "hype" factor associated with long-running franchises.

<img width="858" height="551" alt="randomreg" src="https://github.com/user-attachments/assets/80f5d6a9-57ef-4358-b1d5-d3857b46dfe9" />


<img width="852" height="545" alt="randomres" src="https://github.com/user-attachments/assets/e6ff6a5c-adfa-482d-b08a-e556c993e506" />

## Model Validation: Predicting an Unreleased Game
To evaluate the model's practical usefulness, we tested it against a game that had not yet received a Metacritic score at the time of prediction: *Call of Duty: Black Ops 7*.

### Prediction Results (Pre-Release)
| Metric | Value |
| :--- | :--- |
| **Predicted Metacritic Score** | 76.6 |
| **90% Prediction Interval** | 63.0 – 88.0 |
| **Standard Deviation** | 6.94 |

The Random Forest model predicted a score of 76.6, while estimating that 90% of its decision paths agreed the final score would likely fall between 63 and 88. The relatively small standard deviation suggested moderate confidence in the prediction.

### SHAP Explanation
The model identified several factors that contributed positively and negatively to the baseline prediction:

**Features Increasing the Score**
| Feature | SHAP Impact |
| :--- | :--- |
| Release Year | +1.35 |
| PC Platform | +0.96 |
| FPS Genre | +0.59 |
| First-Person Perspective | +0.33 |
| Difficulty Rating | +0.29 |

**Features Lowering the Score**
| Feature | SHAP Impact |
| :--- | :--- |
| Main Story Length | -0.65 |
| Third-Person Elements | -0.14 |
| Side Scroller | -0.06 |
| Open World | -0.04 |
| Sandbox | -0.03 |

### Actual Outcome
After release, *Call of Duty: Black Ops 7* received a Metacritic score of **65**.

While the model overestimated the point prediction by approximately 11.6 points, the actual score safely fell within the model's predicted 90% confidence interval (63–88).

### Key Takeaway
This example highlights both the strengths and limitations of predictive modeling in media. The model successfully identified a  score range based on quantitative metrics, but the difference between the predicted and actual score reinforces the core finding of this project: **human opinions are inherently subjective.**

## Challenges 
* **Issue one: Redundant Genre Labels:** The raw data contains overlapping genre columns (e.g. “Action", “Action-Adventure”, “Adventure”). Treating these as separate would dilute their importance.
  * **Solution:** We engineered a `merge_column` function to consolidate these genres into definitive categories. We also wrote validation logic to ensure rows didn't lose genre data during the data merge, dropping rows only if they lacked all genre information.

* **Issue two: Missing Data:** The merged datasets had null values and dropping the rows would mean we would lose a lot of data.
  * **Solution:** For critical missing features like "Playtime," we dropped rows to avoid noise. For other numeric fields, we imputed missing values using the column median or average to maintain distribution integrity.

* **Issue three: Model Underfitting:** Our initial Linear Regression model yielded an R² of only 0.51, indicating it wasn't capturing the full complexity of the data.
  * **Solution:** We switched to a Random Forest Regressor. This non-linear model was better able to handle the complex interactions between platform, genre, and playtimes.


## How to Run This Project
1. Clone this repository to your local machine.
2. Ensure you have Python installed, then install the required dependencies:
   ```bash
   pip install -r requirements.txt
## References & Data Sources
*Klein, Patrick. “Metacritic Scores for Games, Movies, TV & Music.” Kaggle, https://www.kaggle.com/datasets/patkle/metacritic-scores-for-games-movies-tv-and-music

*Klein, Patrick. “GameFAQs over 82,100 Rankings + Length & Difficulty.” Kaggle, https://www.kaggle.com/datasets/patkle/gamefaqs-82100-rankings-length-and-difficulty

*b4n4n4p0wer. “How Long To Beat Video Game Playtime Dataset.” Kaggle, https://www.kaggle.com/datasets/b4n4n4p0wer/how-long-to-beat-video-game-playtime-dataset
