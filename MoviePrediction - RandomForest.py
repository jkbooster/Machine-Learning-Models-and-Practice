import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from google.colab import drive
drive.mount("/content/drive")

movies_df = pd.read_csv("/content/drive/MyDrive/movies.csv")
movies_df

copy_movies_df = movies_df

copy_movies_df.dropna(axis=0, subset = ["title", "release_date", "imdb_id", "original_title", "overview", "poster_path", "genres", "keywords"], inplace = True)

copy_movies_df.drop(columns = ["backdrop_path", "homepage", "tagline", "production_companies", "production_countries", "spoken_languages"], axis = 1, inplace = True)

copy_movies_df = copy_movies_df[copy_movies_df["vote_average"] != 0]

copy_movies_df["release_date"] = pd.to_datetime(movies_df["release_date"])
copy_movies_df["release_year"] = movies_df["release_date"].dt.year
copy_movies_df["release_5year"] = (copy_movies_df["release_year"] // 5) * 5
copy_movies_df["release_5year"].head()

copy_movies_df.isnull().sum()

selected_columns = ["budget", "revenue", "popularity", "vote_average"]
sns.pairplot(data = movies_df[selected_columns], kind = "scatter", plot_kws = {"alpha": 0.3})
plt.suptitle("Pair Plot: Budget, Revenue, Popularity, Vote Average", y = 1.02)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data = copy_movies_df, x = "release_5year", y = "vote_average")
plt.title("Distribution of Vote Averages Every 5 Years")
plt.xlabel("Release Year")
plt.ylabel("Vote Average")
plt.xticks(rotation = 45)
plt.show()

copy_movies_df["hit_flop"] = (copy_movies_df["revenue"] > 2 * copy_movies_df["budget"]).astype(int)

copy_movies_df["hit_flop"]

genre_dummies = copy_movies_df["genres"].str.get_dummies(sep=", ")
copy_movies_df = copy_movies_df.join(genre_dummies, lsuffix='_caller', rsuffix='_other')
copy_movies_df.head()

genre_cols = genre_dummies.columns.tolist()
numeric_cols = ["budget", "popularity", "vote_average"]
feature_cols = numeric_cols + genre_cols
X = copy_movies_df[feature_cols]
y = copy_movies_df["hit_flop"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

randomforest1 = RandomForestClassifier(n_estimators=100, random_state=42)

randomforest1.fit(x_train, y_train)

forestpredict1 = randomforest1.predict(x_test)
acc1 = accuracy_score(y_test, forestpredict1)
acc1

importances = randomforest1.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': x_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df)

mini_df = copy_movies_df.iloc[:10000]
plt.figure(figsize = (8, 6))

sns.scatterplot(data = mini_df, x = "budget", y = "vote_average", hue = "hit_flop", alpha = 0.5)
plt.xscale("log")
plt.ylim(0, 10)
plt.title("Budget vs. Vote Average Colored by Hit/Flop")
plt.xlabel("Budget")
plt.ylabel("Vote Average")
plt.show()

copy_movies_df.to_csv('movies_new.csv', index=False)

















