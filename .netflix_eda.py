import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)
df = pd.read_csv("netflix_titles.csv")

print(df.head())
print("Rows and Columns:")
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.isnull().sum())
missing_percentage = (df.isnull().sum()/len(df))*100

print(missing_percentage)
print("Duplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.describe())
print(df.describe(include='object'))
print(df['type'].value_counts())
sns.countplot(x='type', data=df)

plt.title("Movies vs TV Shows")
plt.savefig("movies_vs_tvshows.png")
plt.close()
print(df['rating'].value_counts())
plt.figure(figsize=(10,5))

sns.countplot(
    y='rating',
    data=df,
    order=df['rating'].value_counts().index
)

plt.title("Ratings Distribution")
plt.savefig("ratings_distribution.png")
plt.close()
country_count = df['country'].value_counts().head(10)

print(country_count)
plt.figure(figsize=(10,6))

country_count.plot(kind='bar')

plt.title("Top 10 Countries")
plt.ylabel("Count")

plt.savefig("top_10_countries.png")
plt.close()
df['date_added'] = pd.to_datetime(
    df['date_added'],
    format='mixed',
    errors='coerce'
)
df['year_added'] = df['date_added'].dt.year
year_count = df['year_added'].value_counts().sort_index()

print(year_count)
plt.figure(figsize=(12,6))

sns.lineplot(
    x=year_count.index,
    y=year_count.values
)

plt.title("Netflix Content Added Per Year")
plt.savefig("netflix_content_added_per_year.png")
plt.close()
release_count = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12,6))

sns.lineplot(
    x=release_count.index,
    y=release_count.values
)

plt.title("Content Release Trend")
plt.savefig("content_release_trend.png")
plt.close()
top_directors = df['director'].value_counts().head(10)

print(top_directors)
plt.figure(figsize=(10,6))

top_directors.plot(kind='barh')

plt.title("Top Directors")
plt.savefig("top_directors.png")
plt.close()
genre_count = df['listed_in'].value_counts().head(10)

print(genre_count)
plt.figure(figsize=(12,6))

genre_count.plot(kind='bar')

plt.title("Top Genres")
plt.savefig("top_genres.png")
plt.close()
numeric_df = df.select_dtypes(include=np.number)

print(numeric_df.corr())
plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.savefig("heatmap.png")
plt.close()
plt.figure(figsize=(8,5))

sns.boxplot(
    x=df['release_year']
)

plt.title("Outlier Detection")
plt.savefig("outlier_detection.png")
plt.close()
print("\n===== EDA INSIGHTS =====")

print("1. Netflix has more Movies than TV Shows.")
print("2. USA is the largest content producer.")
print("3. Content addition increased significantly after 2015.")
print("4. Some columns contain missing values.")
print("5. Popular genres and ratings were identified.")
print("6. Release year distribution was analyzed using boxplot.")

