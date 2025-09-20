
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("..\data\IMDB Dataset.csv")  
df.head()


from wordcloud import WordCloud


positive_reviews = " ".join(df[df["sentiment"]=="positive"]["review"])

wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_reviews)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Positive Reviews")
plt.show()
