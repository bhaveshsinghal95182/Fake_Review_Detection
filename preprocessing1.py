# %%
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')

# %% [markdown]
# using head() here to quickly review the first row and some  of the following samples

# %%
df = pd.read_csv('fake reviews dataset.csv')
df.head()

# %% [markdown]
# Now we create a new data frame using isnull but the catch is that it contains "True" for every null value there is and then sums it all up 

# %%
df.isnull().sum()

# %% [markdown]
# the result is 0 for all the columns which means there is no null values in this dataset

# %%
df.info() #pretty much self explanatory

# %% [markdown]
# this will give the statistical overview of data

# %%
df.describe()

# %% [markdown]
# calculating the number of reviews for each rating

# %%
df['rating'].value_counts()

# %% [markdown]
# turning the above numerical data into graphical representation for better understanding

# %%
plt.figure(figsize=(15,8))
labels = df['rating'].value_counts().keys()
values = df['rating'].value_counts().values
explode = (0.1,0,0,0,0)
plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.title('Proportion of each rating',fontweight='bold',fontsize=25,pad=20,color='crimson')
plt.show()

# %%
def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

# %%
df['text_'][0], clean_text(df['text_'][0])

# %%
df['text_'].head().apply(clean_text)

# %%
df.shape

# %%
df['text_'] = df['text_'].astype(str)

# %%
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\b(?:' + '|'.join(stop_words) + r')\b|\d+|[^\w\s]', '', text)
    
    return text

# %%
preprocess(df['text_'][4])

# %%
df['text_'][:10000] = df['text_'][:10000].apply(preprocess)

# %%
df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)


# %%
df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)


# %%
df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)


# %%
df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)


# %%
df['text_'] = df['text_'].str.lower()


# %%
stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])
df['text_'] = df['text_'].apply(lambda x: stem_words(x))

# %%
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))

# %%
df['text_'].head()

# %%
df.to_csv('Preprocessed Dataset.csv')


