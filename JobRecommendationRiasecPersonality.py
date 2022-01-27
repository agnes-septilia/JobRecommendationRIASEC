### Import Libraries
import numpy as np
import pandas as pd
import streamlit as st

### STREAMLIT Introduction
st.markdown("<h1 style='text-align: center; color: black;'>JOB RECOMMENDATION BASED ON RIASEC PERSONALITY</h1>", unsafe_allow_html=True)

st.markdown("***")
st.markdown("<h3 style='text-align: center; color: black;'>Match your RIASEC personality with your dream career!!!</h3>", unsafe_allow_html=True)
st.write("""RIASEC stands for Realistic, Investigative, Artistic, Social, Enterprising, and Conventional.
Created by John Holland - also called Holland Code; RIASEC has been widely used to analyze someone's skill and interest in professional world.

In this program, choose the personalities, career field, and other input related.
And the program will find the suitable jobs for you.""")

st.markdown("***")

### Create function to read file from google drive link
def read_link (link):
    return 'https://drive.google.com/uc?export=download&id='+link.split('/')[-2]

### Combine Realistic dataset with other 5 personalities.
riasec_df = pd.concat([pd.read_csv(read_link('https://drive.google.com/file/d/1Yw8Q-okC156xESWz9ZdYJY8kOZpl3SjR/view?usp=sharing')), # Realistic
                       pd.read_csv(read_link('https://drive.google.com/file/d/1fTj0tFJtQ4htBEa1dpLVjhWe93PYOHA1/view?usp=sharing')), # Investigative
                       pd.read_csv(read_link('https://drive.google.com/file/d/1CR2IHnrhKC-7EtUjP5s-x8nkie6zSfiZ/view?usp=sharing')), # Artistic
                       pd.read_csv(read_link('https://drive.google.com/file/d/1Me5geVIjvEtMPldfzMOwBWavJDjy-e0c/view?usp=sharing')), # Social
                       pd.read_csv(read_link('https://drive.google.com/file/d/1WOTNJ7htmu5jR3gvaCPNLKg0ca1p3jaR/view?usp=sharing')), # Enterprising
                       pd.read_csv(read_link('https://drive.google.com/file/d/1crJJh-svX5jGfVgs3oZlciEWJas_YYPY/view?usp=sharing')) # Conventional
                       ])


### Check Occupation Dataset
occupation_dataset = pd.read_csv(read_link('https://drive.google.com/file/d/1GAURhgjxXMGdlFs2Gk4jaByoSPPPkdPy/view?usp=sharing'))

### Merge with occupation dataset
riasec_df = riasec_df.merge(occupation_dataset, left_on='O*NET-SOC Code', right_on='Code', how = 'left')

### Simplify dataset by taking only necessary columns
riasec_df.drop(riasec_df.columns[[0, 1, 2, 6, 7]], axis = 1, inplace = True)

### Drop duplicate values and reset index
riasec_df.drop_duplicates(inplace = True)
riasec_df = riasec_df.reset_index(drop = True)

### Rename the column names
riasec_df.rename(columns = {'First Interest Area':'First Personality', 
                           'Second Interest Area':'Second Personality',
                           'Third Interest Area':'Third Personality'}, inplace = True)

### Change NaN value into "-" 
riasec_df = riasec_df.fillna('-')

### Import libraries
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
import copy

### Tokenize all words within the dataset
# get all words on dataset as single token
all_text = []
for r in riasec_df.index:
    for c in riasec_df.columns:
        # only input in text type will be tokenized
        if type(riasec_df.loc[r,c]) != "-":
            text = nltk.word_tokenize(riasec_df.loc[r,c])
                        
            # for each tokenized-word list, append each word on all_text list
            for w in text:
                if w not in all_text:
                    all_text.append(w)

# remove non-alphabet character from user input, and set words to lowercase
all_text_ready = []
for wt in all_text:
    wt = re.sub(r'[^a-zA-Z]', '', wt)
    if wt != "":
        all_text_ready.append(wt.lower())

### STREAMLIT : Have user input for test
riasec = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
job_family = []
for i in riasec_df.index:
    family = riasec_df.loc[i,'Job Family']
    if family not in job_family:
        job_family.append(family)
      
### Get user input and Tokenize each word
user_personality = st.multiselect("Choose personalities:", riasec)
user_keywords = st.write("Input job/field or other keywords:")
user_input = user_personality + nltk.word_tokenize(user_keywords)

### Remove non alphabet from user input, and set words to lowercase
user_input_ready = []
for word in user_input:
    word = re.sub(r'[^a-zA-Z]', '', word)
    if word != "":
        user_input_ready.append(word.lower())
        
### Create empty list, to get words that already correct against all_text_ready
final_input = []

### Check the user input, if any word has already correct, put it on final input list
for word in user_input_ready:
    if word in all_text_ready:
        final_input.append(word)

### Check spelling
correct_words = words.words()
for word in user_input_ready:
    temp = [(jaccard_distance(set(ngrams(word, 2)),
                              set(ngrams(w, 2))),w)
            for w in correct_words if w[0].lower()==word[0].lower()]
    for match in sorted(temp):
        if match[1] in all_text_ready and match[1] not in final_input:
            final_input.append(match[1])

### Check spelling
for word in user_input_ready:
    temp = [(jaccard_distance(set(ngrams(word, 2)),
                              set(ngrams(w, 2))),w)
            for w in all_text_ready if w[0].lower()==word[0].lower()]
    for match in sorted(temp):
        if match[1] not in final_input:
            final_input.append(match[1])
            
### Check words based on their root
# create object
ps = PorterStemmer()
wnl = WordNetLemmatizer()

# create list to combine user input with recent final input
combined_input = user_input_ready + final_input

# create empty list to get all root words from combined input after rooting process
root_word = []

# word stemming & lemmatization check from combined input
for word in combined_input:
    # stemming
    root_word.append(ps.stem(word))
    # lemmatization
    if wnl.lemmatize(word) not in root_word:
        root_word.append(wnl.lemmatize(word))

# check words from all_text that has the same root
for text in all_text_ready:
    for root in root_word:
        if ps.stem(text) == root or wnl.lemmatize(text) == root:
            if text not in final_input:
                final_input.append(text)

### STREAMLIT Get output amount
amount = st.selectbox("Enter number of output:", [5, 10, 20, 50])

### Import libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### Replace - with blank string ("")
riasec_df_te = riasec_df.replace(['-'],'')

### change the input to be all lowercase and no character other than alphabet
def extract_string(string):
    new_string = ""
    for i, char in enumerate(string):
        if char.isalpha() == True or char == " " :
            new_string += string[i].lower()
    return new_string

for column in riasec_df_te.columns:
    for row in riasec_df_te.index:
        riasec_df_te.loc[row,column] = extract_string(riasec_df_te.loc[row,column])

### Create combined_features as combination of all string components
def combined_features(row):
    combined_column = []
    for i in riasec_df_te.columns:
        combined_column.append(riasec_df_te.loc[row,i])
    return " ".join(combined_column)

riasec_df_te['combined_features'] = [combined_features(i) for i in riasec_df_te.index]

### Define function to get the anchor row: first row occurence that is most matching with keywords
def get_index_from_keyword(keyword):
    word_match =[0,0] # first place will be the index, second place will be the count
    for i in riasec_df_te.index:
        count = 0
        for word in keyword:
            if word.lower() in riasec_df_te.loc[i, 'combined_features']:
                count +=1
        # program will find the highest count or row with most keyword-match amount, but only take the first occurence
        if count > word_match[1]:
            word_match[0], word_match[1] = i, count
    return word_match[0]

### Define function to get dataframe row from index
def get_row_from_index(index):
    return riasec_df.loc[index]

# Create table to display the anchor row based on keywords
# print("KEYWORDS MOSTLY MATCH WITH THIS OCCUPATION:")
match_row = pd.DataFrame(get_row_from_index(get_index_from_keyword(final_input))).T

### Create the model to whole dataset
cv = CountVectorizer()
cv_matrix = cv.fit_transform(riasec_df_te["combined_features"])
cv_cosine_sim = cosine_similarity(cv_matrix)

### Apply modelling to user input Using CountVectorizer Modelling
# Check cosine similarity value between keyword and all rows --> convert into list-of-tuple type
cv_rec_job = list(enumerate(cv_cosine_sim[get_index_from_keyword(final_input)]))

# Sort cosine similarity value from the highest
sorted_cv_rec_job = sorted(cv_rec_job, key = lambda x: x[1], reverse=True)

# take top-ten similar result
top_cv = sorted_cv_rec_job[:amount]

# show the result
cv_result = pd.DataFrame(columns = riasec_df.columns)
for i, value in enumerate(top_cv):
    cv_result.loc[i] = riasec_df.iloc[value[0], :]

# print("BELOW ARE SIMILAR OCCUPATIONS - BASED ON COUNT VECTORIZER MODELLING:")
# cv_result

### Create the model to whole dataframe
tv = TfidfVectorizer()
tv_matrix = tv.fit_transform(riasec_df_te["combined_features"])
tv_cosine_sim = cosine_similarity(tv_matrix)

### Apply modelling to user input : Using TfidfVectorizer Modelling
# Check cosine similarity value between keyword and all rows --> convert into list-of-tuple type
tv_rec_job = list(enumerate(tv_cosine_sim[get_index_from_keyword(final_input)]))

# Sort cosine similarity value from the highest
sorted_tv_rec_job = sorted(tv_rec_job, key = lambda x: x[1], reverse=True)

# take top-ten similar result
top_tv = sorted_cv_rec_job[:amount]

# show the result
tv_result = pd.DataFrame(columns = riasec_df.columns)
for i, value in enumerate(top_tv):
    tv_result.loc[i] = riasec_df.iloc[value[0], :]

# print("BELOW ARE SIMILAR OCCUPATIONS - BASED ON TFIDF VECTORIZER MODELLING:")
# tv_result

### Import libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors

### Create one-hot-encoding for RIASEC type
# recall variable
riasec = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']

# one-hot-encoding manually
ohc_riasec = pd.DataFrame()
for i in riasec_df.index:
    for value in riasec:
        for interest in range(3):
            if riasec_df.iloc[i,interest] == value:
                ohc_riasec.at[i,value]= 1
                
ohc_riasec = ohc_riasec.fillna(0)

### Create one-hot-encoding for job family
# create one-hot-encoding using Label Binarizer
ohc_job_family = LabelBinarizer().fit_transform(riasec_df['Job Family'])
ohc_job_family = pd.DataFrame(ohc_job_family, columns = sorted(riasec_df['Job Family'].unique()))

### Combine ohc_riasec and ohc_job_family into one dataframe
riasec_df_3 = ohc_riasec.join(ohc_job_family)

### Transform datas in array form so can be read by KNN modelling
# create two-dimentional array for whole dataset
riasec_df_array = []
for i in riasec_df_3.index:
    riasec_df_array.append(list(riasec_df_3.loc[i]))    
riasec_df_array = np.array(riasec_df_array)

# create two-dimentional array for keyword
knn_array = np.array(riasec_df_3.loc[get_index_from_keyword(final_input)]).reshape(1, -1)

### Apply modelling to user input : Using KNN Modelling
knn_riasec = NearestNeighbors(n_neighbors = amount, metric = 'cosine').fit(riasec_df_array)
distance, indices = knn_riasec.kneighbors(knn_array, n_neighbors = amount) 

# take top-ten similar result
top_knn = indices.tolist()[0][:]

# show the result
knn_result = pd.DataFrame(columns = riasec_df.columns)
for i, value in enumerate(top_knn):
    knn_result.loc[i] = riasec_df.iloc[value, :]

# print("BELOW ARE SIMILAR OCCUPATIONS - BASED ON KNN MODELLING:")
# knn_result
model_opt = st.radio("Choose modelling type:", ['Count Vectorizer', 'Tf-idf Vectorizer', 'K-Nearest Neighbor'], index = 0)

if st.button("Proceed") :
    if model_opt == 'Count Vectorizer':
        st.write(cv_result)
    elif model_opt == 'Tf-idf Vectorizer':
        st.write(tv_result)
    else:
        st.write(knn_result)
