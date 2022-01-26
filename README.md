# JOB RECOMMENDATION BASED ON RIASEC PERSONALITY

## INTRODUCTION
In this program we will build job recommendation engine based on RIASEC personality. <br>
RIASEC itself stands for Realistic - Investigative - Artistic - Social - Enterprising - Conventional. <br>
RIASEC Personality Test is commonly used to understand someone's skill and career interest. 

## DATASET
We will use several datasets:<br>
* Taken from https://www.onetonline.org/explore/interests/, we can get 6 separate datasets for each RIASEC personality, and related job options matched to personality
* Taken from https://www.onetonline.org/find/family?f=0&g=Go, we can get occupation dataset for all job listed in the website.

## DATA ANALYSIS
* Beside general overview, we will check the top-5 job family for each personality, to get understanding of each personality characteristic;
* And we will check how each personality is related one to another

## MODELLING APPROACH
* As we will receive input from user as text, first we will do Text Analysis to get the keywords that match with the dataset
* For recommendation engine, we will use 3 types of modelling and all are from scikit.learn library:
  1. Text Extraction: CountVectorizer
  2. Text Extraction: TfidfVectorizer
  3. K-Nearest Neighbor
* To check similarity between text, we use cosine similarity metric

## EXPECTED PROGRAM WORKFLOW
* User will be able to input the keywords
* Program will get the anchor row, or the first row occurence that match keywords the most
* Program will recommend top ten suggestion similar to the input result (1 anchor row + 9 similar to anchor)

## TECHNICAL
Language: Python

Libraries:
* BASIC : numpy, pandas
* EDA : matplotlib.pyplot, seaborn, collection.Counter
* TEXT ANALYZER : copy, re, nltk : punkt, wordnet, omw-1.4, stem.PorterStemmer, stem.WordNetLemmatizer, pyspellchecker.SpellChecker
* MACHINE LEARNING: 
  - sklearn.feature_extraction.text : CountVectorizer, TfidfVectorizer
  - sklearn.metrics.pairwise.cosine_similarity
  - sklearn.preprocessing.LabelBinarizer
  - sklearn.neighbors.NearestNeighbors

## CONSTRAINS
* Dataset is limited -- many recent jobs may not be included in the data yet.
* Dataset is imbalance -- one personality/job family is much more than another.
* Text Analyzer for user input still has limited process. If the input has too much typo, program may not detect properly.

## FUTURE DEVELOPMENT FOR BUSINESS RECOMMENDATION
1. Job recommendation engine based on personality can be very useful for people, to find the balance between their personality and their interest or personality.
2. Recommendation result can be linked with the job application sites, so it does not only show the occupation title, but also can give future employee some options of related applications.
3. From employer side, it will also be useful to see whether the employee personality will match to the job or not.
