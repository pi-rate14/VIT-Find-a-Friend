# Library Imports
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import _pickle as pickle
from random import sample
from PIL import Image
from scipy.stats import halfnorm

# Loading the Profiles
with open("df.pkl",'rb') as fp:
    df = pickle.load(fp)

df.drop('ID',axis='columns', inplace=True)

with open("refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
with open("vectorized_refined.pkl", 'rb') as fp:
    vect_df = pickle.load(fp)
    
# Loading the Classification Model
model = load("refined_model.joblib")

## Helper Functions

def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x
 
    
def vectorization(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    column_name = columns[0]
        
    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'ID', 'Movie','TV', 'Music', 'Book', 'Sport', 'Vacation', 'People', 'Branch']:
        return df, input_df
    
    # Encoding columns with respective values
    if column_name in ['Religion', 'Politics']:
        
        # Getting labels for the original df
        df[column_name.lower()] = df[column_name].cat.codes
        
        # Dictionary for the codes
        d = dict(enumerate(df[column_name].cat.categories))
        
        d = {v: k for k, v in d.items()}
                
        # Getting labels for the input_df
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
                
        # Dropping the column names
        input_df = input_df.drop(column_name, 1)
        
        df = df.drop(column_name, 1)
        
        return vectorization(df, df.columns, input_df)
    
    # Vectorizing the other columns
    else:
          # Instantiating the Vectorizer
        vectorizer = CountVectorizer()
        
        # Fitting the vectorizer to the columns
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))
        
        y = vectorizer.transform(input_df[column_name].values.astype('U'))

        # Creating a new DF that contains the vectorized words
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
        
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names(), index=input_df.index)

        # Concating the words DF with the original DF
        new_df = pd.concat([df, df_wrds], axis=1)
        
        y_df = pd.concat([input_df, y_wrds], 1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)
        
        y_df = y_df.drop(column_name, 1)
        
        return vectorization(new_df, new_df.columns, y_df) 

    
def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()
    
    scaler.fit(df)
    
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
        
    return input_vect
    
def top_ten(cluster, vect_df, input_vect):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = vect_df[vect_df['Cluster #']==cluster[0]].drop('Cluster #', 1)
    
    # Appending the new profile data
    des_cluster = des_cluster.append(input_vect, sort=False)
        
    # Finding the Top 10 similar or correlated users to the new user
    user_n = input_vect.index[0]
    
    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the Top 10 most similar profiles
    top_10_sim = corr.sort_values(ascending=False)[1:11]
        
    # The Top Profiles
    top_10 = df.loc[top_10_sim.index]
        
    # Converting the floats to ints
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


def example_bios():
    """
    Creates a list of random example bios from the original dataset
    """
    # Example Bios for the user
    st.write("-"*100)
    st.text("Some example Bios:\n(Try to follow the same format)")
    for i in sample(list(df.index), 3):
        st.text(df['Bios'].loc[i])
    st.write("-"*100)


# Probability dictionary
p = {}

# Movie Genres
Movie = ['Action',
          'Animated',
          'Classics',
          'Comedy',
          'Drama',
          'Fantasy',
          'Horror',
          'Romance',
          'ScienceFiction'
          'Thriller']

p['Movie'] = [0.126,
               0.058,
               0.039,
               0.039,
               0.039,
               0.155,
               0.078,
               0.087, 
               0.165,
               0.214]

# TV Genres
TV = ['Anime',
      'Action',
      'Comedy',
      'Documentaries',
      'Drama',
      'Fantasy',
      'Horror',
      'Romance',
      'ScienceFiction'
      'Thriller']

p['TV'] = [0.242,
           0.001,
           0.223,
           0.019,
           0.078,
           0.107,
           0.029,
           0.001,
           0.078,
           0.214]

# Music (could potentially create a spectrum)
Music     = ['Pop',
            'ClassicRock',
            'IndieArtists',
            'NewRock',
            'HipHop',
            'ClassicVintage',
            'EDM',
            'MetalHeavyRock',
            'Classical',
            'Kpop']

p['Music'] =    [0.505,
                 0.049,
                 0.126,
                 0.029,
                 0.097,
                 0.049,
                 0.058,
                 0.029,
                 0.029,
                 0.029]

# Books
Book = ['Autobiographies',
         'Comics',
         'Encyclopedias',
         'Fantasy',
         'Horror',
         'Manga',
         'Romance',
         'SelfHelp',
         'ScienceFiction',
         'Thriller']

p['Book'] = [0.068,
              0.058,
              0.019,
              0.126,
              0.049,
              0.087,
              0.155,
              0.117,
              0.146,
              0.175]

# Sports
Sport = ['Badminton',
          'Basketball',
          'Cricket',
          'Esports',
          'Football',
          'Hockey',
          'Motorsports',
          'Tennis',
          'UFC_Wrestling',
          'Not_into_sports']

p['Sport'] = [0.262,
               0.175,
               0.155, 
               0.019,
               0.145,
               0.001,
               0.068,
               0.029,
               0.019,
               0.126]

# Vacation   (could also put on a spectrum)
Vacation = ['Trekking',
            'Chilling',
            'Partying',
            'Exploring',
            'Sailing',
            'Luxury',
            'Concert',
            'NightsOut',
            'Disneyland',
            'Skydiving']

p['Vacation'] = [0.58,
                 0.252,
                 0.039,
                 0.117,
                 0.019,
                 0.097,
                 0.029,
                 0.184,
                 0.049,
                 0.155]

# People
People = ['Festival',
          'Restaurant',
          'Friends',
          'Club',
          'Zoom',
          'SocialMedia',
          'Activity',
          'Travelling',
          'No_preference',
          'Dont_like_meeting']

p['People'] =       [0.058,
                     0.029,
                     0.301,
                     0.029,
                     0.001,
                     0.029,
                     0.184,
                     0.068,
                     0.233,
                     0.058]

# branch
Branch = ['Agriculture',
          'Automotive',
          'Biotechnology',
          'Chemical',
          'Civil',
          'CS_IT',
          'ECE_Communication_Intrumentation',
          'Hotel_management',
          'Industrial_Architectural',
          'Mechanical']

p['Branch'] =       [0.010,
                     0.010,
                     0.010,
                     0.010,
                     0.010,
                     0.835,
                     0.097,
                     0.019,
                     0.001,
                     0.001]

# Age (generating random numbers based on half normal distribution)
# age = halfnorm.rvs(loc=18,scale=8, size=df.shape[0]).astype(int)

# Lists of Names and the list of the lists
categories = [Movie, TV, Music, Book, Sport, Vacation, People, Branch]

names = ['Movie', 'TV', 'Music', 'Book', 'Sport', 'Vacation', 'People', 'Branch']

combined = dict(zip(names, categories))


## Interactive Section

# Creating the Titles and Image
st.title("AI-MatchMaker")

st.header("Finding a Date with Artificial Intelligence")
st.write("Using Machine Learning to Find the Top Dating Profiles for you")

image = Image.open('robot_matchmaker.jpg')

st.image(image, use_column_width=True)

# Instantiating a new DF row to classify later
new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

# Asking for new profile data
new_profile['Bios'] = st.text_input("Enter a Bio for yourself: ")

# Printing out some example bios for the user        
example_bios()

# Checking if the user wants random bios instead
random_vals = st.checkbox("Check here if you would like random values for yourself instead")

# Entering values for the user
# if random_vals:
#     # Adding random values for new data
#     for i in new_profile.columns[1:]:
#         if i in ['Religion', 'Politics']:  
#             new_profile[i] = np.random.choice(combined[i], 1, p=p[i])
            
#         elif i == 'Age':
#             new_profile[i] = halfnorm.rvs(loc=18,scale=8, size=1).astype(int)
            
#         else:
#             new_profile[i] = list(np.random.choice(combined[i], size=(1,3), p=p[i]))
            
#             new_profile[i] = new_profile[i].apply(lambda x: list(set(x.tolist())))

# else:
#     # Manually inputting the data
#     for i in new_profile.columns[1:]:
#         if i in ['Religion', 'Politics']:  
#             new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])
            
#         elif i == 'Age':
#             new_profile[i] = st.slider("What is your age?", 18, 50)
            
#         else:
#             options = st.multiselect(f"What is your preferred choice for {i}?", combined[i])
            
#             # Assigning the list to a specific row
#             new_profile.at[new_profile.index[0], i] = options
            
#             new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))
            
        
for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])
            
        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)
            
        else:
            options = st.multiselect(f"What is your preferred choice for {i}?", combined[i])
            
            # Assigning the list to a specific row
            new_profile.at[new_profile.index[0], i] = options
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))

# Looping through the columns and applying the string_convert() function (for vectorization purposes)
for col in df.columns:
    df[col] = df[col].apply(string_convert)
    
    new_profile[col] = new_profile[col].apply(string_convert)
            

# Displaying the User's Profile 
st.write("-"*1000)
st.write("Your profile:")
st.table(new_profile)

# Push to start the matchmaking process
button = st.button("Click to find your Top 10!")

if button:    
    with st.spinner('Finding your Top 10 Matches...'):
        # Vectorizing the New Data
        df_v, input_df = vectorization(df, df.columns, new_profile)
        print(df_v)
        print(input_df)                
        # Scaling the New Data
        new_df = scaling(df_v, input_df)
                
        # Predicting/Classifying the new data
        cluster = model.predict(new_df)
        
        # Finding the top 10 related profiles
        top_10_df = top_ten(cluster, vect_df, new_df)
        
        # Success message   
        st.success("Found your Top 10 Most Similar Profiles!")    
        st.balloons()

        # Displaying the Top 10 similar profiles
        st.table(top_10_df)
        

        

    

