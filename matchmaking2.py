# Library Imports
from operator import mod
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


def vectorization(new_profile):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    # Vectorizing the data
    # Assigning the split variables
    X = cluster_df.drop(["Cluster #"], 1)
    y = cluster_df['Cluster #']

    ## Vectorizing
    # Instantiating the Vectorizer
    vectorizer = CountVectorizer()

    # Fitting the vectorizer to the Bios
    x = vectorizer.fit_transform(X['Bios'])

    # Creating a new DF that contains the vectorized words
    df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

    # Concating the words DF with the original DF
    X = pd.concat([X, df_wrds], axis=1)

    # Dropping the Bios because it is no longer needed in place of vectorization
    X.drop(['Bios'], axis=1, inplace=True)
    # X.drop(['ID'], axis=1, inplace=True)
        # Scaling the Data
    scaler = MinMaxScaler()

    X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        # Vectorizing the new data
    vect_new_prof = vectorizer.transform(new_profile['Bios'])

    # Quick DF of the vectorized words
    new_vect_w = pd.DataFrame(vect_new_prof.toarray(), columns=vectorizer.get_feature_names(), index=new_profile.index)

    # Concatenating the DFs for the new profile data
    new_vect_prof = pd.concat([new_profile, new_vect_w], 1).drop('Bios', 1)

    # Scaling the new profile data
    new_vect_prof = pd.DataFrame(scaler.transform(new_vect_prof), columns=new_vect_prof.columns, index=new_vect_prof.index)

    return new_vect_prof

    
# def scaling(df, input_df):
#     """
#     Scales the new data with the scaler fitted from the previous data
#     """
#     scaler = MinMaxScaler()
    
#     scaler.fit(df)
    
#     input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
        
#     return input_vect
    


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

    print(top_10)

    print(top_10.index[0])
    
    with open("df.pkl",'rb') as fp:
        res = pickle.load(fp)

    results = res.iloc[[top_10.index[0]]]

    print(res.iloc[top_10.index[0]])

    return results.astype('object')



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
          'ScienceFiction',
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
      'ScienceFiction',
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


Dict = {'Movie': {"Action": 0,"Animated": 1,"Classics": 2,"Comedy": 3,"Drama": 4,"Fantasy": 5,"Horror": 6,"Romance": 7, "ScienceFiction": 8, "Thriller":9},
        'TV': {"Anime": 0,"Action": 1,"Comedy": 2,"Documentaries": 3,"Drama": 4,"Fantasy": 5,"Horror": 6,"Romance": 7, "ScienceFiction": 8, "Thriller":9},
        'Music': {"Pop": 0,"ClassicRock": 1,"IndieArtists": 2,"NewRock": 3,"HipHop": 4,"ClassicVintage": 5,"EDM": 6,"MetalHeavyRock": 7, "Classical": 8, "Kpop":9},
        'Book': {"Autobiographies": 0,"Comics": 1,"Encyclopedias": 2,"Fantasy": 3,"Horror": 4,"Manga": 5,"Romance": 6,"SelfHelp": 7, "ScienceFiction": 8, "Thriller":9},
        'Sport': {"Badminton": 0,"Basketball": 1,"Cricket": 2,"Esports": 3,"Football": 4,"Hockey": 5,"Motorsports": 6,"Tennis": 7, "UFC_Wrestling": 8, "Not_into_sports":9},
        'Vacation': {"Trekking": 0,"Chilling": 1,"Partying": 2,"Exploring": 3,"Sailing": 4,"Luxury": 5,"Concert": 6,"NightsOut": 7, "DisneyLand": 8, "Skydiving":9},
        'People': {"Festival": 0,"Restaurant": 1,"Friends": 2,"Club": 3,"Zoom": 4,"SocialMedia": 5,"Activity": 6,"Travelling": 7, "No_preference": 8, "Dont_like_meeting":9},
        'Branch': {"Agriculture": 0,"Automotive": 1,"Biotechnology": 2,"Chemical": 3,"Civil": 4,"CS_IT": 5,"ECE_Communication_Intrumentation": 6,"Hotel_management": 7, "Industrial_Architectural": 8, "Mechanical":9},
}
        


 ## Interactive Section

# Creating the Titles and Image
st.title("AI-Find a Friend")

st.header("Finding a Friend in VIT with Artificial Intelligence")
st.write("Using Machine Learning to Find the top match for you!")

image = Image.open('VITU.jpg')

st.image(image, use_column_width=True)

# Instantiating a new DF row to classify later
new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])

# Asking for new profile data
new_profile['Bios'] = st.text_input("Enter a Bio for yourself: ")

# Printing out some example bios for the user        
example_bios()
     
        
for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])
            
        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)
            
        else:
            options = st.selectbox(f"What is your preferred choice for {i}?", combined[i])
            
            chosen_option = Dict[i][options]
            print(chosen_option)
            # Assigning the list to a specific row
            new_profile.at[new_profile.index[0], i] = chosen_option
            
            # new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))

# Looping through the columns and applying the string_convert() function (for vectorization purposes)
for col in df.columns:
    df[col] = df[col].apply(string_convert)
    
    new_profile[col] = new_profile[col].apply(string_convert)
            

# Displaying the User's Profile 
st.write("-"*1000)
st.write("Your profile:")
st.table(new_profile)

# Push to start the matchmaking process
button = st.button("Click to find your Top Match!")

if button:    
    with st.spinner('Finding your Top Match...'):
        # Vectorizing the New Data
        # df_v, input_df = vectorization(df, df.columns, new_profile)
        # print(df_v)
        # print(input_df)                
        # # Scaling the New Data
        # new_df = scaling(df_v, input_df)

        new_df = vectorization(new_profile)

        print(new_df)
                
        # Predicting/Classifying the new data
        cluster = model.predict(new_df)
        
        # Finding the top 10 related profiles
        top_10_df = top_ten(cluster, vect_df, new_df)
        
        # Success message   
        st.success("Found your Top Match!")    
        st.balloons()

        # Displaying the Top 10 similar profiles
        st.table(top_10_df)
        regNo = top_10_df.iloc[0]['ID']
        link = 'https://teams.microsoft.com/_#/apps/a2da8768-95d5-419e-9441-3b539865b118/search?q={}'.format(regNo)
        st.write("Chat with your top match here: ")
        st.write(link)
        

        

    

