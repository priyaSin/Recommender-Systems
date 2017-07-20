
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd

#Class for Popularity Based Recommender System
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popular_recommendations = None
    
    
    #Creating a popularity based recommender system model or training your model
    def create(self , train_data , user_id , item_id):
        
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        
        #Counting the number of users for each unique song and storing that as a score
        grouped_train_data = train_data.groupby(self.item_id).agg({self.user_id : 'count'}).reset_index()
        grouped_train_data.rename(columns = {'user_id' : 'score'} , inplace = True) 
        #inplace = True means all the changes will take place in teh existing data frame, not in copy made
        
        
        #sorting the dataframe based on score(songs heard by most of the users) in descending order
        train_data_sort = grouped_train_data.sort_values(['score' , self.item_id] , ascending = False)
        
        #Giving a rank to each song in the 'song' column in the new 'rank' column
        train_data_sort["Rank"] = train_data_sort["score"].rank(ascending = False , method = 'first')
        
        #Extracting the top 10 recommendations and saving it in popular_recommedations for recommending any new user
        self.popularity_recommendations = train_data_sort.head(10)
    
    #Creating a recommender which recommender music to a new user
    def recommend(self , user_id):
        user_recommendations = self.popularity_recommendations
        
        #Adding a new column "user_id" with values= user_id of the new user
        user_recommendations["user_id"] = user_id
        
        #Bringing the user_id column up to the front
        columns = user_recommendations.columns.tolist()  #Making a list of columns
        columns = columns [-1:] + columns[:-1]   #Arranging the columns, bringing user_id columsn to the front
        user_recommendations = user_recommendations[columns]   #making a new copy of dataframe with arranged columns
        
        return user_recommendations
             


# In[6]:

#Class for Item-similarity Based Recommendar system
class item_similarity_recommender_py():
    
    def __init__ (self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    
    #Get list of unique songs corresponding to a given user
    def get_user_items(self , user):
        user_data = self.train_data[self.train_data[self.user_id] == user] #Extracting all the data for a given user
        user_items = list(user_data[self.item_id].unique()) #Extracting a list of unique songs listened by the user
        
        return user_items
    
    
    #Get list of unique users for a given item(songs)
    def get_item_users(self , item):
        item_data = self.train_data[self.train_data[self.item_id] == item ] #Extracting all the data for a given song
        item_users = set(item_data[self.user_id].unique()) #Extracting all the unique users
        
        return item_users
    
    
    #Get unique items(songs) in the training data
    def get_all_unique_items(self):
        all_items = list(self.train_data[self.item_id].unique()) #getting a list of songs
        
        return all_items
    
    
    #Contructing the co-occurence Matrix
    def construct_cooccurence_matrix(self , user_songs , all_songs):
        
        #user_songs is the list of all the songs already listened by our test_user
        #all_songs is the list of all unique songs
        
        #Get users for all the songs of user_songs
        user_songs_users =[]
        for i in range(len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            # a list with number of rows = len(user_songs) or number of songs user has heard and 
            #for each song we will have a list of users who have heard that song
            
            
        #Initialize the item coccurence matrix of the size
        #len(user_songs) X len(all_songs)
        cooccurence_matrix = np.matrix(np.zeros(shape = (len(user_songs) , len(all_songs))) , float)
        
        
        #Calculate similarity between user_songs and all unique songs in the training data
        for i in range(len(all_songs)):
            #Calculate unique listensers(users) of the song (i)
            song_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(song_i_data[self.user_id].unique())
            
            for j in range(len(user_songs)):
                #Get unique listeners(users) of the item j (song j)
                users_j = user_songs_users[j] 
                
                #Calculate the intersection of listeners of songs i and j 
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jacard Index
                if len(users_intersection) !=0:
                    #Calculate union of the listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    #Filling the values in the cooccurence_matrix according to Jacard Normalization
                    cooccurence_matrix[j,i] = float(len(users_intersection)) /float (len(users_union))
                    
                else:
                    cooccurence_matrix[j,i] =0
                    
        return cooccurence_matrix

    
    #Using the Co-occurence matrix to make top recommendations
    def gen_top_recommendations(self , user , cooccurence_matrix , all_songs , user_songs):
        
        print("Non-zero values in the co-occurence matrix : %d" %np.count_nonzero(cooccurence_matrix))
        
        
        #Calculate weighted_average of the scores in cooccurence matrix for all user_songs
        user_similar_score = cooccurence_matrix.sum(axis = 0) / float(cooccurence_matrix.shape[0]) 
        #The above will return a list of shape(1,len(all_songs))
        user_similar_score = np.array(user_similar_score)[0].tolist() 
        #Converting the list to numpy array(2d) taking values all rows and putting them into one list
        
        #Sort the indices along with values of user_similar_score based upon their values(scores)
        #in the reverse order or descending order
        sorted_index_score = sorted(((score , index) for index, score in enumerate(list(user_similar_score))) , reverse = True)
        
        
        #Create a dataframe for the current user with recommendations of the songs
        #Dataframe will have following column names
        columns = ['user_id' , 'song' , 'score' , 'rank']
        df = pd.DataFrame(columns = columns)
        
        
        #Filling the dataframe with top 10 song recommendations
        rank = 1
        for i in range(len(sorted_index_score)):
            if ~np.isnan(sorted_index_score[i][0]) and all_songs[sorted_index_score[i][1]] not in user_songs and rank<=10:
                #Above means, if score column of sorted_index_score is not null AND
                #index col gives a song which is not in user_songs AND
                #ofcourse rank remains less than 10 then take that song and put it in dataframe
                df.loc[len(df)] = [user , all_songs[sorted_index_score[i][1]] , sorted_index_score[i][0] , rank]
                #filling a row in df accoring to the columns assinged, 'user' is passed in the function
                #it is the user for whom recommendations are being made
                rank +=1
            if rank>10:
                break
        
        
        #For the cases when there is no recommendations, means df is empty
        #(in case when the user is new and has not yeat heard any song)
        if df.shape[0] ==0: #if the number of rows in a dataframe =0
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
        
        
    #Create the item similarity based recommender system model
    def create(self , train_data , user_id , item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        
    #Using the item similarity based recommender system model to make recommendations
    def recommend(self , user):
        
        # 1.Getting all the unique_songs he/she hears (user)
        user_songs = self.get_user_items(user)
        print("Number of unique songs for the user: %d" %len(user_songs))
        
        # 2. Getting all the unique songs present in the dataframe
        all_songs = self.get_all_unique_items()
        print("Number of unique songs in the training set: %d" %len(all_songs))
        
        # 3. Construct item cooccurence matrix of size len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs , all_songs)
        
        # 4. Use the cooccurence_matrix to make recommedations
        df_recommendations = self.gen_top_recommendations(user , cooccurence_matrix , all_songs , user_songs)
        
        return df_recommendations
    
    #Getting similar items to the given item
    def get_similar_items(self , item_list):
        
        #REPEATING ALL THE FOUR STEPS AGAIN
        
        user_songs = item_list
        
        all_songs = self.get_all_unique_items()
        print("Number of unique songs in the training set: %d" %len(all_songs))
        
        
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs , all_songs)
        
        user = " " #as we are trying to find songs similar to given song, there is no need of the user_column
        
        df_recommendations = self.gen_top_recommendations(user , cooccurence_matrix , all_songs , user_songs)
        
        return df_recommendations      


# In[ ]:



