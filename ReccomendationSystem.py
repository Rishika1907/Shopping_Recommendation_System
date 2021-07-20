# Importing the libraries
import tkinter as tk
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules






'''def Gui():
    def on_change(e):
    print( e.widget.get())
    
    window=tk.Tk()
    stateLabel = tk.Label(window,text="Enter your state")
    stateLabel.pack()
    stateEntry=tk.Text(window,height = 5,
                   width = 10)
    state = stateEntry.get(1.0, "end-1c") 
    stateEntry.pack()
    stateEntry.bind("<Return>", on_change)  
    
    cityLabel = tk.Label(window,text="Enter your city")
    cityLabel.pack()
    cityEntry = tk.Entry(window)
    city = cityEntry.get()
    cityEntry.pack()
    
    
    button = tk.Button(window,text ="Click here to get recommendations",command=lambda:(recommendation(state,city)))
    button.pack()
    
    window.geometry("500x200")
    window.mainloop()'''
    

def my_encode_units(x):
        if x<=0:
            return 0
        if x>=1:
            return 1
        
        
def cityBasedRecommendation(city,data_merge_df):
    
    #listing all the cities
    cities = list(data_merge_df['City'].unique())
    
    '''creating a basket for each state for analysis based on state if the state given 
    is present in the dataset'''
    
    if city in cities:
        #for each order in that state calculating the quantity of each item ordered
        mybasket=(data_merge_df[data_merge_df['City']==city]
                  .groupby(['Order ID','Sub-Category'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('Order ID')
                 )
        my_basket_sets=mybasket.applymap(my_encode_units)
    
        #frequent items with a minimum support value of 0.001
        #applying apriori algo to generate the frequent item sets
        my_frequent_itemsets =  apriori(my_basket_sets,min_support=0.001,use_colnames=True)
        
        #genarating the association rules with a minimum threshold value for the lift
        my_rules = association_rules(my_frequent_itemsets,metric="lift",min_threshold=5)
        
        #selecting the items that have lift greater than 3 and confidence greater than 0.7
        df=my_rules[(my_rules['lift'] >= 3)& (my_rules['confidence']>0.7)]
        
        #sorting the values based on descending order of lift
        df=df.sort_values('lift', ascending=False)
        df= df.head(5)
        consequents = list(df['consequents'].unique())
        print("\n Your recommendations based on your city: ")
        for x in consequents:
            print(list(x))
    else:
        print("\n Sorry we have no recommendations for you based on your city!")
        return
    
    
        
def stateBasedRecommendation(state,data_merge_df):
    print("State is",state)
    
    #listing all the states
    states = list(data_merge_df['State'].unique())
    
    
    '''creating a basket for each state for analysis based on state if the state given 
    is present in the dataset'''
    
    if state in states:
        #for each order in that state calculating the quantity of each item ordered
        mybasket=(data_merge_df[data_merge_df['State']==state]
                  .groupby(['Order ID','Sub-Category'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('Order ID')
                 )
        my_basket_sets=mybasket.applymap(my_encode_units)
    
        #frequent items with a minimum support value of 0.1
        #applying apriori algo to generate the frequent item sets
        my_frequent_itemsets =  apriori(my_basket_sets,min_support=0.01,use_colnames=True)
        
        #genarating the association rules with a minimum threshold value for the lift
        my_rules = association_rules(my_frequent_itemsets,metric="lift",min_threshold=3)
        
        #selecting the items that have lift greater than 3 and confidence greater than 0.7
        df=my_rules[(my_rules['lift'] >= 3)& (my_rules['confidence']>0.7)]
        
        #sorting the values based on descending order of lift
        df=df.sort_values('lift', ascending=False)
        df= df.head(5)
        consequents = list(df['consequents'].unique())
        print("\n Your recommendations based on your state: ")
        for x in consequents:
            print(list(x))
        #print([list(x) for x in consequents])
    else:
        print("\n Sorry we have no recommendations for you based on your state!")
        return
    
    
            

def recommendation(state,city):
    
    #Reading the datasets
    #list and orders
    lst = pd.read_csv('list.csv')
    orders = pd.read_csv('Order.csv')
    
    #merging the dataset based on order ID
    data_merge_df = orders.merge(lst,on = 'Order ID')
    
    #dropping order date attribute
    data_merge_df.drop('Order Date',axis = 'columns', inplace = True)
    data_merge_df.drop('CustomerName',axis = 'columns', inplace = True)
    
    #removing extra spaces present in state attribute
    data_merge_df['State'] = data_merge_df['State'].str.strip()
    
    #removing extra spaces present in state attribute
    data_merge_df['City'] = data_merge_df['City'].str.strip()
    
    #recommendation based on state
    stateBasedRecommendation(state,data_merge_df)
    
    #reccomendation based on city
    cityBasedRecommendation(city,data_merge_df)
    
    

#taking input from the user
state = input("Enter your state: ")
city = input("Enter your city: ")


recommendation(state,city)



