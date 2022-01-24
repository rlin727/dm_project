import pandas as pd
import streamlit as st
import pydeck as pdk
import random
import numpy as np
import pickle
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial

st.set_page_config(layout="wide")

data = pd.read_csv("BOS_CLEANED_CLUSTERED.csv", index_col=0)
reg_data = pd.read_csv("reg_data.csv")

cls = pickle.load(open("cls_bos.pkl", "rb"))
reg = pickle.load(open("reg_bos.pkl", 'rb'))

def get_comp(target_cate_list,cluster):
    count = 0
    cur_cluster = data[(data['cluster'] == cluster)]
    for index,row in cur_cluster.iterrows():
        row['categories'] = row['categories'].replace('[','')
        row['categories'] = row['categories'].replace(']','')
        row['categories'] = row['categories'].replace('\'','')
        cate_list = row['categories'].split(", ")
        count += len(set(target_cate_list) & set(cate_list))
        #print(cate_list)
    return (count/len(cur_cluster))

def get_similarity(attr_list, cluster):
    cos_similarity = {}
    cur_cluster = data[(data['cluster'] == cluster)]
    attr = cur_cluster.iloc[:, 7:45]
    count = 0
    for index, row1 in attr.iterrows():
        attr_list2 = []
        for col,row2 in attr.iteritems():
            attr_list2.append(row1[col])
        count += 1 - spatial.distance.cosine(attr_list, attr_list2)
    return (count/len(cur_cluster))

clusters = data['cluster'].unique()




reg_data['cluster1']=reg_data['cluster'].apply(lambda x:x.split("_")[0])


def get_pred(comp,attr,cluster1):
    pred = [] 
    res ={}
    clusters2 = reg_data[reg_data['cluster1'] == cluster1]['cluster']
    for cluster2 in clusters2:
        curr_comp = get_comp(comp,cluster2)
        curr_simi = get_similarity(attr,cluster2)
        input_data = [reg_data[reg_data['cluster'] == cluster2]['density'].iloc[0]]

        input_data.append(reg_data[reg_data['cluster'] == cluster2]['rating'].iloc[0])
        input_data.append(reg_data[reg_data['cluster'] == cluster2]['review'].iloc[0])
        input_data.append(curr_comp)
        input_data.append(curr_simi)
        for cluster in clusters:
            curr = np.append(input_data,cluster)
            yhat = reg.predict([curr])
            pred.append([yhat,cluster])
    pred.sort()

    temp = []
    for i in pred[-6:-1]:
        temp.append(i[1])
    return temp






random.seed(7777)
colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255]for i in range(40)]

st.title("Where your next Business should be?")

attrs = {}
cols = st.columns(4)
with cols[0]:
    city = st.selectbox("City", options=["Boston"])

    attrs["stars"] = st.slider("Rating", min_value = 0.5, max_value =5.0, step=0.1, format=("%f\U00002B50"), value=3.0)

    options = ["$", "$"*2, "$"*3, "$"*4]
    price = len(st.select_slider("Price Range", options=options, value="$$"))

    options = ["Nightlife", 'Bars', 'Sandwiches', 'American (New)', 'American (Traditional)', 'Italian', 'Breakfast & Brunch', 'Coffee & Tea', 'Pizza',
               'Seafood', 'Chinese', 'Fast Food', 'Mexican', 'Japanese', 'Bakeries', 'Delis', 'Desserts', 'Mediterranean']
    categories = ["Restaurants"]
    categories += st.multiselect("Categories", options=options)

with cols[1]:
    attrs["RestaurantsTableService"] = int(st.checkbox("Restaurants Table Service", False))
    attrs["RestaurantsReservations"] = int(st.checkbox("Restaurants Reservations", False))
    attrs["WheelchairAccessible"] = int(st.checkbox("Wheelchair Accessible", False))
    attrs["RestaurantsPriceRange2"] = price
    attrs["HasTV"] = int(st.checkbox("Has TV", False))
    attrs["RestaurantsTakeOut"] = int(st.checkbox("Take Out", False))
    attrs["RestaurantsDelivery"] = int(st.checkbox("Delivery", False))
    attrs["GoodForKids"] = int(st.checkbox("Good For Kids", False))
    attrs["CoatCheck"] = int(st.checkbox("Coat Check", False))
with cols[2]:
    options=['romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual']
    temp = st.multiselect("Environment", options=options)
    for x in options:
        attrs[x]=int(x in temp)

    options=['dj', 'background_music', 'no_music', 'jukebox', 'live', 'video', 'karaoke']
    temp = st.multiselect("Music", options=options)
    for x in options:
        attrs[x]=int(x in temp)

    options=['free', 'no', 'paid']
    temp = st.selectbox("WiFi", options=options)
    for x in options:
        attrs["WiFi_"+x]=int(x in temp)

with cols[3]:
    options=['full_bar', 'none', 'beer_and_wine']
    temp = st.selectbox("Alcohol", options=options)
    for x in options:
        attrs["Alcohol_"+x]=int(x in temp)
    
    options=['average', 'quiet', 'loud', 'very loud',]
    temp = st.selectbox("Noise Level", options=options)
    for x in options:
        attrs["Noise_Level_"+x]=int(x in temp)
    
    options=['casual', 'dressy', 'formal']
    temp = st.multiselect("Attire", options=options)
    for x in options:
        attrs["RestaurantsAttire_"+x]=int(x in temp)
    
temp = np.array(list(attrs.values())).reshape(1, -1)
label = cls.predict(temp)
temp = np.array(list(attrs.values())[1:]).reshape(1, -1)
res = get_pred(categories, temp, str(label[0]))





temp = []
clu1 = []
clu2 = []
for l in res[:-1]:
    temp.append(data[data["cluster"]==l])

res_layers =[]

res_layers=[pdk.Layer( 
                "ScatterplotLayer",
                data=data[data["cluster"]==res[-1]],
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                get_radius=300,          
                get_fill_color=[0, 30, 200, 160],
            ),
            pdk.Layer( 
                "ScatterplotLayer",
                data=pd.concat(temp),
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                get_radius=300,          
                get_fill_color=[200, 30, 0, 160],
            )]  

cols = st.columns(2)
with cols[1]:
    st.subheader("Recommended locations for your Business!")
    st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",
                initial_view_state={"latitude": 42.32,
                                    "longitude": -71.05, "zoom": 10.5},
                layers=res_layers,
                
            ),use_container_width=True)


all_layers =[]

for i in range(data["cluster1"].nunique()-1):
    all_layers.append(
        pdk.Layer( 
            "ScatterplotLayer",
            data=data[data["cluster1"]==i],
            get_position=["longitude", "latitude"],
            auto_highlight=True,
            get_radius=70,          
            get_fill_color=colors[i],
        )
    )

with cols[0]:
    st.subheader(f"All restaurants at {city}")
    st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",
                initial_view_state={"latitude": 42.32,
                                    "longitude": -71.05, "zoom": 10.5},
                layers=all_layers,
                
            ),use_container_width=True)