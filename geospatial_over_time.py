# Final Project Part 1: The Data
# Use this IDE for importing and visualizing Data Source #1


import pandas as pd
import geopandas as gpd
import plotly.express as px


def load_data_world(mental_health_disorder):
    df_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    filename = './data/mental_health.csv'
    data = pd.read_csv(filename, index_col=0)
    data = data[data['Year'].notnull() & data[mental_health_disorder].notnull()]
    data = data[data['Code'].notnull()]
    countries = data
    countries = countries.rename(columns={'Code': 'iso_a3'})
    countries_depression = countries[['iso_a3', mental_health_disorder, 'Year']]
    df_world_total = pd.merge(df_world, countries_depression, on="iso_a3", how="left")
    print("Type of DataFrame : ", type(df_world_total), df_world_total.shape[0])
    print(df_world_total)
    plot_data_world(df_world_total, mental_health_disorder)

def plot_data_world(df_world_total, mental_health_disorder):
    title = "Global " + mental_health_disorder + " Over Time"
    fig = px.choropleth(df_world_total, 
                     locations="iso_a3", 
                     color=mental_health_disorder, 
                     hover_name="name", 
                     animation_frame="Year",  
                     color_continuous_scale='OrRd',
                     range_color=(0, df_world_total[mental_health_disorder].max()), 
                     projection="natural earth", 
                     title=title
                     )
    fig.update_geos(visible=False, resolution=110)
    fig.show()
