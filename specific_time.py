# Final Project Part 1: The Data
# Use this IDE for importing and visualizing Data Source #1


import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import seaborn as sns

def load_data_specific_time(statistic_to_plot, year):
    df_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    filename = './data/mental_health.csv'
    data = pd.read_csv(filename, index_col=0)
    data = data[data['Year'].notnull()]
    data = data[data['Year'] == str(year)]
    data = data[data['Code'].notnull()]
    countries = data
    countries = countries.rename(columns={'Code': 'iso_a3'})
    df_world_total = pd.merge(df_world, countries, on="iso_a3", how="left")
   


    # Load economic data
    economic_data_raw = pd.read_csv('./data/life_expectancy.csv', index_col=0)
    economic_data = economic_data_raw[economic_data_raw['Year'] == year]
    economic_data = economic_data.rename(columns={'Country Code': 'iso_a3'})
 
    df_world_total = pd.merge(df_world_total, economic_data, on="iso_a3", how="left")
    

    # Load gini_index
    gini_index_raw = pd.read_csv('./data/gini_index.csv')
    mask = gini_index_raw['year'] == year
    gini_index = gini_index_raw[mask]
    gini_index = gini_index.rename(columns={'country': 'name'})
    gini_index['name'] = gini_index['name'].replace({'United States': 'United States of America'})
    df_world_total = pd.merge(df_world_total, gini_index, on="name", how="left")


    # Load GDP data
    gdp_raw = pd.read_csv('./data/gdp_1960_2020.csv')
    gdp = gdp_raw[gdp_raw['year'] == year]
    gdp = gdp.rename(columns={'country': 'name'})
    gdp['name'] = gdp['name'].replace({'the United States': 'United States of America'})
    df_world_total = pd.merge(df_world_total, gdp, on="name", how="left")

    
    plot_data_specific_time(df_world_total, statistic_to_plot, year)
    return df_world_total

def plot_data_specific_time(df_world_total, statistic_to_plot, year):
    title = "Global " + statistic_to_plot + " during " + str(year)
    fig = px.choropleth(df_world_total, 
                     locations="iso_a3", 
                     color=statistic_to_plot, 
                     hover_name="name", 

                     color_continuous_scale='OrRd',
                     range_color=(0, df_world_total[statistic_to_plot].max()), 
                     projection="natural earth", 
                     title=title
                     )
    fig.update_geos(visible=False, resolution=110)
    fig.show()

def create_df_for_model(df):
    df.dropna(subset=['Health Expenditure %', 'Education Expenditure %', 'gini_index', 'gdp'], inplace=True)
    final_df = df[['Health Expenditure %', 'Education Expenditure %', 'gini_index', 'gdp', 'Depression (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)']]
    return final_df


def create_and_train_model(mental_health_disorder, final_df):
    df_mental_health_disorder = final_df.dropna(subset=[mental_health_disorder])
    df_mental_health_disorder_only = df_mental_health_disorder[[mental_health_disorder]]
    df_input = df_mental_health_disorder[['Health Expenditure %', 'Education Expenditure %', 'gini_index', 'gdp']]

    X_train, X_test, y_train, y_test = train_test_split(df_input, df_mental_health_disorder_only, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_features = 4
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
    # Evaluate model
    loss = model.evaluate(X_test_scaled, y_test)
    print("Test Loss:", loss)
    plot_model_statistics(history, model, X_test_scaled, y_test, mental_health_disorder, X_train_scaled, X_train, y_train)

def plot_model_statistics(history, model, X_test_scaled, y_test, mental_health_disorder, X_train_scaled, X_train, y_train):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/loss_function.png')
    plt.show()

    y_pred = model.predict(X_test_scaled).flatten()

    # Plot actual vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([0, 5], [0, 5], color='red', linestyle='--') 
    title = 'Actual vs. Predicted ' + mental_health_disorder
    plt.title(title)
    xlabel = 'Actual ' + mental_health_disorder
    ylabel = 'Predicted ' + mental_health_disorder
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.grid(True)
    plt.savefig('plots/actual_vs_predicted.png')
    plt.show()

    df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    df[mental_health_disorder] = y_train

# Correlation matrix heatmap
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.show()

