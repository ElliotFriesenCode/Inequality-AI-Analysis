# Final Project Part 1: The Data
# Use this IDE for importing and visualizing Data Source #1

import pandas as pd
import matplotlib.pyplot as plt


def load_data_for_country(chosenCountry, disorder):
    mental_health_raw = pd.read_csv('./data/mental_health.csv')
    gdp_raw = pd.read_csv('./data/gdp_1960_2020.csv')
    economic_raw = pd.read_csv('./data/life_expectancy.csv')
    economic_raw['Year'] = pd.to_numeric(economic_raw['Year'], errors='coerce')
    economic_stats = economic_raw.rename(columns={'Country Name': 'Entity'})
    gdp = gdp_raw.rename(columns={'country': 'Entity', 'year': 'Year'})
    print(gdp)
    mental_health  = mental_health_raw[mental_health_raw['Year'].notnull()]
    print(mental_health_raw['Year'].unique())
    mental_health['Year'] = pd.to_numeric(mental_health['Year'], errors='coerce')
    mental_health = mental_health[(mental_health['Year'] >= 2001) & (mental_health['Year'] <= 2017)]
    gdp['Year'] = pd.to_numeric(gdp['Year'], errors='coerce')
    gdp['Entity'] = gdp['Entity'].replace({'the United States': 'United States'})
    mental_health = mental_health.rename(columns={'Code': 'iso_a3'})
    print(mental_health)
    countries_gdp_and_mental_health = pd.merge(mental_health, gdp, on=["Entity", 'Year'], how="left")
    countries_gdp_and_mental_health = pd.merge(countries_gdp_and_mental_health, economic_stats, on=["Entity", 'Year'], how="left")
    print(countries_gdp_and_mental_health)
    return handle_data_for_country(countries_gdp_and_mental_health, chosenCountry, disorder)

def handle_data_for_country(countries_gdp_and_mental_health, country, disorder):
    country_data = countries_gdp_and_mental_health[countries_gdp_and_mental_health['Entity'] == country] # set country
    country_data[disorder + " normalized"] = (country_data[disorder] - country_data[disorder].min()) / (country_data[disorder].max() - country_data[disorder].min())
    country_data['gdp normalized'] = (country_data['gdp'] - country_data['gdp'].min()) / (country_data['gdp'].max() - country_data['gdp'].min())
    country_data['health percent normalized'] = (country_data['Health Expenditure %'] - country_data['Health Expenditure %'].min()) / (country_data['Health Expenditure %'].max() - country_data['Health Expenditure %'].min())
    country_data['education percent normalized'] = (country_data['Education Expenditure %'] - country_data['Education Expenditure %'].min()) / (country_data['Education Expenditure %'].max() - country_data['Education Expenditure %'].min())
    return plot_data_for_country(country_data, disorder, country)

def plot_data_for_country(country_data, disorder, country):
    plt.scatter(country_data['Year'], country_data[disorder + " normalized"], color='blue', label=disorder)
    plt.scatter(country_data['Year'], country_data['health percent normalized'], color='green', label='Health Expenditure %')
    plt.scatter(country_data['Year'], country_data['education percent normalized'], color='orange', label='Education Expenditure %')
    plt.scatter(country_data['Year'], country_data['gdp normalized'], color='red', label='GDP (%)')
    plt.xticks(rotation=45) 
    plt.xlabel('Year')
    plt.ylabel('Normalized Values')
    title = disorder + ", Health Expenditure, Education Expenditure, and GDP in " + country + " over Time"
    plt.title(title)
    plt.legend()
    plt.savefig('plots/country_data_over_time.png')
    plt.show()
    return CalcSpearman(country_data, disorder)

def CalcSpearman(country_data, disorder):
    spearman_health_percent_disorder = country_data[disorder + " normalized"].corr(country_data['health percent normalized'], method='spearman')
    spearman_education_percent_disorder = country_data[disorder + " normalized"].corr(country_data['education percent normalized'], method='spearman')
    spearman_gdp_disorder = country_data[disorder + " normalized"].corr(country_data['gdp normalized'], method='spearman')
    return [spearman_health_percent_disorder, spearman_education_percent_disorder, spearman_gdp_disorder]

def print_spearmen(chosen_mental_health_disorder, spearman_array):
    print("Spearman's rank correlation coefficient for health expenditure vs.", chosen_mental_health_disorder, spearman_array[0])
    print("Spearman's rank correlation coefficient for education expenditure vs.", chosen_mental_health_disorder, spearman_array[1])
    print("Spearman's rank correlation coefficient for GDP vs.", chosen_mental_health_disorder, spearman_array[2])
