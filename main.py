from functions.geospatial_over_time import *
from functions.specific_country_plots import *
from functions.specific_time import *


# input variables #
mental_health_disorders = ['Depression (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)']
economic_stats = ['Health Expenditure %', 'Education Expenditure %', 'gini_index', 'gdp']
year = 2012


# Generate GeoSpatial plots for a specific characteristic, either economic or mental health #
df_with_combined_stats = load_data_specific_time(economic_stats[2], year) # can take either mental health or economic factor, returns dataframe for model
# make model #
final_df = create_df_for_model(df_with_combined_stats) # returns cleaned dataframe for model
create_and_train_model(mental_health_disorders[0], final_df) # train model on dataframe and plot




# plot all economic data from a specific country vs a single mental health statistic #
country = 'Bangladesh' # choose a country
chosen_mental_health_disorder = mental_health_disorders[0] # change this index [0, 2]

spearman_array = load_data_for_country(country, chosen_mental_health_disorder) # takes only a mental health disorder (not economic), returns array of spearmen correlations (0: health expenditure, 1: education expenditure, 2: GDP)
print_spearmen(chosen_mental_health_disorder, spearman_array)


# plots a mental health statistic GeoSpatially over time #
load_data_world(chosen_mental_health_disorder) # takes only mental disorder (not economic)

