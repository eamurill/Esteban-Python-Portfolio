# Esteban Murillo
# ITP 449
# Homework 5
# Question 3

import pandas as pd


df = pd.read_csv('../CSV Files/10_02_2022.csv')

print(df)

total_deaths_by_state = df.groupby('Province_State')['Deaths'].sum()
# myDF2 = pd.read_csv('time_series_covid19_confirmed_US_1_.csv')
# myDF2 = pd.DataFrame(myDF2)
# myDF3 = pd.read_csv('time_series_covid19_deaths_US_1_.csv')
# myDF3 = pd.DataFrame(myDF3)


state_highest_deaths = df["Province_State"].values[df.Deaths == df.Deaths.max()]
highest_deaths = state_highest_deaths.max()

print(df.columns)

print("the state with the highest number of deaths is", state_highest_deaths, "with", highest_deaths, "deaths")

# California has the highest number of Deaths

# 2.
sorted_incident_rate = df[['Province_State', 'Incident_Rate']].sort_values(by = 'Incident_Rate', ascending=False)


second_lowest_incident_rate_state = sorted_incident_rate.iloc[1]

print("The state with the 2nd lowest incident rate is", second_lowest_incident_rate_state['Province_State'],
      "with an incident rate of", second_lowest_incident_rate_state['Incident_Rate'])

#3:
df = df.dropna(subset=['Case_Fatality_Ratio'])

highest_cfr_state = df[df['Case_Fatality_Ratio'] == df['Case_Fatality_Ratio'].max()]['Province_State'].values[0]
lowest_cfr_state = df[df['Case_Fatality_Ratio'] == df['Case_Fatality_Ratio'].min()]['Province_State'].values[0]

# Calculate difference in case fatality ratio
cfr_difference = abs(df['Case_Fatality_Ratio'].max() - df['Case_Fatality_Ratio'].min())

print("State with highest case fatality ratio:", highest_cfr_state)
print("State with lowest case fatality ratio:", lowest_cfr_state)
print("Difference in case fatality ratio:", cfr_difference)

# 4

import pandas as pd
import matplotlib.pyplot as plt

df_confirmed = pd.read_csv('../CSV Files/time_series_covid19_confirmed_US_1_.csv')
df_deaths = pd.read_csv('../CSV Files/time_series_covid19_deaths_US_1_.csv')
df_states_info = pd.read_csv('../CSV Files/10_02_2022.csv')

top_5_states = df_states_info[['Province_State', 'Confirmed']].sort_values(by='Confirmed', ascending=False)

top_5_states_names = top_5_states['Province_State'].tolist()

date_columns = df_confirmed.columns[11:]

df_top5_confirmed = df_confirmed[df_confirmed['Province_State'].isin(top_5_states_names)]
df_top5_deaths = df_deaths[df_deaths['Province_State'].isin(top_5_states_names)]

df_daily_confirmed = df_top5_confirmed[date_columns].diff(axis=1).fillna(0)
df_daily_deaths = df_top5_deaths[date_columns].diff(axis=1).fillna(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for i in range(len(top_5_states_names)):
    state = top_5_states_names[i]
    ax1.plot(date_columns, df_daily_confirmed.iloc[1], label=state)
ax1.set_title('Daily new Case in Top 5 States')
ax1.set_ylabel('Number of Cases')
ax1.legend()

for i in range(len(top_5_states_names)):
    state = top_5_states_names[i]
    ax2.plot(date_columns, df_daily_deaths.iloc[1], label=state)
ax2.set_title('Daily new Deaths in Top 5 States')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Deaths')
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()