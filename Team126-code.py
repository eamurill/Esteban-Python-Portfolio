import pandas as pd
import pyarrow.dataset as pads
import os
from numpy import sqrt, abs
from SMT_data_starter import readDataSubset
from statsmodels.stats.weightstats import ztest
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def read_game_events(data_path):
    # only looking for pickoffs and actual pitches
    game_events_subset = readDataSubset('game_events', data_path)
    game_events = game_events_subset.to_table(filter=(pads.field('Season') == "Season_1884") & (pads.field('event_code').isin([1, 6]))).to_pandas()
    return game_events

def read_player_pos(data_path):
    player_pos_subset = readDataSubset('player_pos', data_path)
    player_pos_runner = player_pos_subset.to_table(
        filter=(pads.field('Season') == "Season_1884") & (pads.field('player_position') == 11)).to_pandas()
    player_pos_1b = player_pos_subset.to_table(
        filter=(pads.field('Season') == "Season_1884") & (pads.field('player_position') == 3)).to_pandas()
    return player_pos_1b, player_pos_runner

def read_game_info(data_path):
    game_info_subset = readDataSubset('game_info', data_path)
    game_info = game_info_subset.to_table(filter=(pads.field('Season') == "Season_1884")).to_pandas()
    return game_info

def read_csv_from_day_directories(directory):
    """Read CSV files from day directories and concatenate them into a single DataFrame."""
    dfs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def pitcher_hand(data_path):
    # Read datasets
    game_events_subset = readDataSubset('game_events', data_path)
    game_events = game_events_subset.to_table(filter=(pads.field('Season') == "Season_1884")).to_pandas()

    game_info_path = os.path.join(data_path, 'game_info/Season_1884')
    game_info = read_csv_from_day_directories(game_info_path)

    ball_pos_subset = readDataSubset('ball_pos', data_path)
    ball_pos = ball_pos_subset.to_table(filter=(pads.field('Season') == "Season_1884")).to_pandas()

    # Merge dataframes
    pitches = game_events[game_events['event_code'] == 1]

    pitches_with_pitcher = pd.merge(pitches, game_info[['game_str', 'pitcher', 'play_per_game']],
                                    on=['game_str', 'play_per_game'], how='left')
    pitches_with_ball_pos = pd.merge(pitches_with_pitcher,
                                     ball_pos[['game_str', 'play_id', 'timestamp', 'ball_position_x']],
                                     on=['game_str', 'play_id', 'timestamp'], how='left')

    # Perform sorting and drop duplicates
    pitches_with_ball_pos_sorted = pitches_with_ball_pos.sort_values(by=['game_str', 'play_id', 'timestamp'],
                                                                     ascending=[True, True, False])
    pitches_with_pitchers_unique = pitches_with_ball_pos_sorted.drop_duplicates(subset=['game_str', 'play_id'],
                                                                                keep='first')

    # Determine pitcher hand based on ball position
    pitches_with_pitchers_unique = pitches_with_pitchers_unique.copy()
    pitches_with_pitchers_unique.loc[:, 'pitcher_hand'] = pitches_with_pitchers_unique['ball_position_x'].apply(
        lambda x: 'Right' if x < 0 else ('Left' if x > 0 else 'Unknown')
    )

    # Prepare final summary table
    pitcher_hands_table = pitches_with_pitchers_unique[
        ['game_str', 'HomeTeam', 'pitcher', 'pitcher_hand']].drop_duplicates()
    pitcher_hands_table = pitcher_hands_table.sort_values(by=['game_str', 'HomeTeam', 'pitcher'],
                                                          ascending=[True, True, True])
    return pitcher_hands_table

def pickoff_events_and_after_data(game_events):
    # Find indices where event_code == 6 (the pickoff event code)
    pickoff_indices = game_events[game_events['event_code'] == 6].index
    # Initialize a list to hold the indices for the new DataFrame
    indices_to_include = []
    # Select the rows with the indices to include. (pitch before, pickoff, pitch after)
    for idx in pickoff_indices:
        indices_to_include.extend(
            range(idx-1, min(idx + 2, len(game_events))))
    # Remove duplicate indices if any (though there shouldn't be in this case)
    indices_to_include = list(dict.fromkeys(indices_to_include))
    # We need the play after in order to decide if the pickoff was sucessful or not
    pickoffs_and_after = game_events.loc[indices_to_include]
    # Reset the index
    pickoffs_and_after.reset_index(drop=True, inplace=True)
    return pickoffs_and_after

def merge_datasets(game_events, game_info, player_pos_1b, player_pos_runner, pitcher_hands):
    events_and_info = pd.merge(left=game_events, right=game_info, how='left', on=['game_str', 'play_per_game'], suffixes=('', '_info'))
    player_pos = pd.merge(left=player_pos_1b, right=player_pos_runner, how='right', on=['game_str', 'timestamp'], suffixes=('_1b', '_runner'))
    merged_df_all_columns = pd.merge(left=events_and_info, right=player_pos, how='left',
                                                   on=['game_str', 'timestamp'], suffixes=('', '_pos'))
    merged_df_all_columns = pd.merge(left=merged_df_all_columns, right=pitcher_hands, how='left', on='pitcher', suffixes=('', 'pitcher_hand'))
    columns_wanted = ['game_str', 'play_id', 'at_bat', 'timestamp', 'player_position', 'event_code', 'Season', 'HomeTeam',
                      'AwayTeam', 'Day', 'inning', 'top_bottom', 'pitcher', 'pitcher_hand','first_baserunner', 'second_baserunner','third_baserunner','field_x_1b',
                      'field_y_1b', 'field_x_runner', 'field_y_runner']
    merged_df = merged_df_all_columns.loc[:, columns_wanted]
    # center of 1B: (62.58, 63.64)
    merged_df.loc[:, 'distance_from_1b'] = sqrt((merged_df['field_x_runner'] - 62.58)**2 + (merged_df['field_y_runner'] - 63.64)**2)
    merged_df.loc[:, 'distance_to_baseline'] = (abs(1.017*merged_df['field_x_runner'] + merged_df['field_y_runner'] - 127.281)/
                                                sqrt(1.017**2 + 1**2))
    merged_df.loc[:, 'lead_distance'] = sqrt(merged_df['distance_from_1b']**2 - merged_df['distance_to_baseline']**2)
    merged_df['pitch_label'] = merged_df.index % 3 # modulo function to label pitches as before, pickoff, and after
    merged_df['pitch_label'] = merged_df['pitch_label'].map({0: 'before', 1: 'pickoff', 2: 'after'})
    # Filter so that we only use rows with a runner on first and other bases empty
    merged_df = merged_df[
        merged_df['first_baserunner'].notna() &
        merged_df['second_baserunner'].isna() &
        merged_df['third_baserunner'].isna() &
        merged_df['lead_distance'].notna()
    ]
    return merged_df

def pickoff_trios(merged_df):
    # to create 'trios' of before, pickoff, after. each 'before' starts a new trio:
    merged_df['group_number'] = (merged_df['pitch_label'] == 'before').cumsum()
    # take the first three rows from each group_number
    merged_df_trios = merged_df.groupby('group_number').head(3)
    merged_df_trios = merged_df.groupby('group_number').filter(lambda x: len(x) == 3) # only keeping groups that have exactly three rows (some previously had two)
    # 749 trios is a healthy sample size
    return merged_df_trios

def means_by_split(merged_df):
    means = pd.DataFrame()
    before_left_col = merged_df.loc[(merged_df['pitcher_hand'] == 'Left') & (merged_df['pitch_label'] == 'before'), 'lead_distance']
    pickoff_left_col = merged_df.loc[(merged_df['pitcher_hand'] == 'Left') & (merged_df['pitch_label'] == 'pickoff'), 'lead_distance']
    after_left_col = merged_df.loc[(merged_df['pitcher_hand'] == 'Left') & (merged_df['pitch_label'] == 'after'), 'lead_distance']
    before_right_col = merged_df.loc[(merged_df['pitcher_hand'] == 'Right') & (merged_df['pitch_label'] == 'before'), 'lead_distance']
    pickoff_right_col = merged_df.loc[(merged_df['pitcher_hand'] == 'Right') & (merged_df['pitch_label'] == 'pickoff'), 'lead_distance']
    after_right_col = merged_df.loc[(merged_df['pitcher_hand'] == 'Right') & (merged_df['pitch_label'] == 'after'), 'lead_distance']
    # assign to means dataframe
    means.loc['Before', 'Left'] = before_left_col.mean()
    means.loc['Pickoff', 'Left'] = pickoff_left_col.mean()
    means.loc['After', 'Left'] = after_left_col.mean()
    means.loc['Before', 'Right'] = before_right_col.mean()
    means.loc['Pickoff', 'Right'] = pickoff_right_col.mean()
    means.loc['After', 'Right'] = after_right_col.mean()
    print("Hand and Trios Mean Matrix: \n", means)
    return before_right_col, pickoff_left_col, after_left_col, before_right_col, pickoff_right_col, after_right_col

def ztest_by_hand(pitches_pickoffs_merged_df):
    pitches_merged_df = pitches_pickoffs_merged_df.loc[pitches_pickoffs_merged_df['event_code'] == 1] # just want pitches, not pickoffs as well
    left_lead = pitches_merged_df.loc[pitches_merged_df['pitcher_hand'] == 'Left']['lead_distance']
    right_lead = pitches_merged_df.loc[pitches_merged_df['pitcher_hand'] == 'Right']['lead_distance']
    z_stat, pvalue = ztest(list(left_lead), list(right_lead), value=0)
    print(f"Pitcher Hand Z Test P-Value: {pvalue}")
    return z_stat, pvalue

def anova_and_posthoc_trio(pickoffs_merged_df_trios):
    before_list = list(pickoffs_merged_df_trios.loc[pickoffs_merged_df_trios['pitch_label'] == 'before']['lead_distance'])
    pickoff_list = list(pickoffs_merged_df_trios.loc[pickoffs_merged_df_trios['pitch_label'] == 'pickoff']['lead_distance'])
    after_list = list(pickoffs_merged_df_trios.loc[pickoffs_merged_df_trios['pitch_label'] == 'after']['lead_distance'])
    f_stat, pvalue = f_oneway(before_list, pickoff_list, after_list)
    # Significant at .05 level
    tukey = pairwise_tukeyhsd(endog=pickoffs_merged_df_trios['lead_distance'], groups=pickoffs_merged_df_trios['pitch_label'], alpha=.05)
    return tukey

def anova_and_posthoc_levels(pitches_pickoffs_merged_df):
    pitches_merged_df = pitches_pickoffs_merged_df.loc[pitches_pickoffs_merged_df['event_code'] == 1]
    means = pd.DataFrame()
    a_column = pitches_pickoffs_merged_df.loc[pitches_pickoffs_merged_df['HomeTeam'] == 'Home1A']['lead_distance']
    aa_column = pitches_pickoffs_merged_df.loc[pitches_pickoffs_merged_df['HomeTeam'] == 'Home2A']['lead_distance']
    aaa_column = pitches_pickoffs_merged_df.loc[pitches_pickoffs_merged_df['HomeTeam'] == 'Home3A']['lead_distance']
    aaaa_column = pitches_pickoffs_merged_df.loc[pitches_pickoffs_merged_df['HomeTeam'] == 'Home4A']['lead_distance']
    means.loc['A', 'Mean'] = a_column.mean()
    means.loc['AA', 'Mean'] = aa_column.mean()
    means.loc['AAA', 'Mean'] = aaa_column.mean()
    means.loc['AAAA', 'Mean'] = aaaa_column.mean()
    print("Level Means: \n", means)

    f_stat, pvalue = f_oneway(a_column, aa_column, aaa_column, aaaa_column)
    tukey = pairwise_tukeyhsd(endog=pitches_pickoffs_merged_df['lead_distance'], groups=pitches_pickoffs_merged_df['HomeTeam'], alpha=.05)
    return tukey

def main():
    data_path = "/Team-126/2024_SMT_Data_Challenge"
    pitcher_hands = pitcher_hand(data_path)
    pitcher_hands_unique = pitcher_hands.drop_duplicates(subset='pitcher', keep='first')
    game_events = read_game_events(data_path)
    game_info = read_game_info(data_path)
    pickoffs_and_after = pickoff_events_and_after_data(game_events)
    player_pos_1b, player_pos_runner = read_player_pos(data_path)
    pickoffs_merged_df = merge_datasets(pickoffs_and_after, game_info, player_pos_1b, player_pos_runner, pitcher_hands_unique)
    pickoffs_merged_df_trios = pickoff_trios(pickoffs_merged_df)
    before_right_col, pickoff_left_col, after_left_col, before_right_col, pickoff_right_col, after_right_col = means_by_split(pickoffs_merged_df_trios)
    pitches_pickoffs_merged_df = merge_datasets(game_events, game_info, player_pos_1b, player_pos_runner, pitcher_hands)
    hand_z_stat, hand_pvalue = ztest_by_hand(pitches_pickoffs_merged_df) # Z Test comparing pitcher hands
    tukey_pickoffs = anova_and_posthoc_trio(pickoffs_merged_df_trios)
    print("Tukey Test, Pickoff Trios: \n", tukey_pickoffs)
    tukey_levels = anova_and_posthoc_levels(pitches_pickoffs_merged_df)
    print("Tukey Test, Levels: \n", tukey_levels)

main()