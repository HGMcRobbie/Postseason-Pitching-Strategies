# Hunter Galusha-McRobbie
# Analyzing Postseason Pitching Strategies
# 18 September 2026 -

# Import Statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------
# Introduction
# --------------------

# Initial DataFrame
pitching_df = pd.read_csv("pitching.csv")
pitching_df["date"] = pd.to_datetime(pitching_df["date"].astype(str), format="%Y%m%d")

# DataFrame from 2000-2025, excluding 2020
pitching_df["season"] = pitching_df["date"].dt.year
pitching_df_since_2000 = pitching_df[(pitching_df["season"] >= 2000) & (pitching_df["season"] <= 2025) & (pitching_df["season"] != 2020)]

# Data Cleaning
# print(pitching_df_since_2000.isna().sum())
pitching_df_since_2000[["wp", "lp", "save", "p_gs", "p_gf", "p_cg"]] = \
    (pitching_df_since_2000[["wp", "lp", "save", "p_gs", "p_gf", "p_cg"]].fillna(0))
# print(pitching_df_since_2000.isna().sum())

# Creating Regular Season DataFrame
regular_df = pitching_df_since_2000[pitching_df_since_2000["gametype"] == "regular"]

# Creating Postseason DataFrame
# print(pitching_df_since_2000["gametype"].value_counts())
playoff_game_types = ["divisionseries", "lcs", "worldseries", "wildcard", "playoff"] # playoff corresponds to game 163s,
# which we will include as those are strategically more similar to playoff games than regular season games
postseason_df = pitching_df_since_2000[pitching_df_since_2000["gametype"].isin(playoff_game_types)]

# Creating one row for each team in each postseason game
postseason_team_games_df = postseason_df.drop_duplicates(
    subset=["gid", "team"]).reset_index(drop=True)


# --------------------
# Part 1 - Bringing in the Big Guns: Starters in Relief
# --------------------

# Definitions

# DataFrames

# Identifying Qualified Starting Pitchers in Each Season
starters_df = regular_df.groupby(["id", "season"]).agg(
    starts = ("p_gs", "sum"),
    outs = ("p_ipouts", "sum")
).reset_index()
qualified_starters_df = starters_df[(starters_df["starts"] >= 20) & (starters_df["outs"] >= 486)]
# print(qualified_starters_df)

# Identifying Instances where these Starters pitched in relief in the postseason
qualified_starters_in_relief_appearances = []
all_others_in_relief_appearances = []

for index, appearance in postseason_df.iterrows():
    if appearance["p_seq"] > 1:
        matching_starter = qualified_starters_df[
            (qualified_starters_df["id"] == appearance["id"]) &
            (qualified_starters_df["season"] == appearance["season"])
        ]
        if len(matching_starter) > 0:
            qualified_starters_in_relief_appearances.append(appearance)
        else:
            all_others_in_relief_appearances.append(appearance)
# print(len(qualified_starters_in_relief_appearances))
# print(len(all_others_in_relief_appearances))

# Convert lists into DataFrames
qualified_starters_in_relief_df = pd.DataFrame(qualified_starters_in_relief_appearances).reset_index(drop=True)
all_others_in_relief_df = pd.DataFrame(all_others_in_relief_appearances).reset_index(drop=True)
# print(qualified_starters_in_relief_df)
# print(all_others_in_relief_df)

# a) Most Appearances
qualified_starters_in_relief_appearances_by_pitcher = (qualified_starters_in_relief_df["id"].value_counts())
# print(qualified_startesr_in_relief_appearances_by_pitcher[qualified_starters_in_relief_appearances_by_pitcher > 6])

# b) Over Time
qualified_starter_in_relief_count_by_season = (qualified_starters_in_relief_df.groupby("season")["id"].count())
# print(qualified_starter_in_relief_count_by_season)

# c) ERA and WHIP
# Determining ERA and WHIP for Qualified Starters
qualified_starters_in_relief_outs = qualified_starters_in_relief_df["p_ipouts"].sum()
qualified_starters_in_relief_earned_runs = qualified_starters_in_relief_df["p_er"].sum()
qualified_starters_in_relief_hits = qualified_starters_in_relief_df["p_h"].sum()
qualified_starters_in_relief_walks = qualified_starters_in_relief_df["p_w"].sum()

qualified_starters_in_relief_era = (qualified_starters_in_relief_earned_runs * 27) / qualified_starters_in_relief_outs
qualified_starters_in_relief_whip = ((qualified_starters_in_relief_hits + qualified_starters_in_relief_walks) * 3) / qualified_starters_in_relief_outs

# print(len(qualified_starters_in_relief_df)) # >>> 281
# print(round(qualified_starters_in_relief_era, 2)) # >>> 4.05
# print(round(qualified_starters_in_relief_whip, 2)) # >>> 1.35

# Determining ERA and WHIP for all others in relief
all_others_in_relief_outs = all_others_in_relief_df["p_ipouts"].sum()
all_others_in_relief_earned_runs = all_others_in_relief_df["p_er"].sum()
all_others_in_relief_hits = all_others_in_relief_df["p_h"].sum()
all_others_in_relief_walks = all_others_in_relief_df["p_w"].sum()

all_others_in_relief_era = (all_others_in_relief_earned_runs * 27) / all_others_in_relief_outs
all_others_in_relief_whip = ((all_others_in_relief_hits + all_others_in_relief_walks) * 3) / all_others_in_relief_outs

# print(len(all_others_in_relief_df))
# print(round(all_others_in_relief_era, 2)) # >>> 3.52
# print(round(all_others_in_relief_whip, 2)) # >>> 1.24

# d) K% and BB%
# Determining K% and BB% for Qualified Starters
qualified_starters_in_relief_strikeouts = qualified_starters_in_relief_df["p_k"].sum()
qualified_starters_in_relief_batters_faced = qualified_starters_in_relief_df["p_bfp"].sum()

qualified_starters_in_relief_k_percentage = (
    qualified_starters_in_relief_strikeouts /
    qualified_starters_in_relief_batters_faced
) * 100
# print(round(qualified_starters_in_relief_k_percentage, 2)) # >>> 21.49%

qualified_starters_in_relief_bb_percentage = (
    qualified_starters_in_relief_walks /
    qualified_starters_in_relief_batters_faced
) * 100
# print(round(qualified_starters_in_relief_bb_percentage, 2)) # >>> 9.76%

# Determining K% and BB% for All Other Pitchers
all_others_in_relief_strikeouts = all_others_in_relief_df["p_k"].sum()
all_others_in_relief_batters_faced = all_others_in_relief_df["p_bfp"].sum()

all_others_in_relief_k_percentage = (
    all_others_in_relief_strikeouts /
    all_others_in_relief_batters_faced
) * 100
# print(round(all_others_in_relief_k_percentage, 2)) # >>> 23.33%

all_others_in_relief_bb_percentage = (
    all_others_in_relief_walks /
    all_others_in_relief_batters_faced
) * 100
# print(round(all_others_in_relief_bb_percentage, 2)) # >>> 9.47%

# e) Win%
# Determining Win% with and without bringing in a qualified starter in relief

# Creating one row for each team-game where a qualified starter appeared in relief
qualified_starter_team_games_df = qualified_starters_in_relief_df.drop_duplicates(
    subset=["gid", "team"]).reset_index(drop=True)

team_games_with_qualified_starter = []
team_games_without_qualified_starter = []

for index, team_game in postseason_team_games_df.iterrows():
    matching_team_game = qualified_starter_team_games_df[
        (qualified_starter_team_games_df["gid"] == team_game["gid"]) &
        (qualified_starter_team_games_df["team"] == team_game["team"])
    ]

    if len(matching_team_game) > 0:
        team_games_with_qualified_starter.append(team_game)
    else:
        team_games_without_qualified_starter.append(team_game)

# Convert lists into DataFrames
team_games_with_qualified_starter_df = pd.DataFrame(
    team_games_with_qualified_starter).reset_index(drop=True)

team_games_without_qualified_starter_df = pd.DataFrame(
    team_games_without_qualified_starter).reset_index(drop=True)

# Calculating team win percentage for each group
wins_with_qualified_starter = team_games_with_qualified_starter_df["win"].sum()
wins_without_qualified_starter = team_games_without_qualified_starter_df["win"].sum()

win_percentage_with_qualified_starter = (
    wins_with_qualified_starter /
    len(team_games_with_qualified_starter_df)
) * 100

win_percentage_without_qualified_starter = (
    wins_without_qualified_starter /
    len(team_games_without_qualified_starter_df)
) * 100

# print(len(team_games_with_qualified_starter_df)) # >>> 259
# print(round(win_percentage_with_qualified_starter, 2)) # >>> 36.29%

# print(len(team_games_without_qualified_starter_df)) # >>> 1525
# print(round(win_percentage_without_qualified_starter, 2)) # >>> 52.33%


# --------------------
# Part 2 - Let Him Ride: Extended Starts
# --------------------

# Definitions
# Extended Start: Starter records at least 21 outs, equivalent to 7+ innings pitched

# DataFrames
# Extended Starts DataFrame
extended_starts_df = postseason_df[
    (postseason_df["p_seq"] == 1) &
    (postseason_df["p_ipouts"] >= 21)
]
# print(len(extended_starts_df)) # >>> 387

# All Other Starts DataFrame
all_other_starts_df = postseason_df[
    (postseason_df["p_seq"] == 1) &
    (postseason_df["p_ipouts"] < 21)
]
# print(len(all_other_starts_df)) # >>> 1397

# a) Most Appearances
extended_starts_by_pitcher = (extended_starts_df["id"].value_counts())
# print(extended_starts_by_pitcher[extended_starts_by_pitcher >= 9])

# b) Over Time
extended_start_count_by_season = (extended_starts_df.groupby("season")["id"].count())
# print(extended_start_count_by_season)

# c) ERA and WHIP
# Determining ERA and WHIP for Extended Starts
extended_start_outs = extended_starts_df["p_ipouts"].sum()
extended_start_earned_runs = extended_starts_df["p_er"].sum()
extended_start_hits = extended_starts_df["p_h"].sum()
extended_start_walks = extended_starts_df["p_w"].sum()

extended_start_era = (extended_start_earned_runs * 27) / extended_start_outs
extended_start_whip = ((extended_start_hits + extended_start_walks) * 3) / extended_start_outs

# print(round(extended_start_era, 2)) # >>> 1.61
# print(round(extended_start_whip, 2)) # >>> 0.81

# Determining ERA and WHIP for All Other Starts
all_other_start_outs = all_other_starts_df["p_ipouts"].sum()
all_other_start_earned_runs = all_other_starts_df["p_er"].sum()
all_other_start_hits = all_other_starts_df["p_h"].sum()
all_other_start_walks = all_other_starts_df["p_w"].sum()

all_other_start_era = (all_other_start_earned_runs * 27) / all_other_start_outs
all_other_start_whip = ((all_other_start_hits + all_other_start_walks) * 3) / all_other_start_outs

# print(round(all_other_start_era, 2)) # >>> 5.11
# print(round(all_other_start_whip, 2)) # >>> 1.48

# d) K% and BB%
# Determining K% and BB% for Extended Starts
extended_start_strikeouts = extended_starts_df["p_k"].sum()
extended_start_batters_faced = extended_starts_df["p_bfp"].sum()

extended_start_k_percentage = (
    extended_start_strikeouts /
    extended_start_batters_faced
) * 100
# print(round(extended_start_k_percentage, 2)) # >>> 23.93%

extended_start_bb_percentage = (
    extended_start_walks /
    extended_start_batters_faced
) * 100
# print(round(extended_start_bb_percentage, 2)) # >>> 5.27%

# Determining K% and BB% for All Other Starts
all_other_start_strikeouts = all_other_starts_df["p_k"].sum()
all_other_start_batters_faced = all_other_starts_df["p_bfp"].sum()

all_other_start_k_percentage = (
    all_other_start_strikeouts /
    all_other_start_batters_faced
) * 100
# print(round(all_other_start_k_percentage, 2)) # >>> 19.49%

all_other_start_bb_percentage = (
    all_other_start_walks /
    all_other_start_batters_faced
) * 100
# print(round(all_other_start_bb_percentage, 2)) # >>> 9.15%

# e) Win%
# Determining Win% With and Without an Extended Start

# Creating one row for each team-game with an extended start
extended_start_team_games_df = extended_starts_df.drop_duplicates(
    subset=["gid", "team"]).reset_index(drop=True)

team_games_with_extended_start = []
team_games_without_extended_start = []

for index, team_game in postseason_team_games_df.iterrows():
    matching_team_game = extended_start_team_games_df[
        (extended_start_team_games_df["gid"] == team_game["gid"]) &
        (extended_start_team_games_df["team"] == team_game["team"])
    ]

    if len(matching_team_game) > 0:
        team_games_with_extended_start.append(team_game)
    else:
        team_games_without_extended_start.append(team_game)

# Convert lists into DataFrames
team_games_with_extended_start_df = pd.DataFrame(
    team_games_with_extended_start).reset_index(drop=True)

team_games_without_extended_start_df = pd.DataFrame(
    team_games_without_extended_start).reset_index(drop=True)

# Calculating team win percentage for each group
wins_with_extended_start = team_games_with_extended_start_df["win"].sum()
wins_without_extended_start = team_games_without_extended_start_df["win"].sum()

win_percentage_with_extended_start = (
    wins_with_extended_start /
    len(team_games_with_extended_start_df)
) * 100

win_percentage_without_extended_start = (
    wins_without_extended_start /
    len(team_games_without_extended_start_df)
) * 100

# print(len(team_games_with_extended_start_df)) # >>> 387
# print(round(win_percentage_with_extended_start, 2)) # >>> 71.58%

# print(len(team_games_without_extended_start_df)) # >>> 1397
# print(round(win_percentage_without_extended_start, 2)) # >>> 44.02%


# --------------------
# Part 3 - All Hands on Deck: Bullpen Games
# --------------------

# Definitions
# Bullpen Game: No pitcher records more than 30% of the team's outs, the team uses at least five pitchers,
# and the starter was not a qualified starter during the regular season

# DataFrames
# Identifying Bullpen Games
bullpen_games = []
all_other_games = []

for index, team_game in postseason_team_games_df.iterrows():
    team_pitchers = postseason_df[
        (postseason_df["gid"] == team_game["gid"]) &
        (postseason_df["team"] == team_game["team"])]

    total_team_outs = team_pitchers["p_ipouts"].sum()
    number_of_pitchers = len(team_pitchers)
    most_outs_by_one_pitcher = team_pitchers["p_ipouts"].max()

    most_outs_percentage = (most_outs_by_one_pitcher / total_team_outs) * 100

    starting_pitcher = team_pitchers[
        team_pitchers["p_seq"] == 1
        ].iloc[0]

    matching_qualified_starter = qualified_starters_df[
        (qualified_starters_df["id"] == starting_pitcher["id"]) &
        (qualified_starters_df["season"] == starting_pitcher["season"])]

    team_game["number_of_pitchers"] = number_of_pitchers
    team_game["most_outs_percentage"] = most_outs_percentage
    team_game["starting_pitcher_id"] = starting_pitcher["id"]

    if (
        most_outs_percentage <= 30
        and number_of_pitchers >= 5
        and len(matching_qualified_starter) == 0
    ):
        bullpen_games.append(team_game)
    else:
        all_other_games.append(team_game)

# Convert lists into DataFrames
bullpen_games_df = pd.DataFrame(bullpen_games).reset_index(drop=True)

all_other_games_df = pd.DataFrame(all_other_games).reset_index(drop=True)

# print(len(bullpen_games_df)) # >>> 50
# print(len(all_other_games_df)) # >>> 1734
print(bullpen_games_df[["date", "team", "opp", "starting_pitcher_id", "number_of_pitchers", "most_outs_percentage", "win"]])

# a) Most Appearances

# b) Over Time

# c) ERA and WHIP

# d) K% and BB%

# e) Win%


# --------------------
# Part 4 - No Days Off: Starters on Short Rest
# --------------------

# Definitions

# DataFrames

# a) Most Appearances

# b) Over Time

# c) ERA and WHIP

# d) K% and BB%

# e) Win%


# --------------------
# Conclusion
# --------------------





# OTHER PITCHING STRATEGIES TO ANALYZE
# 1. Bringing in the Big Guns: Starters in Relief
# 2. Let Him Ride: Extended Starts
# ---> 7+ Innings Pitched
# 3. All Hands on Deck: Bullpen Games
# ---> >= 5 Pitchers and Starter went <= 3 innings ??
# 4. No Days Off: Starters on Short Rest
# ---> Maybe 2 or less full days of rest