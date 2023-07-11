import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

def read_data(file_path):
    """
    Read the data from the specified file path.
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        return data
    except FileNotFoundError:
        st.error("Error: CSV file not found. Please provide the correct file path.")
        return None

def filter_data(data, start_year, end_year, league=None, division=None):
    """
    Filter the data based on the specified start and end years, league, and division.
    """
    filtered_data = data[(data['yearID'] >= start_year) & (data['yearID'] <= end_year)]

    if league:
        filtered_data = filtered_data[filtered_data['lgID'] == league]
    if division:
        filtered_data = filtered_data[filtered_data['divID'] == division]

    return filtered_data

def sort_data(data, sort_by):
    """
    Sort the data based on the specified column.
    """
    sorted_data = data.sort_values(sort_by)
    return sorted_data

def calculate_batting_average(hits, at_bats):
    """
    Calculate the batting average given the number of hits and at-bats.
    """
    return hits / at_bats

def visualize_team_performance(teams_data):
    """
    Visualize the team performance over time.
    """
    teams_data['WinningPercentage'] = teams_data['W'] / teams_data['G']
    teams_data['RunDifferential'] = teams_data['R'] - teams_data['RA']

    fig, ax = plt.subplots()
    ax.plot(teams_data['yearID'], teams_data['WinningPercentage'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Winning Percentage')
    ax.set_title('Team Performance: Winning Percentage Over Time')

    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(teams_data['yearID'], teams_data['RunDifferential'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Run Differential')
    ax.set_title('Team Performance: Run Differential Over Time')

    st.pyplot(fig)

def visualize_home_runs(batting_data):
    """
    Visualize the number of home runs hit each year.
    """
    home_runs_by_year = batting_data.groupby('yearID')['HR'].sum()

    fig, ax = plt.subplots()
    ax.bar(home_runs_by_year.index, home_runs_by_year.values)
    ax.set_xlabel('Year')
    ax.set_ylabel('Home Runs')
    ax.set_title('Home Runs in MLB: 2012-2022')

    st.pyplot(fig)

def visualize_average_era(pitching_data):
    """
    Visualize the average ERA of MLB over the 2012-2022 seasons.
    """
    average_era_by_year = pitching_data.groupby('yearID')['ERA'].mean()

    fig, ax = plt.subplots()
    ax.plot(average_era_by_year.index, average_era_by_year.values, marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average ERA')
    ax.set_title('Average ERA in MLB: 2012-2022')

    st.pyplot(fig)

def visualize_average_batting_average(batting_data):
    """
    Visualize the average batting average (BA) of MLB over the 2012-2022 seasons.
    """
    batting_data['BA'] = calculate_batting_average(batting_data['H'], batting_data['AB'])
    average_batting_average_by_year = batting_data.groupby('yearID')['BA'].mean()

    fig, ax = plt.subplots()
    ax.plot(average_batting_average_by_year.index, average_batting_average_by_year.values, marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Batting Average (BA)')
    ax.set_title('Average Batting Average (BA) in MLB: 2012-2022')

    st.pyplot(fig)

def get_player_batting_stats(player_name, batting_data):
    """
    Get batting stats for a specific player over the 2012-2022 seasons.
    """
    player_stats = batting_data[batting_data['playerID'] == player_name]
    player_stats = filter_data(player_stats, 2012, 2022)
    return player_stats
    
def visualize_cubs_performance(cubs_data):
    """
    Visualize the Chicago Cubs' performance over time.
    """
    cubs_data['WinningPercentage'] = cubs_data['W'] / cubs_data['G']
    cubs_data['RunDifferential'] = cubs_data['R'] - cubs_data['RA']

    fig, ax = plt.subplots()
    ax.plot(cubs_data['yearID'], cubs_data['WinningPercentage'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Winning Percentage')
    ax.set_title('Chicago Cubs Performance: Winning Percentage Over Time')

    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(cubs_data['yearID'], cubs_data['RunDifferential'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Run Differential')
    ax.set_title('Chicago Cubs Performance: Run Differential Over Time')

    st.pyplot(fig)

def visualize_yearly_stats(data, start_year, end_year):
    """
    Visualize the statistics for each year.
    """
    year = st.sidebar.selectbox("Select Year", data['yearID'].unique())
    league_options = ["All", "AL", "NL"]
    division_options = ["All", "C", "E", "W"]

    league = st.sidebar.selectbox("Select League", league_options)
    division = st.sidebar.selectbox("Select Division", division_options)

    # Filter data for the selected year
    year_data = data[data['yearID'] == year]

    # Filter data based on league and division
    if league != "All":
        year_data = year_data[year_data['lgID'] == league]
    if division != "All":
        year_data = year_data[year_data['divID'] == division]

    # Display the statistics for the selected year, league, and division
    st.write(f"Year: {year}, League: {league}, Division: {division}")
    st.write(year_data)
        
def visualize_run_differential(teams_data):
    """
    Visualize the run differential of MLB teams over time.
    """
    teams_data['RunDifferential'] = teams_data['R'] - teams_data['RA']

    fig, ax = plt.subplots()
    ax.plot(teams_data['yearID'], teams_data['RunDifferential'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Run Differential')
    ax.set_title('Team Performance: Run Differential Over Time')

    st.pyplot(fig)
    
def visualize_bryce_harper_stats(player_stats):
    """
    Visualize Bryce Harper's stats over the 2012-2022 seasons and predict future performance .
    """
    next_years = 4  # Number of years to predict
    # Prepare the data for prediction
    X = player_stats[['yearID']].values
    y_hr = player_stats['HR'].values
    y_rbi = player_stats['RBI'].values

    # Create linear regression models for home runs and RBIs
    model_hr = LinearRegression()
    model_hr.fit(X, y_hr)
    model_rbi = LinearRegression()
    model_rbi.fit(X, y_rbi)
    
    # Predict future stats for the specified number of years
    future_years = player_stats['yearID'].max() + 1 + range(next_years)
    future_stats_hr = model_hr.predict(future_years.reshape(-1, 1))
    future_stats_rbi = model_rbi.predict(future_years.reshape(-1, 1))

    # Display the predicted stats for the next years
    st.subheader("Bryce Harper: Home Runs and RBIs from 2012-2026 (Prediction)")
    fig, ax = plt.subplots()
    ax.plot(player_stats['yearID'], player_stats['HR'], marker='o', label='Home Runs')
    ax.plot(player_stats['yearID'], player_stats['RBI'], marker='o', label='RBIs')
    ax.plot(future_years, future_stats_hr, linestyle='--', marker='o', label='Predicted Home Runs')
    ax.plot(future_years, future_stats_rbi, linestyle='--', marker='o', label='Predicted RBIs')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Bryce Harper: Home Runs and RBIs from 2012-2026 (Prediction)')
    ax.legend()

    st.pyplot(fig)
    
    # Bryce Harper's HRs, and RBIs over 10 years
    fig, ax = plt.subplots()
    ax.plot(player_stats['yearID'], player_stats['HR'], marker='o', label='Home Runs')
    ax.plot(player_stats['yearID'], player_stats['RBI'], marker='o', label='RBIs')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Bryce Harper: Home Runs and RBIs from 2012-2022')
    ax.legend()

    st.pyplot(fig)
    
def visualize_mike_trout_stats(player_stats):
    """
    Visualize Mike Trout's stats over the 2012-2022 seasons and predict future performance.
    """
    next_years = 4  # Number of years to predict

    # Prepare the data for prediction
    X = player_stats[['yearID']].values
    y_hr = player_stats['HR'].values
    y_rbi = player_stats['RBI'].values

    # Create linear regression models for home runs and RBIs
    model_hr = LinearRegression()
    model_hr.fit(X, y_hr)

    model_rbi = LinearRegression()
    model_rbi.fit(X, y_rbi)

    # Predict future stats for the specified number of years
    future_years = player_stats['yearID'].max() + 1 + range(next_years)
    future_stats_hr = model_hr.predict(future_years.reshape(-1, 1))
    future_stats_rbi = model_rbi.predict(future_years.reshape(-1, 1))

    # Display the predicted stats for the next years
    st.subheader("Mike Trout: Home Runs and RBIs Over from 2012-2026 (Prediction)")
    fig, ax = plt.subplots()
    ax.plot(player_stats['yearID'], player_stats['HR'], marker='o', label='Home Runs')
    ax.plot(player_stats['yearID'], player_stats['RBI'], marker='o', label='RBIs')
    ax.plot(future_years, future_stats_hr, linestyle='--', marker='o', label='Predicted Home Runs')
    ax.plot(future_years, future_stats_rbi, linestyle='--', marker='o', label='Predicted RBIs')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Mike Trout: Home Runs and RBIs from 2012-2026 (Prediction)')
    ax.legend()

    # Mike Trouts HR, and RBIs over 10 years
    st.pyplot(fig)
    fig, ax = plt.subplots()
    ax.plot(player_stats['yearID'], player_stats['HR'], marker='o', label='Home Runs')
    ax.plot(player_stats['yearID'], player_stats['RBI'], marker='o', label='RBIs')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Mike Trout: Home Runs and RBIs Over the Years')
    ax.legend()

    st.pyplot(fig)

def open_dashboard():
    st.title("Baseball Data Analysis")

    # Read the data
    batting_data = read_data(r'C:\Users\Josh\Downloads\lahmanbaseball\Batting.csv')
    pitching_data = read_data(r'C:\Users\Josh\Downloads\lahmanbaseball\Pitching.csv')
    teams_data = read_data(r'C:\Users\Josh\Downloads\lahmanbaseball\Teams.csv')

    # Filter data for the past 10 years (2012-2022)
    start_year = 2012
    end_year = 2022

    batting_data = filter_data(batting_data, start_year, end_year)
    pitching_data = filter_data(pitching_data, start_year, end_year)
    teams_data = filter_data(teams_data, start_year, end_year)

    # Sort data by yearID
    batting_data = sort_data(batting_data, 'yearID')
    pitching_data = sort_data(pitching_data, 'yearID')
    teams_data = sort_data(teams_data, 'yearID')

    # Create sidebar selection
    selected_option = st.sidebar.selectbox(
        "Select Visualization",
        ("Yearly Stats", "Home Runs", "Average ERA", "Average Batting Average", "Player Stats", "Cubs Performance", "Team Performance")
    )

    if selected_option == "Team Performance":
        st.header("Team Performance")
        st.subheader("Winning Percentage Over Time")
        visualize_team_performance(teams_data)
        st.subheader("Run Differential Over Time")
        visualize_run_differential(teams_data)
    elif selected_option == "Home Runs":
        st.header("Home Runs")
        st.subheader("Home Runs Over Time")
        visualize_home_runs(batting_data)
    elif selected_option == "Average ERA":
        st.header("Average ERA")
        st.subheader("Average ERA Over Time")
        visualize_average_era(pitching_data)
    elif selected_option == "Average Batting Average":
        st.header("Average Batting Average")
        st.subheader("Average Batting Average Over Time")
        visualize_average_batting_average(batting_data)
    elif selected_option == "Player Stats":
        st.header("Player Stats")
        st.subheader("Mike Trout vs Bryce Harper: Home Runs and RBIs Over the 2012-2022 Season")
        # Get player stats for Mike Trout
        mike_trout_stats = get_player_batting_stats('troutmi01', batting_data)
        visualize_mike_trout_stats(mike_trout_stats)
        # Get player stats for Bryce Harper
        bryce_harper_stats = get_player_batting_stats('harpebr03', batting_data)
        visualize_bryce_harper_stats(bryce_harper_stats)
    elif selected_option == "Cubs Performance":
        st.header("Cubs Performance")
        st.subheader("Winning Percentage Over Time")
        cubs_stats = teams_data[teams_data['teamID'] == 'CHN']  # Filter Cubs data
        visualize_cubs_performance(cubs_stats)
    elif selected_option == "Yearly Stats":
        st.header("Yearly Stats")
        visualize_yearly_stats(teams_data, start_year, end_year)


# Run the dashboard
open_dashboard()
