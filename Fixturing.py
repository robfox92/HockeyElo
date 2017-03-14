
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import networkx as nx

processresults_ladies = False
processresults_mixed = True


# In[2]:

teams_url = 'https://docs.google.com/spreadsheets/d/1Oo7fzq3nJP1HxfxTYfSiig0t2Gjt19yA4Agd85CbtHU/export?format=xlsx&id=1Oo7fzq3nJP1HxfxTYfSiig0t2Gjt19yA4Agd85CbtHU'
mixed_teams_2017a = pd.read_excel(teams_url,sheetname='Mens  Mixed Teams').reset_index()

mixed_teams_2017a.columns = ['Old Name', '2017a Name','Team Age','Notes']
mixed_teams = mixed_teams_2017a['2017a Name'].values
ladies_teams_2017a = pd.read_excel(teams_url,sheetname = 'Ladies',header=None)
ladies_teams = ladies_teams_2017a[0].values


# In[3]:

data_url = 'https://docs.google.com/spreadsheets/d/1OEIBqmZ3y1bCWOkZbdKsJu6ko7OYA3PvBjlQTHrN0mI/export?format=xlsx&id=1OEIBqmZ3y1bCWOkZbdKsJu6ko7OYA3PvBjlQTHrN0mI'

mixed_results = pd.read_excel(data_url,sheetname = 'Mixed-Scores')
ladies_results = pd.read_excel(data_url,sheetname = 'Ladies-Scores')

mixed_fixtured = pd.read_excel(data_url,sheetname = 'Mixed-Fixtured Games')
ladies_fixtured = pd.read_excel(data_url,sheetname = 'Ladies-Fixtured Games')

mixed_requests = pd.read_excel(data_url,sheetname='Mixed-Requests')
ladies_requests = pd.read_excel(data_url,sheetname='Ladies-Requests')

ladies_elos = {team:700 for team in ladies_teams}

mixed_elos_df = pd.read_excel(data_url,sheetname = 'Mixed-Starting Elos',index='TEAM NAME')
mixed_elos_df.index = mixed_elos_df['TEAM NAME']
del mixed_elos_df['TEAM NAME']
mixed_elos = {team:mixed_elos_df.loc[team,'STARTING ELO'] for team in mixed_elos_df.index}
mixed_K = {team:mixed_elos_df.loc[team,'K Value'] for team in mixed_elos_df.index}


# In[5]:

# Process mixed results
if processresults_mixed:
    for game in range(len(mixed_results)):
        
      
        
        # Grab data from the results sheet
        home_team = mixed_results.loc[game,'Home Team']
        away_team = mixed_results.loc[game,'Away Team']
        
        home_score = mixed_results.loc[game,'Home Score']
        away_score = mixed_results.loc[game,'Away Score']
        
        # Cast as float to ensure float division later
        total_score = float(home_score + away_score)
        
        home_score_perc = home_score / total_score
        away_score_perc = away_score / total_score
        
        # Grab elos from the elos dict
        home_elo = mixed_elos[home_team]
        away_elo = mixed_elos[away_team]
        home_K = mixed_K[home_team]
        away_K = mixed_K[away_team]
        
        # Calculate expected game outcome
        home_expected = 1 / (1 + 10 ** -((home_elo - away_elo) / 400))
        away_expected = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))
        
        # Calculate updated elos
        home_elo_new = home_elo + home_K * (home_score_perc - home_expected)
        away_elo_new = away_elo + away_K * (away_score_perc - away_expected)
        
        # Ensure winning teams don't lose elo
        if home_score > away_score:
            home_elo_new = max(home_elo_new,home_elo)
        if away_score > home_score:
            away_elo_new = max(away_elo_new,away_elo)   
        
        # Update the elos dict
        mixed_elos.update({home_team:home_elo_new, away_team:away_elo_new})


# In[5]:

# Process ladies result
if processresults_ladies:
    for game in range(len(ladies_results)):
        
        # Temp - need to get K values from somewhere
        home_K = 50
        away_K = 50
        
        # Grab data from the results sheet
        home_team = ladies_results.loc[game,'Home Team']
        away_team = ladies_results.loc[game,'Away Team']
        
        home_score = ladies_results.loc[game,'Home Score']
        away_score = ladies_results.loc[game,'Away Score']
        
        # Cast as float to ensure float division later
        total_score = float(home_score + away_score)
        
        home_score_perc = home_score / total_score
        away_score_perc = away_score / total_score
        
        # Grab elos from the elos dict
        home_elo = ladies_elos[home_team]
        away_elo = ladies_elos[away_team]
        
        # Calculate expected game outcome
        home_expected = 1 / (1 + 10 ** -((home_elo - away_elo) / 400))
        away_expected = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))
        
        # Calculate updated elos
        home_elo_new = home_elo + home_K * (home_score_perc - home_expected)
        away_elo_new = away_elo + away_K * (away_score_perc - away_expected)
        
        # Ensure winning teams don't lose elo
        if home_score > away_score:
            home_elo_new = max(home_elo_new,home_elo)
        if away_score > home_score:
            away_elo_new = max(away_elo_new,away_elo)   
        
        # Update the elos dict
        ladies_elos.update({home_team:home_elo_new, away_team:away_elo_new})


# In[8]:

def getScaledOutcome(teamA, teamB,league):
    if league == 'Mixed':
        eloA = mixed_elos[teamA]
        eloB = mixed_elos[teamB]
        outcome = 1 / (1 + 10 ** -((eloA - eloB) / 400))
        out = 1-abs(outcome - 0.5)
    elif league == 'Ladies':
        eloA = ladies_elos[teamA]
        eloB = ladies_elos[teamB]
        outcome = 1 / (1 + 10 ** -((eloA - eloB) / 400))
        out = 1-abs(outcome - 0.5)  
    return out


# In[11]:

# Fixture mixed teams
# Create empty graph
teams_graph = nx.Graph()
# Add nodes from the list of teams (the teams that are in the elos dict)
teams_graph.add_nodes_from(mixed_elos.keys())

for teamA in teams_graph.nodes():
    for teamB in teams_graph.nodes():
        if teamA != teamB:
            # Create the weight for the edge between teamA and teamB, a function of the expected outcome
            # Larger is better
            w = getScaledOutcome(teamA,teamB,'Mixed')
            code1 = teamA + ' vs ' + teamB
            code2 = teamB + ' vs ' + teamA
            
            # Increase the weight if the game has been requested
            if code1 in mixed_requests['Game Code'].unique() or code2 in mixed_requests['Game Code'].unique():
                w += 2
                
            # Decrease the weight if the game has occurred
            if code1 in mixed_fixtured['Game Code'].unique() or code2 in mixed_fixtured['Game Code'].unique():
                w -= 10
            teams_graph.add_edge(teamA,teamB,weight=w)
            
mixed_pairs = nx.max_weight_matching(teams_graph)


# In[23]:

# Determine who is playing at home
hometeams = mixed_fixtured['Home Team'].values
fixtured = pd.DataFrame(columns = ['Home Team','Away Team','Game Code'])

r = 0
for teamA in mixed_pairs.keys():
    teamB = mixed_pairs[teamA]
    
    # Determine number of games played at home
    homeA = len(hometeams[hometeams == teamA])
    homeB = len(hometeams[hometeams == teamB])
    
    # Work out who's played more games
    if homeB > homeA:
        hometeam = teamA
        awayteam = teamB
    else:
        hometeam = teamB
        awayteam = teamA
    
    # Write to df
    # Ensuring that the game hasn't been recorded
    if (hometeam not in fixtured['Home Team'].unique() and 
        hometeam not in fixtured['Away Team'].unique() and
        awayteam not in fixtured['Home Team'].unique() and 
        awayteam not in fixtured['Away Team'].unique()):
        
        fixtured.loc[r,'Home Team'] = hometeam
        fixtured.loc[r,'Away Team'] = awayteam
        fixtured.loc[r,'Game Code'] = hometeam + " vs " + awayteam
        # Increment the row to write to
        r += 1


# In[25]:

roundno = raw_input("What round are you fixturing?\n")
fname = 'Round %i Mixed Fixtures 2017a.csv' % int(roundno)
fixtured.to_csv(fname)


# In[29]:

# Fixture mixed teams
# Create empty graph
ladies_graph = nx.Graph()
# Add nodes from the list of teams (the teams that are in the elos dict)
ladies_graph.add_nodes_from(ladies_elos.keys())

for teamA in ladies_graph.nodes():
    for teamB in ladies_graph.nodes():
        if teamA != teamB:
            # Create the weight for the edge between teamA and teamB, a function of the expected outcome
            # Larger is better
            w = getScaledOutcome(teamA,teamB,'Ladies')
            code1 = teamA + ' vs ' + teamB
            code2 = teamB + ' vs ' + teamA
            
            # Increase the weight if the game has been requested
            if code1 in ladies_requests['Game Code'].unique() or code2 in ladies_requests['Game Code'].unique():
                w += 2
                
            # Decrease the weight if the game has occurred
            if code1 in ladies_fixtured['Game Code'].unique() or code2 in ladies_fixtured['Game Code'].unique():
                w -= 10
            ladies_graph.add_edge(teamA,teamB,weight=w)
            
ladies_pairs = nx.max_weight_matching(ladies_graph)


# In[33]:

# Determine who is playing at home
hometeams = ladies_fixtured['Home Team'].values
fixtured = pd.DataFrame(columns = ['Home Team','Away Team','Game Code'])

r = 0
for teamA in ladies_pairs.keys():
    teamB = ladies_pairs[teamA]
    
    # Determine number of games played at home
    homeA = len(hometeams[hometeams == teamA])
    homeB = len(hometeams[hometeams == teamB])
    
    # Work out who's played more games
    if homeB > homeA:
        hometeam = teamA
        awayteam = teamB
    else:
        hometeam = teamB
        awayteam = teamA
    
    # Write to df
    # Ensuring that the game hasn't been recorded
    if (hometeam not in fixtured['Home Team'].unique() and 
        hometeam not in fixtured['Away Team'].unique() and
        awayteam not in fixtured['Home Team'].unique() and 
        awayteam not in fixtured['Away Team'].unique()):
        
        fixtured.loc[r,'Home Team'] = hometeam
        fixtured.loc[r,'Away Team'] = awayteam
        fixtured.loc[r,'Game Code'] = hometeam + " vs " + awayteam
        # Increment the row to write to
        r += 1


# In[35]:

fname = 'Round %i Ladies Fixtures 2017a.csv' % int(roundno)
fixtured.to_csv(fname)


# In[ ]:



