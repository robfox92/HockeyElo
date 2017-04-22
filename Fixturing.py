
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import networkx as nx


# In[2]:

processresults_ladies = True
processresults_mixed = True

verbose = True


# In[90]:

if verbose: print "Getting results from google sheet"

teams_url = 'https://docs.google.com/spreadsheets/d/1Oo7fzq3nJP1HxfxTYfSiig0t2Gjt19yA4Agd85CbtHU/export?format=xlsx&id=1Oo7fzq3nJP1HxfxTYfSiig0t2Gjt19yA4Agd85CbtHU'
mixed_teams_2017a = pd.read_excel(teams_url,sheetname='Mens  Mixed Teams').dropna(how='all').reset_index()

mixed_teams_2017a.columns = ['Old Name', '2017a Name','Team Age','Notes']
mixed_teams = mixed_teams_2017a['2017a Name'].values
ladies_teams_2017a = pd.read_excel(teams_url,sheetname = 'Ladies',header=None)
ladies_teams = ladies_teams_2017a[0].values




data_url = 'https://docs.google.com/spreadsheets/d/1OEIBqmZ3y1bCWOkZbdKsJu6ko7OYA3PvBjlQTHrN0mI/export?format=xlsx&id=1OEIBqmZ3y1bCWOkZbdKsJu6ko7OYA3PvBjlQTHrN0mI'

# Grab the data, 
#     drop rows where there is no home team
#     sort the values by round
mixed_results = pd.read_excel(data_url,sheetname = 'Mixed-Scores').dropna(how='all')
mixed_results.sort_values(by='Round',inplace=True)
ladies_results = pd.read_excel(data_url,sheetname = 'Ladies-Scores').dropna(how='all')
ladies_results.sort_values(by='Round',inplace=True)



mixed_fixtured = pd.read_excel(data_url,sheetname = 'Mixed-Fixtured Games').dropna(how='all')

ladies_fixtured = pd.read_excel(data_url,sheetname = 'Ladies-Fixtured Games').dropna(how='all')

mixed_requests = pd.read_excel(data_url,sheetname='Mixed-Requests').dropna(how='all')
mixed_requests = mixed_requests[mixed_requests['Game Fixtured'] == 0]
ladies_requests = pd.read_excel(data_url,sheetname='Ladies-Requests').dropna(how='all')
ladies_requests = ladies_requests[ladies_requests['Game Fixtured'] == 0]

ladies_elos = {team:700 for team in ladies_teams}

mixed_elos_df = pd.read_excel(data_url,sheetname = 'Mixed-Starting Elos',index='TEAM NAME').dropna(how='all')
mixed_elos_df.index = mixed_elos_df['TEAM NAME']
del mixed_elos_df['TEAM NAME']
mixed_elos = {team:mixed_elos_df.loc[team,'STARTING ELO'] for team in mixed_elos_df.index}
mixed_K = {team:mixed_elos_df.loc[team,'K Value'] for team in mixed_elos_df.index}


# In[4]:

# Process mixed results
if processresults_mixed:
    if verbose:
        print "Processing mixed results to update ratings"
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




# Process ladies result
if processresults_ladies:
    if verbose:
        print "Processing ladies results to update ratings"
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


# In[5]:

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



if verbose:
    print "Creating graph to fixture mixed league"
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
                
                # grab the date the request was made
                indexes = mixed_requests[mixed_requests[['Game Code','Code 2']] == code1].dropna(how='all').index
                i = indexes[0]
                mixed_requests.loc[i,'Timestamp']
                #Calculate the difference between when the request was first made and the current time
                now = pd.tslib.Timestamp.now()
                request_first_made = mixed_requests.loc[i,'Timestamp']
                diff = request_first_made - now
                # Add the difference/70.0 to the weight
                # This will help priotitise old requests
                # Need to use 70.0 to ensure float division
                w += abs(diff.days)/70.0
                
            # Decrease the weight if the game has occurred
            if code1 in mixed_fixtured['Game Code'].unique() or code2 in mixed_fixtured['Game Code'].unique():
                w -= 10
            teams_graph.add_edge(teamA,teamB,weight=w)
    
if verbose:
    print "Finding optimal pairings"
mixed_pairs = nx.max_weight_matching(teams_graph)


if verbose:
    print "Determining home and away teams"
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




roundno = raw_input("What round are you fixturing?\n")
roundno = int(roundno)
fname = 'Round %i Mixed Fixtures 2017a.csv' % roundno
fixtured.to_csv(fname)
print "Fixtures written to csv"


# In[86]:

if roundno%2 != 0:
    print "Fixturing ladies league"
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
                    
                    # grab the date the request was made
                    indexes = ladies_requests[ladies_requests[['Game Code','Code 2']] == code1].dropna(how='all').index
                    i = indexes[0]
                    ladies_requests.loc[i,'Timestamp']
                    #Calculate the difference between when the request was first made and the current time
                    now = pd.tslib.Timestamp.now()
                    request_first_made = ladies_requests.loc[i,'Timestamp']
                    diff = request_first_made - now
                    # Add the difference/70.0 to the weight
                    # This will help priotitise old requests
                    # Need to use 70.0 to ensure float division
                    w += abs(diff.days)/70.0
                
                # Decrease the weight if the game has occurred
                if code1 in ladies_fixtured['Game Code'].unique() or code2 in ladies_fixtured['Game Code'].unique():
                    w -= 10
                ladies_graph.add_edge(teamA,teamB,weight=w)
                
    ladies_pairs = nx.max_weight_matching(ladies_graph)
    ladies_1_bye = ladies_pairs['Bye']


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



    
    fname = 'Round %i Ladies Fixtures 2017a.csv' % int(roundno)
    r_1 = fixtured.copy()
    #fixtured.to_csv(fname)
    print "Ladies fixturing complete"
    # Need to create a new graph and fixture the second round (roundno+1) here
    #  Need to account for games in the (roundno)th round
    fixtured_current = ladies_fixtured['Game Code'].append(fixtured['Game Code']).reset_index()
    byeteam1 = ladies_pairs['Bye']
    
    ladies_graph = nx.Graph()
    # Add nodes from the list of teams (the teams that are in the elos dict)
    ladies_graph.add_nodes_from(ladies_elos.keys())

    for teamA in ladies_graph.nodes():
        for teamB in ladies_graph.nodes():
            if teamA != teamB and byeteam1 not in [teamA,teamB]:
                
                # Create the weight for the edge between teamA and teamB, a function of the expected outcome
                # Larger is better
                w = getScaledOutcome(teamA,teamB,'Ladies')
                if 'Bye' not in [teamA,teamB]:
                    code1 = teamA + ' vs ' + teamB
                    code2 = teamB + ' vs ' + teamA
                elif teamA == 'Bye':
                    code1 = byeteam1 + ' vs ' + teamB
                    code2 = teamB + ' vs ' + byeteam1
                elif teamB == 'Bye':
                    code1 = teamA + ' vs ' + byeteam1
                    code2 = byeteam1 + ' vs ' + teamA
                
                # Increase the weight if the game has been requested
                if code1 in ladies_requests['Game Code'].unique() or code2 in ladies_requests['Game Code'].unique():
                    w += 2
                    
                # Decrease the weight if the game has occurred
                if (code1 in ladies_fixtured['Game Code'].unique() or 
                    code2 in ladies_fixtured['Game Code'].unique() or
                    code1 in fixtured_current['Game Code'].values or
                    code2 in fixtured_current['Game Code'].values):
                    w -= 10
                    
                
                
                
                ladies_graph.add_edge(teamA,teamB,weight=w)
                
    ladies_pairs = nx.max_weight_matching(ladies_graph)
    ladies_2_bye = ladies_pairs['Bye']
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



    r_2 = fixtured.copy()
    fname = 'Round %i Ladies Fixtures 2017a.csv' % (int(roundno) + 1)
    fixtured.to_csv(fname)
    f_1 = 'Round %i Ladies Fixtures 2017a.csv'% int(roundno)
    f_2 = 'Round %i Ladies Fixtures 2017a.csv'% (int(roundno)+1)
    #r_1 = pd.read_csv(f_1)
    #r_2 = pd.read_csv(f_2)
    
    # Combine the two rounds worth of results
    r_12 = r_1.append(r_2)
    # Drop the bye games - bye should never be the Away team
    #     (as their name never appears in the list of home teams)
    # but it is included for the off chance that it is
    r_12 = r_12[r_12['Home Team'] != 'Bye']
    r_12 = r_12[r_12['Away Team'] != 'Bye'].reset_index()
    r_12 = r_12[['Home Team','Away Team','Game Code']]
    
    row = len(r_12)
    r_12.loc[row,'Home Team'] = ladies_1_bye
    r_12.loc[row,'Away Team'] = ladies_2_bye
    r_12.loc[row,'Game Code'] = ladies_1_bye + ' vs ' + ladies_2_bye
    
    missing = []
    for t in ladies_teams[ladies_teams != 'Bye']:
        h = len(r_12[r_12['Home Team'] == t])
        a = len(r_12[r_12['Away Team'] == t])
        tot = h+a
        
        if tot != 2:
            missing.append(t)
    
    if len(missing) == 2:
        row = len(r_12)
        r_12.loc[row,'Home Team'] = missing[0]
        r_12.loc[row,'Away Team'] = missing[1]
        r_12.loc[row,'Game Code'] = missing[0] + ' vs ' + missing[1]
    
    rn = '%s-%i' %(roundno,int(roundno)+1)
    fname  = 'Round %s Ladies Fixtures 2017a.csv' % rn
    
    r_12.to_csv(fname)
    
    print "Ladies fixturing complete"
print "Fixturing complete"

elos_fname = "Mixed Elos at round %i.csv" %int(roundno)
with open(elos_fname,'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in mixed_elos.items()]




