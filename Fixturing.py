import pandas as pd
import numpy as np
import networkx as nx
import sys
import random

#reload(sys)
#sys.setdefaultencoding('utf8')


if len(sys.argv)<2:
    print("What round are you fixturing?")
    roundNumber = input()
else:
    roundNumber = str(sys.argv[1])
    print("Fixturing round ",roundNumber)

def getExpectedOutcome(eloA,eloB):
    # Standard function for finding the expected outcome in an 
    #     Elo ranking system
    if eloA == eloB:
        out = 0.5
    else:
        out = 1 / (1 + 10 ** -((eloA - eloB) / 400))
    return out

def getScaledOutcome(eloA,eloB):
    # Returns a scaled outcome for a game
    # Bounded in [0,1]
    # Higher means the elos are closer
    outcome = getExpectedOutcome(eloA,eloB)
    out = 2*(0.5-abs(outcome - 0.5))
    return out

def updateElos(initialElos,teamKvalues, results):
    # For each game in results, update the initialElos
    # Return the updated elo dictionary
    teamElos = initialElos.copy()
    if len(results) > 0:
        for game in range(len(results)):

            homeTeam = results.loc[game,'Home Team']
            awayTeam = results.loc[game,'Away Team']

            homeK = teamKvalues[homeTeam]
            awayK = teamKvalues[awayTeam]

            homeScore = results.loc[game,'Home Score']
            awayScore = results.loc[game,'Away Score']
            totalScore = float(homeScore + awayScore)       # Cast as float to ensure float division later
            homeScorePerc = homeScore / totalScore
            awayScorePerc = awayScore / totalScore

            homeElo = teamElos[homeTeam]
            awayElo = teamElos[awayTeam]

            homeExpected = getExpectedOutcome(homeElo,awayElo)
            awayExpected = getExpectedOutcome(awayElo,homeElo)

            homeEloNew = homeElo + homeK * (homeScorePerc - homeExpected)
            awayEloNew = awayElo + awayK * (awayScorePerc - awayExpected)

            # Ensure that winning teams do not lose elo
            if homeScore > awayScore:
                homeEloNew = max(homeElo,homeEloNew)
            if awayScore > homeScore:
                awayEloNew = max(awayElo,awayEloNew)

            # Update the dictionary with new elos
            newElos = {homeTeam:homeEloNew, awayTeam:awayEloNew}
            teamElos.update(newElos)

    # Output the new elos
    return teamElos

def findBestPairings(elos,fixtured,requested,homeGames,byeTeam):
    # Function to find the best pairings for a given list of elos and a round number
    # Left the if True statement in because I cbf unindenting the lines
    # Maybe I'll do it later? Probably not.
    if True:    
        # Create graph from elos.keys()
        #  Each team will be a node and the game 'quality' will be the weight
        graph = nx.Graph()
        graph.add_nodes_from(elos.keys())

        for teamA in graph.nodes():
            for teamB in graph.nodes():

                # If the teams are not the same and an edge has not been made yet for the two teams
                if teamA != teamB and not graph.has_edge(teamA,teamB):
                    w = 50
                    if byeTeam is not None:
                        if teamA == 'Bye':
                            teamA = byeTeam
                        if teamB == 'Bye':
                            teamB = byeTeam
                    
                    
                    # Create unique game codes
                    code1 = teamA + " vs " + teamB
                    code2 = teamB + " vs " + teamA
                    
                    
                    # Grab the respective elos from the dict
                    eloA = elos[teamA]
                    eloB = elos[teamB]

                    # Create an initial weight for the graph
                    w += getScaledOutcome(eloA,eloB)

                    # Increase the weight if the game has been requested
                    if code1 in requested['Game Code'].unique() or code2 in requested['Game Code'].unique():
                        w += 2
                        indexes = requested[['Game Code','Code 2']][requested[['Game Code','Code 2']]==code1].dropna(how='all').index
                        
                        firstRequestDate = requested.loc[indexes[0],'Timestamp']
                        now = pd.tslib.Timestamp.now()
                        diff = firstRequestDate - now

                        # Add 1/70 to the weight for each day since the request was made
                        # This will slightly prioritise older requests
                        # At a rate of 0.1/week which seems fair?
                        # Just made up the number because it sounded nice tbh
                        w += abs(diff.days)/70.0

                    # Decrease the weight if the game has been fixtured
                    # Need to decrement as this will correctly handle things if
                    #     teams must play each other twice in a season

                    if code1 in fixtured:
                        count = len(fixtured[fixtured == code1])
                        w -= 10*count
                    if code2 in fixtured:
                        count = len(fixtured[fixtured == code2])
                        w -= 10*count

                    # Make sure the weight is non-negative, as negative weights make the pairing algorithm sad
                    w = max(0,w)

                    
                    graph.add_edge(teamA,teamB,weight=w)
        # Find the optimal pairing for the graph of teams
        optimalPairing = nx.max_weight_matching(graph)
        newFixture = pd.DataFrame(columns = ['Home Team','Away Team','Game Code'])
        
        # r stores the row to write to
        # Probably a poor way to do it but it works I guess
        r = 0
        for teamA in optimalPairing.keys():
            teamB = optimalPairing[teamA]

            if (teamA not in newFixture['Home Team'].unique() and
                teamB not in newFixture['Home Team'].unique() and
                teamA not in newFixture['Away Team'].unique() and
                teamB not in newFixture['Away Team'].unique()):

                if len(homeGames[homeGames == teamA]) <= len(homeGames[homeGames == teamB]):
                    homeTeam = teamA
                    awayTeam = teamB
                else:
                    homeTeam = teamB
                    awayTeam = teamA

                newFixture.loc[r,'Home Team'] = homeTeam
                newFixture.loc[r,'Away Team'] = awayTeam
                newFixture.loc[r,'Game Code'] = homeTeam + " vs " + awayTeam
                r += 1
    return newFixture

# Get the data from the google sheet
#teams_url = 'https://docs.google.com/spreadsheets/d/1UNFeLJQsP08k9QJxIfnmGqvf3W-nqxGpeUdlvpFHEYU/export?format=xlsx&id=1UNFeLJQsP08k9QJxIfnmGqvf3W-nqxGpeUdlvpFHEYU'
data_url = 'https://docs.google.com/spreadsheets/d/10RYpl2cOuze8CfiLjb1NiXsJlbWyrm-k_q33euzIdN8/export?format=xlsx&id=10RYpl2cOuze8CfiLjb1NiXsJlbWyrm-k_q33euzIdN8'

mixed_teams_df = pd.read_excel(data_url,sheetname='Mixed-Starting Elos').dropna(how='all').reset_index()
mixed_teams = mixed_teams_df['TEAM NAME'].values

ladies_teams_df = pd.read_excel(data_url,sheetname = 'STANLEY LADDER').dropna(how='all').reset_index()
ladies_teams = ladies_teams_df['Team'].values




mixed_results = pd.read_excel(data_url,sheetname = 'Mixed-Scores').dropna(how='all')
mixed_results.sort_values(by='Round',inplace=True)
ladies_results = pd.read_excel(data_url,sheetname = 'Ladies-Scores').dropna(how='all')
ladies_results.sort_values(by='Round',inplace=True)



mixed_fixtured = pd.read_excel(data_url,sheetname = 'Mixed-Fixtured Games').dropna(how='all')
mixed_fixtured_list = mixed_fixtured['Game Code'].unique()
ladies_fixtured = pd.read_excel(data_url,sheetname = 'Ladies-Fixtured Games').dropna(how='all')
ladies_fixtured_list = ladies_fixtured['Game Code'].unique()

mixed_requests = pd.read_excel(data_url,sheetname='Mixed-Requests').dropna(how='all')
mixed_requests = mixed_requests[mixed_requests['Game Fixtured'] == 0]
ladies_requests = pd.read_excel(data_url,sheetname='Ladies-Requests').dropna(how='all')
# Don't only include fixtured games as there will be rematches in the ladies league
#ladies_requests = ladies_requests[ladies_requests['Game Fixtured'] == 0]

ladies_elos = {team:700 for team in ladies_teams}
ladies_K = {team:75 for team in ladies_teams}

mixed_teams_df.index = mixed_teams_df['TEAM NAME']
mixed_elos = {team:mixed_teams_df.loc[team,'STARTING ELO'] for team in mixed_teams_df['TEAM NAME']}
mixed_K = {team:mixed_teams_df.loc[team,'K Value'] for team in mixed_teams_df['TEAM NAME']}


mixed_elos = updateElos(mixed_elos,mixed_K,mixed_results)
ladies_elos = updateElos(ladies_elos,ladies_K,ladies_results)



if len(mixed_elos.keys()) %2 == 0:
    # If there are an even number of teams, do this normally
    mixedNewFixture = findBestPairings(mixed_elos,mixed_fixtured_list,
        mixed_requests,mixed_fixtured['Home Team'].values,None)
    mixed_fname = "Round %s Mixed Fixtures 2018a.csv" %roundNumber
    mixedNewFixture.to_csv(mixed_fname)  
    elos_fname = "Mixed Elos at round %i.csv" %int(roundNumber)
    with open(elos_fname,'w') as f:
        #[f.write(u'{0},{1}\n'.format(key, value).encode('utf8)')) for key, value in mixed_elos.items()]
        for k,v in mixed_elos.items():
            f.write(str(k)+","+str(v)+"\n")

elif len(mixed_elos.keys())%2 != 0 and int(roundNumber)%2 != 0:
    # If there are an odd number of teams BUT the roundnumber is odd

    # Randomly select the elo for the bye team
    byeElo = random.choice(list(mixed_elos.values()))

    mixed_elos.update({'Bye':byeElo})
    mixedNewFixture = findBestPairings(mixed_elos,mixed_fixtured_list,
                                       mixed_requests,mixed_fixtured['Home Team'].values,None)
    
    nowRequests = mixed_requests.copy()
    nowHome = np.concatenate((mixedNewFixture['Home Team'].values,mixed_fixtured['Home Team'].values))
    nowFixtured = np.concatenate((mixedNewFixture['Game Code'].unique(),mixed_fixtured['Game Code'].unique()))
    
    r = mixedNewFixture[mixedNewFixture == 'Bye'].dropna(how='all').index[0]
    c = mixedNewFixture[mixedNewFixture == 'Bye'].dropna(how='all',axis=1).columns[0]
    if c == 'Away Team':
        c1 = 'Home Team'
    else:
        c1 = 'Away Team'
    
    byeTeam1 = mixedNewFixture.loc[r,c1]
    
    
    
    mixedNewFixture2 = findBestPairings(mixed_elos,nowFixtured,
                                         nowRequests,nowHome,byeTeam1)
    mixedNewFixture1 = mixedNewFixture.copy()
    mixedNewFixture = pd.concat([mixedNewFixture1,mixedNewFixture2])
    r = mixedNewFixture1[mixedNewFixture1=='Bye'].dropna(how='all').index[0]
    c = mixedNewFixture1[mixedNewFixture1=='Bye'].dropna(how='all',axis=1).columns[0]
    mixedNewFixture = mixedNewFixture.reset_index()[['Home Team','Away Team','Game Code']]
    byeTeam2 = ''
    for team in mixed_teams:
        homeFixtured = len(mixedNewFixture[mixedNewFixture['Home Team'] == team].values)
        awayFixtured = len(mixedNewFixture[mixedNewFixture['Away Team'] == team].values)
        totalFixtured = homeFixtured + awayFixtured
        
        if totalFixtured != 2:
            byeTeam2 = team
    mixedNewFixture.loc[r,c] = byeTeam2
    gname = mixedNewFixture.loc[r,'Home Team'] + " vs " + mixedNewFixture.loc[r,'Away Team']
    mixedNewFixture.loc[r,'Game Code'] = gname
    # Make the filename
    currentRd = int(roundNumber)
    nextRd = currentRd + 1
    rdno = '%i-%i' %(currentRd,nextRd)
    mixed_fname = "Round %s Mixed Fixtures 2018a.csv" %rdno
    mixed_elos.pop('Bye',None)
    mixedNewFixture.to_csv(mixed_fname)    
    elos_fname = "Mixed Elos at round %s.csv" %rdno
    with open(elos_fname,'w') as f:
#        [f.write(u'{0},{1}\n'.format(key, value).encode('utf8)')) for key, value in mixed_elos.items()]
        for k,v in mixed_elos.items():
            f.write(str(k)+","+str(v)+"\n")



    
ladies_elos.pop('Bye',None)

if len(ladies_elos.keys()) %2 == 0:
    ladiesNewFixture = findBestPairings(ladies_elos,ladies_fixtured_list,
        ladies_requests,ladies_fixtured['Home Team'].values,None)
    ladies_fname = "Round %s Ladies Fixtures 2018a.csv" %roundNumber
    ladiesNewFixture.to_csv(ladies_fname)

    elos_fname = "Ladies Elos at round %i.csv" %int(roundNumber)
    with open(elos_fname,'w') as f:
#        [f.write(u'{0},{1}\n'.format(key, value).encode('utf8)')) for key, value in ladies_elos.items()]
        for k,v in ladies_elos.items():
            f.write(str(k)+","+str(v)+"\n")

elif len(ladies_elos.keys())%2 != 0 and int(roundNumber)%2 != 0:
    byeElo = random.choice(list(mixed_elos.values()))

    ladies_elos.update({'Bye':byeElo})

    ladiesNewFixture = findBestPairings(ladies_elos,ladies_fixtured_list,
                                       ladies_requests,ladies_fixtured['Home Team'].values,None)
    
    nowRequests = ladies_requests.copy()
    nowHome = np.concatenate((ladiesNewFixture['Home Team'].values,ladies_fixtured['Home Team'].values))
    nowFixtured = np.concatenate((ladiesNewFixture['Game Code'].unique(),ladies_fixtured['Game Code'].unique()))
    
    r = ladiesNewFixture[ladiesNewFixture == 'Bye'].dropna(how='all').index[0]
    c = ladiesNewFixture[ladiesNewFixture == 'Bye'].dropna(how='all',axis=1).columns[0]
    if c == 'Away Team':
        c1 = 'Home Team'
    else:
        c1 = 'Away Team'
    
    byeTeam1 = ladiesNewFixture.loc[r,c1]
    
    
    
    ladiesNewFixture2 = findBestPairings(ladies_elos,nowFixtured,
                                         nowRequests,nowHome,byeTeam1)
    ladiesNewFixture1 = ladiesNewFixture.copy()
    ladiesNewFixture = pd.concat([ladiesNewFixture1,ladiesNewFixture2])
    r = ladiesNewFixture1[ladiesNewFixture1=='Bye'].dropna(how='all').index[0]
    c = ladiesNewFixture1[ladiesNewFixture1=='Bye'].dropna(how='all',axis=1).columns[0]
    ladiesNewFixture = ladiesNewFixture.reset_index()[['Home Team','Away Team','Game Code']]
    byeTeam2 = ''
    for team in ladies_teams:
        homeFixtured = len(ladiesNewFixture[ladiesNewFixture['Home Team'] == team].values)
        awayFixtured = len(ladiesNewFixture[ladiesNewFixture['Away Team'] == team].values)
        totalFixtured = homeFixtured + awayFixtured
        
        if totalFixtured != 2:
            byeTeam2 = team
    ladiesNewFixture.loc[r,c] = byeTeam2
    gname = ladiesNewFixture.loc[r,'Home Team'] + " vs " + ladiesNewFixture.loc[r,'Away Team']
    ladiesNewFixture.loc[r,'Game Code'] = gname
    # Make the filename
    currentRd = int(roundNumber)
    nextRd = currentRd + 1
    rdno = '%i-%i' %(currentRd,nextRd)
    ladies_fname = "Round %s Ladies Fixtures 2018a.csv" %rdno
    ladies_elos.pop('Bye',None)
    ladiesNewFixture.to_csv(ladies_fname)
    elos_fname = "Ladies Elos at round %s.csv" %rdno
    with open(elos_fname,'w') as f:
#        [f.write(u'{0},{1}\n'.format(key, value).encode('utf8)')) for key, value in ladies_elos.items()]
        for k,v in ladies_elos.items():
            f.write(str(k)+","+str(v)+"\n")


print("Fixturing Complete")

