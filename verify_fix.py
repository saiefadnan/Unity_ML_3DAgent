import json

with open('output/Individual_Testing_Analysis/2DMultiAgent_Test_Results/metrics.json') as f:
    data = json.load(f)
    tvr = data['numeric_metrics']['TeamVictimsRescued']
    tv = data['numeric_metrics']['TotalVictims']
    
    print(f'TeamVictimsRescued - Mean: {tvr["mean"]}, Max: {tvr["max"]}, Min: {tvr["min"]}')
    print(f'TotalVictims - Mean: {tv["mean"]}, Max: {tv["max"]}, Min: {tv["min"]}')
    
    if tvr['max'] == tv['max']:
        print('\n✅ FIX VERIFIED: TeamVictimsRescued max now matches TotalVictims max!')
    else:
        print(f'\n❌ ISSUE: TeamVictimsRescued max ({tvr["max"]}) still exceeds TotalVictims max ({tv["max"]})')
