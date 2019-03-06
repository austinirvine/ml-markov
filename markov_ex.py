import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pydot as pdot
#%matplotlib inline

#this gives the probability of what's possible

states = ['Hi', 'Bye', 'Here']
pi = [0.35, 0.35, 0.3]

state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

# Creationg of the transition matrix
# Equal transition probability matrix of changing states give a state
# matrix is size (M x M)  where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.4, 0.2, 0.4]
q_df.loc[states[1]] = [0.45, 0.45, 0.1]
q_df.loc[states[2]] = [0.45, 0.25, 0.3]

print(q_df)

q = q_df.values

print('\n', q, q.shape, '\n')
print(q_df.sum(axis=1))

from pprint import pprint

# Create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges

edges_wts = _get_markov_edges(q_df)
pprint(edges_wts)

# Now Create A Graph
# Use nx.MultiDiGraphy() to model the directed graphy

G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states) #correspond to states
print(f'Nodes:\n{G.nodes()}\n')

# edges represent transtion probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
print(f'Edges:')
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)

# Create edge labels for jupyter plot but is not necessary
# ...
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'say_hi_markov.dot')


# What if we want want to know the probility of liking people
# saying hi or bye
# create state space and initial state probabilities

hidden_states = ['healthy', 'sick']
pi = [0.5, 0.5]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())

# Create hidden transition matrxi
# a or alpha
# = transition probability matrix of changing states given a state
# matrxi is size (M x M) where M is number of states

a_df = pd.DataFrame(columns = hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))

print("KEEP MOVING TO SOLVE HIDDEN PROBLEM")

# M x O; M is number of hidden states, O is number of possible states
# Create matrix of observation (emission) probabilities
# b or beta = obsv probab. given state
# matrix is size M x O where M and O are respectively state above

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

print(b_df)

b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))

# Now Create Graph With Edges
hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)

# CREATE Graphy Object
G = nx.MultiDiGraph()

# Nodes correspond to states
G.add_nodes_from(hidden_states)
print(f'Nodes:\n{G.nodes()}\n')

# Edges rep hidden probabi.
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

#Edges rep emission probabi.s
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

print(f'Edges:')
pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)

# Create edge labels for jupyter plot
emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'hi_bye_hidden_markov.dot')

# Observations of Behavior via sequence (give list of words etc...)
# Observations are encoded numerically

obs_map = {'Hi':0, 'Bye':1, 'Here':2}
obs = np.array([1,1,2,1,0,1,2,1,0,2,2,0,1,0,1])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print( pd.DataFrame(np.column_stack([obs, obs_seq]),
                columns=['Obs_code', 'Obs_seq']) )

###
#High level, the Viterbi algorithm increments over each 
#time step, finding the maximum probability of any path
# that gets to state i at time t, that also has the 
#correct observations for the sequence up to time t.
###

# DEFINE VITERBI ALGORITHM FOR SHORTEST PATH
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
# and: https://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

def verterbi(pi, a, b, obs):
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Random Walk Forward\n')
    # Forward Algorithm Extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s,t]))

    # Find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        a = int(path[t+1])
        b = [t+1]
        path[t] = phi[a, b]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi

path, delta, phi = verterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

# Done with Verterbi
print("Verterbi Complete")

state_map = {0:'healthy', 1:'sick'}
state_path = [state_map[v] for v in path]

(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))
