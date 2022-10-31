# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:02:02 2022

@author: nikesh
"""


********************************************************************************************
********************************************Graph Node and adjacency Marix*****************
pip install evaluation
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import evaluation
G = pd.read_excel("I:/Data science/DS Project/New folder/Optimal traffic routes.xlsx")
G.info()

g = nx.Graph()
g = nx.from_pandas_edgelist(G, source = 'Origin_code', target = 'Dest_code')

print(nx.info(g))

# Degree Centrality
d = nx.degree_centrality(g)
print(d) 

b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

cluster_coeff = nx.clustering(g)
print(cluster_coeff)

pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')



B = nx.complete_graph(G)
nx.draw(B)

A = nx.to_numpy_matrix(B)
print(A)
********************************************************************************************
********************************************Auto EDA**************************************

pip install pandas-profiling
import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_excel("I:/Data science/DS Project/New folder/Optimal traffic routes.xlsx")
df.head
import matplotlib.pyplot
pip install pandas-profiling --upgrade
profile = ProfileReport(df, explorative = True)
profile.to_file("I:/Data science/DS Project/New folder/Optimal traffic routes.html")

*******************************************************************************************
######################## Girvan Newman ##################################################
from networkx.algorithms.community.centrality import girvan_newman
import networkx.algorithms.community as nx_comm
from sklearn.model_selection import train_test_split

communities = girvan_newman(g)

node_groups = []
for com in next(communities):
  node_groups.append(list(com))

print(node_groups)

color_map = []
for node in g:
    if node in node_groups[0]:
        color_map.append('blue')
    elif node in node_groups[1]:
        color_map.append('red')
    elif node in node_groups[2]:
        color_map.append('orange')
    elif node in node_groups[3]:
        color_map.append('orange')
    else: 
        color_map.append('green')  
nx.draw(g, node_color=color_map, with_labels=True)
plt.show()
nx_comm.modularity(g, node_groups)
nx_comm.partition_quality(g, node_groups)
nx_comm.coverage(g, node_groups)
nx_comm.performance(g, node_groups)

g_train,g_test = train_test_split(G,test_size = 0.2)
g2 = nx.Graph()
g2 = nx.from_pandas_edgelist(g_test, source = 'Origin_code', target = 'Dest_code')

pos2 = nx.spring_layout(g2)
nx.draw_networkx(g2, pos2, node_size = 15, node_color = 'red')
plt.show()
communities2 = girvan_newman(g2)

node_groups2 = []
for com in next(communities2):
  node_groups2.append(list(com))

print(node_groups2)

color_map2 = []
for node in g2:
    if node in node_groups2[0]:
        color_map2.append('blue')
    else: 
        color_map2.append('green')  
nx.draw(g2, node_color=color_map2, with_labels=True)
plt.show()
g2 = nx.Graph()
g2 = nx.from_pandas_edgelist(g_test, source = 'Origin_code', target = 'Dest_code')

pos2 = nx.spring_layout(g2)
nx.draw_networkx(g2, pos2, node_size = 15, node_color = 'red')
plt.show()

############## K_clique #####################
from networkx.algorithms.community import k_clique_communities

g_pos = nx.spring_layout(g)
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
cliques = list(nx.find_cliques(g))
max_clique = max(cliques, key=len)
node_color = [(0.5, 0.5, 0.5) for v in g.nodes()]
for i, v in enumerate(g.nodes()):
     if v in max_clique:
         node_color[i] = (0.5, 0.5, 0.9)
nx.draw_networkx(g, node_color=node_color, pos=g_pos)

******************************************************************************************
##community_louvain########################################################################

!pip install python-louvain
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
partition = community_louvain.best_partition(g)
pos4 = nx.spring_layout(g)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(g, pos4, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(g, pos4, alpha=0.5)
plt.show()
print("Modularity:",community_louvain.modularity(partition,g))

******************************************************************************
########################kernighan#############################################
from networkx.algorithms.community import kernighan_lin_bisection
lb = kernighan_lin_bisection(g, partition = None , max_iter = 2)
lb

color_map = []
for node in g:
    if node in lb[0]:
        color_map.append('red')
    else:
        color_map.append('green')
nx.draw(g , node_color = color_map , with_labels = True) 
plt.show()       
import networkx.algorithms.community as nx_comm
nx_comm.modularity(g, lb)
nx_comm.partition_quality(g, lb)
nx_comm.coverage(g, lb)
nx_comm.performance(g, lb)
*******************************************************************************
#############################greedy_modularity_communities###########################
from networkx.algorithms.community import greedy_modularity_communities
c = greedy_modularity_communities(g)
c
sorted(c[0])

color_map = []
for node in g:
    if node in c[0]:
        color_map.append('orange')
    elif node in c[1]:
        color_map.append('red')
    else:
        color_map.append('green')
nx.draw(g , node_color = color_map , with_labels = True)
plt.show()        
import networkx.algorithms.community as nx_comm
nx_comm.modularity(g, c)
nx_comm.partition_quality(g, c)
nx_comm.coverage(g, c)
nx_comm.performance(g, c)

#######################################################################################
# Splitting the data into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(G, test_size = 0.3, random_state = 31)
#creating the graph data
graph = nx.Graph()
train_greedy = nx.from_pandas_edgelist(train, source = 'Origin_code', target = 'Dest_code')
test_greedy= nx.from_pandas_edgelist(test, source = 'Origin_code', target = 'Dest_code')
#Graph data info
print(nx.info(train_greedy))
print(nx.info(test_greedy))
# applying Greedy modularity communities algorithm on train graph data
c_train = greedy_modularity_communities(train_greedy)
c_train
sorted(c_train[0])
# applying Greedy modularity communities algorithm on test graph data
c_test = greedy_modularity_communities(test_greedy)
c_test
sorted(c_test[0])

# Train graph data Modularity,partition_quality, coverage, performance

nx_comm.modularity(train_greedy, c_train)
nx_comm.partition_quality(train_greedy, c_train)
nx_comm.coverage(train_greedy, c_train)
nx_comm.performance(train_greedy, c_train)

# Test graph data Modularity,partition_quality, coverage, performance

nx_comm.modularity(test_greedy, c_test)
nx_comm.partition_quality(test_greedy, c_test)
nx_comm.coverage(test_greedy, c_test)
nx_comm.performance(test_greedy, c_test)
######################################################################
g_train,g_test = train_test_split(G,test_size = 0.2)
g1 = nx.Graph()
g1 = nx.from_pandas_edgelist(g_train, source = 'Origin_code', target = 'Dest_code')

pos1 = nx.spring_layout(g1)
nx.draw_networkx(g1, pos1, node_size = 15, node_color = 'red')

communities1 = greedy_modularity_communities(g1)

node_groups1 = []
for com in next(iter(communities1)):
  node_groups1.append(list(com))

print(node_groups1)

color_map1 = []
for node in g1:
    if node in node_groups1[0]:
        color_map1.append('blue')
    else: 
        color_map1.append('green')  
nx.draw(g1, node_color=color_map1, with_labels=True)
plt.show()

***********************************************
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
dataset = pd.read_excel('I:/Data science/DS Project/New folder/Optimal traffic routes.xlsx')
dataset['Origin_code'].fillna(0,inplace=True)
dataset['Dest_code'].fillna(dataset['Dest_code'].mean(),inplace=True)
X = dataset.iloc[:, :3]
def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'Eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0: 0}
    return word_dict[word]
X['Origin_code'] = X['Origin_code'].apply(lambda x : convert_to_int(x))
y = dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model1.pkl','rb'))

from sklearn.metrics import accuracy_score
print(accuracy_score.x_train)
