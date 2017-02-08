import networkx as nx
import matplotlib.pyplot as plt
G = nx.read_edgelist(  "test1.edgelist", nodetype=int)
metric = nx.betweenness_centrality(G)
f = open('new_result/' + 'test.txt', 'w')
for key, value in metric.iteritems():
    f.write(str(key) + ' ' + str(value) + '\n')
f.close()

pos_dict = nx.spring_layout(G)
nx.draw(G,pos=pos_dict)
labels=nx.draw_networkx_labels(G,pos=pos_dict)
plt.show()
