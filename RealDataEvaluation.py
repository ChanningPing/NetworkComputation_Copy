from __future__ import division
import collections
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import operator
from scipy import stats
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stat
from sklearn.metrics import jaccard_similarity_score
import csv



def read_real_data(file_names):
    '''
    :param file_names: a list of file names, such as ['preprocessed_facebook_combined','facebook-100percent-noise']
    :return: a list of graphs
    '''
    graphs = []  # all graphs read from real data files
    for file_name in file_names:
        G = nx.read_edgelist(file_name + ".edgelist", nodetype=int)  # read one file from data, save into G
        graphs.append(G) # push G into graphs list
    return graphs

def draw_node_degree_plot(graphs,data_name):

    # create axis
    ax = plt.gca()
    # create legend handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    styles = ['-ko','-b^','-gs','-r*','-c+','-mx'] # line stlyes
    line_labels = ['original','k=5','k=10','k=15','k=20','k=25'] # line names



    lines = [] # lines
    count_distributions = []


    for index, G in enumerate(graphs):
        # get vertex degree sequence
        degree_sequence = nx.degree(G).values()  # degree sequence
        # count frequency of each degree ( <degree:frequency> dict)
        degreeCount = collections.Counter(degree_sequence)
        # sort <degree:frequency> dict by degree in increasing order (a list of tuples as result)
        degreeCount = sorted(degreeCount.items(), key=operator.itemgetter(0))

        # convert a list of tuples to 2 lists
        degrees, counts = map(list, zip(*degreeCount))
        line, = ax.plot(degrees, counts, styles[index], label=line_labels[index])
        lines.append(line)
        print('----------------degree distribution:' + data_name)
        print(degreeCount)
        print(degrees)
        print(counts)
        count_distributions.append(counts)

    # use the distributions to do k-s test
    K_S_test(count_distributions, data_name)

    plt.rcParams["figure.figsize"] = [10, 10]

    # linking lines with ledgend
    plt.legend(lines, line_labels,prop={'size': 18})

    # set y-axis to be log-scale
    ax.set_yscale('log')

    # set plot titles, y-label, x-label
    #plt.title("Distribution of Vertex Degree:" + data_name +' data')
    plt.ylabel("Num of Vertices")
    plt.xlabel("Degree")
    plt.gcf().set_facecolor('white')
    plt.rcParams.update({'font.size': 22})
    #plt.margins(0.2, 0.4)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    fig = plt.gcf()
    fig.tight_layout()

    # save and show plot
    plt.savefig('new_figures/'+data_name +"_degree_distribution.eps", format='eps', dpi=1000)


    plt.show()

def draw_node_betweenness_plot(graphs,data_name):


    # create axis
    ax = plt.gca()
    # create legend handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    styles = ['-ko', '-b^', '-gs', '-r*', '-c+', '-mx']  # line stlyes
    line_labels = ['original', 'k=5', 'k=10', 'k=15', 'k=20', 'k=25']  # line names
    lines = [] # lines
    count_distributions = []


    for index, G in enumerate(graphs):
        node_betweenness_centralities = nx.betweenness_centrality(G)
        save_data(node_betweenness_centralities, data_name, 'betweenness', index)
        # get vertex betweenness sequence
        betweenness_sequence = node_betweenness_centralities.values()
        decimal = 0.001
        betweenness_sequence = [round(b / decimal) * decimal for b in betweenness_sequence]
        # count frequency of each degree ( <degree:frequency> dict)
        betweennessCount = collections.Counter(betweenness_sequence)
        # sort <degree:frequency> dict by degree in increasing order (a list of tuples as result)
        betweennessCount = sorted(betweennessCount.items(), key=operator.itemgetter(0))


        # convert a list of tuples to 2 lists

        betweennesses, counts = map(list, zip(*betweennessCount))
        line, = ax.plot(betweennesses, counts, styles[index], label=line_labels[index])
        lines.append(line)
        print('----------------betweenness distribution:' + data_name)
        print(betweennessCount)
        print(betweennesses)
        print(counts)
        count_distributions.append(counts)

    # use the distributions to do k-s test
    K_S_test(count_distributions, data_name)

    plt.rcParams["figure.figsize"] = [10, 10]

    # linking lines with ledgend
    plt.legend(lines, line_labels,prop={'size': 18})

    # set y-axis to be log-scale
    ax.set_yscale('log')

    # set plot titles, y-label, x-label
    #plt.title("Distribution of Vertex Betweenness:" + data_name +' data')
    plt.ylabel("Num of vertices")
    plt.xlabel("Betweenness")
    plt.gcf().set_facecolor('white')
    plt.rcParams.update({'font.size': 22})
    plt.locator_params(axis='x', nbins=5)
    #plt.margins(0.2, 0.4)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    # save and show plot
    plt.savefig('new_figures/' + data_name + "_betweenness_distribution.eps", format='eps', dpi=1000)
    plt.show()


def draw_node_closeness_plot(graphs, data_name):
    # create axis
    ax = plt.gca()
    # create legend handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    styles = ['-ko', '-b^', '-gs', '-r*', '-c+', '-mx']  # line stlyes
    line_labels = ['original', 'k=5', 'k=10', 'k=15', 'k=20', 'k=25']  # line names
    lines = []  # lines
    count_distributions = []

    for index, G in enumerate(graphs):
        node_closeness_centralities = nx.closeness_centrality(G)
        save_data(node_closeness_centralities, data_name, 'closeness', index)
        # get vertex betweenness sequence
        closeness_sequence = node_closeness_centralities.values()
        decimal = 0.035
        closeness_sequence = [round(b / decimal) * decimal for b in closeness_sequence]
        # count frequency of each degree ( <degree:frequency> dict)
        closenessCount = collections.Counter(closeness_sequence)
        # sort <degree:frequency> dict by degree in increasing order (a list of tuples as result)
        closenessCount = sorted(closenessCount.items(), key=operator.itemgetter(0))

        # convert a list of tuples to 2 lists

        closeness, counts = map(list, zip(*closenessCount))
        line, = ax.plot(closeness, counts, styles[index], label=line_labels[index])
        lines.append(line)
        print('----------------closeness distribution:' + data_name)
        print(closenessCount)
        print(closeness)
        print(counts)
        count_distributions.append(counts)

    # use the distributions to do k-s test
    K_S_test(count_distributions, data_name)

    plt.rcParams["figure.figsize"] = [10, 10]

    # linking lines with ledgend
    plt.legend(lines, line_labels,loc = 2, prop={'size': 18})

    # set y-axis to be log-scale
    ax.set_yscale('log')

    # set plot titles, y-label, x-label
    #plt.title("Distribution of Vertex Closeness:" + data_name + ' data')
    plt.ylabel("Num of vertices")
    plt.xlabel("Closeness")
    plt.gcf().set_facecolor('white')
    plt.rcParams.update({'font.size': 22})
    #plt.margins(0.2, 0.2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    # save and show plot
    plt.savefig('new_figures/' + data_name + "_closeness_distribution.eps", format='eps', dpi=1000)
    plt.show()

def draw_node_pagerank_plot(graphs,data_name):


    # create axis
    ax = plt.gca()
    # create legend handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    styles = ['-ko', '-b^', '-gs', '-r*', '-c+', '-mx']  # line stlyes
    line_labels = ['original', 'k=5', 'k=10', 'k=15', 'k=20', 'k=25']  # line names
    lines = [] # lines
    count_distributions = []


    for index, G in enumerate(graphs):
        pagerank_alpha = 0.85 # default damping factor
        node_pageranks = nx.pagerank(G, alpha=pagerank_alpha)
        save_data(node_pageranks, data_name, 'pagerank', index)
        # get vertex betweenness sequence
        pagerank_sequence = node_pageranks.values()
        decimal = 0.0001
        pagerank_sequence = [round(b / decimal) * decimal for b in pagerank_sequence]
        # count frequency of each degree ( <degree:frequency> dict)
        pagerankCount = collections.Counter(pagerank_sequence)
        # sort <degree:frequency> dict by degree in increasing order (a list of tuples as result)
        pagerankCount = sorted(pagerankCount.items(), key=operator.itemgetter(0))


        # convert a list of tuples to 2 lists

        pageranks, counts = map(list, zip(*pagerankCount))
        line, = ax.plot(pageranks, counts, styles[index], label=line_labels[index])
        lines.append(line)
        print('----------------pagerank distribution:' + data_name)
        print(pagerankCount)
        print(pageranks)
        print(counts)
        count_distributions.append(counts)

    # use the distributions to do k-s test
    K_S_test(count_distributions, data_name)

    plt.rcParams["figure.figsize"] = [8,10]

    # linking lines with ledgend
    plt.legend(lines, line_labels,prop={'size': 18})

    # set y-axis to be log-scale
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))

    # set plot titles, y-label, x-label
    #plt.title("Distribution of Vertex PageRank:" + data_name +' data')
    plt.ylabel("Num of vertices")
    plt.xlabel("PageRank")
    plt.gcf().set_facecolor('white')
    plt.rcParams.update({'font.size': 22})
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.locator_params(axis='x', nbins=5)

    #plt.margins(0.2, 0.2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # save and show plot
    plt.savefig('new_figures/' + data_name + "_pagerank_distribution.eps", format='eps')
    plt.show()

def K_S_test(count_distributions, data_name):

    original =  count_distributions[0]
    k_s_pvalue = []
    k_s_stats = []
    for index, count in enumerate(count_distributions):
        if index>0:
            stat,pvalue = stats.ks_2samp(original, count)
            k_s_pvalue.append(pvalue)
            k_s_stats.append((stat))
            print(str(stat)+','+str(pvalue))
    print('-----------the k_s test p-value:' + data_name)
    print(k_s_pvalue)
    print('-----------the k_s test stat:' + data_name)
    print(k_s_stats)

def draw_K_S_test_plot():


    plt.ylim(ymax=1.05)

    metric = 'vertex degree'
    x = [5,10,15,20,25]
    p_values = [
        [0.019102569100619804, 0.017621145374449365, 0.041527419109828323, 0.041304265973869492, 0.04466885686552674] #stats # [data:Facebook]
        #[0.0235791543756,0.0265486725664,0.0398230088496,0.0308538627123,0.0430057444496] #stats # [data:WikiVotes]
    ]
    line_labels = ['Facebook']
    line_styles = ['-bo']
    lines = []

    for index,p in enumerate(p_values):
        line, = plt.plot(x, p, line_styles[index], label=line_labels[index])
        lines.append(line)

    # linking lines with ledgend
    plt.legend(lines, line_labels,loc='upper right',  prop={'size': 18}, borderaxespad=0.)

    # set plot titles, y-label, x-label
    # plt.title("K-S test: " + metric)
    plt.ylabel("K-S test statistics")
    plt.xlabel("K")
    plt.gcf().set_facecolor('white')
    plt.rcParams.update({'font.size': 22})
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # save and show plot
    plt.savefig('new_figures/' + metric + "_K_S_test.eps", format='eps')
    #plt.savefig('new_figures/'+metric + "_K_S_test.png")
    plt.show()

def calculate_ACC_APL (graphs, data_name):
    ACC = []
    APL = []
    for G in graphs:
        ACC.append(nx.average_clustering(G))
        APL.append(nx.average_shortest_path_length(G))
    print('-------------ACC of ' + data_name)
    print(ACC)
    print('-------------APL of ' + data_name)
    print(APL)


def calculate_number_of_modified_edges(graphs,data_name):
    '''
    :param graphs: all graphs of a data: original, k=5,k=30...
    :return:
    '''
    original =[]
    # calculate all shortest paths
    for index, G in enumerate(graphs):
        A = nx.adjacency_matrix(G)
        A = A.todense()
        A = A.tolist()
        A = list(itertools.chain.from_iterable(A))
        #print(A)
        if index ==0:
            original = A
        else:
            diffs = [a - o for a, o in zip(A, original)]
            added_edges = diffs.count(1)/2
            deleted_edges = diffs.count(-1)/2
            modifed_edges = added_edges + deleted_edges
            print('--------------------' + data_name)
            print('[number of added edges]=' + str(added_edges))
            print('[number of deleted edges]=' + str(deleted_edges))
            print('[number of modified edges]=' + str(modifed_edges))

def draw_added_edges(number_of_edges, data_name):
    plt.ylim(ymin=0,ymax=200000)

    x = [5,30]
    line_labels = ['Added edges', 'Deleted edges','Modifed edges']
    line_styles = ['-g^','-rs','-bo']
    lines = []

    for index, e in enumerate(number_of_edges):
        line, = plt.plot(x, e, line_styles[index], label=line_labels[index])
        lines.append(line)

    # linking lines with ledgend
    plt.legend(lines, line_labels, loc='lower left', borderaxespad=0.)

    # set plot titles, y-label, x-label
    plt.title("Number of added/delted/modified edges: " + data_name)
    plt.ylabel("Number of added/delted/modified edges: " + data_name)
    plt.xlabel("K")
    plt.gcf().set_facecolor('white')

    # save and show plot
    plt.savefig("number_of_add_deleted_modifed_edges_" + data_name +".png")
    plt.show()

def save_data(metric,data_name, metric_name, data_index):
    f = open('new_result/' + data_name + '_'+ metric_name + '_' + str(data_index) + '.txt', 'w')
    f.write('node,' + metric_name + '\n')
    for key,value in metric.iteritems():
        f.write(str(key) + ' ' + str(value) + '\n')
    f.close()




def main():

    # 'data/FaceBook_original','data/FaceBook_K_5', 'data/FaceBook_K_10','data/FaceBook_K_15','data/FaceBook_K_20','data/FaceBook_K_25'
    # 'data/WiKi_original','data/WiKi_K_5', 'data/WiKi_K_10','data/WiKi_K_15','data/WiKi_K_20','data/WiKi_K_25'
    data_name = 'wiki_RandomRanked_TopK100'
    #file_names =['data/FaceBook_original','data/FaceBook_K_5', 'data/FaceBook_K_10','data/FaceBook_K_15','data/FaceBook_K_20','data/FaceBook_K_25']
    file_names = ['new_data/' +data_name+'_original','new_data/' +data_name+'_K5', 'new_data/' +data_name+'_K10','new_data/' +data_name+'_K15','new_data/' +data_name+'_K20','new_data/' +data_name+'_K25']

    graphs = read_real_data(file_names)

    #draw_node_degree_plot(graphs, data_name)

    draw_node_betweenness_plot(graphs, data_name)

    #draw_node_closeness_plot(graphs, data_name)

    #draw_node_pagerank_plot(graphs, data_name)

    #draw_K_S_test_plot()




    '''
    number_of_edges = [
        [100038, 99020],  # [data:added edges]
        [99548, 97053],  # [data:deleted edges]
        [199585, 196072]  # [data:modifed edges]
    ]
    draw_added_edges(number_of_edges, "WiKi Votes")
    '''




main()

