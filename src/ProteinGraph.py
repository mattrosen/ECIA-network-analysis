import numpy as np
import scipy.io as sio
import math
import copy
import networkx as nx
import pandas as pd 
from networkx.algorithms.centrality import betweenness_centrality
import community
try:
    from src import utilities
except:
    import utilities

import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

class ProteinGraphBase:

    def __init__(self, adj, proteinnames, figpath):
        self.adj = adj
        self.proteinnames = proteinnames
        self.figpath = figpath
        return

    ################################################################################
    def form_by_sparsity(self, pct, save_fn=None):
        """ Forms graph from weight matrix W given sparsity percentage pct.

            Args:    pct     (float)
                     save_fn (None or str)
            
            Returns: graph   (ProteinGraph obj)

        """ 
        
        # do some argument verification
        assert pct >= 0 and pct <= 1, "Argument `pct` should be in range [0, 1]."

        # select top `pct` % of links
        threshold_weight = np.percentile(self.adj.flatten(), 100 - 100 * pct)
        graph = copy.deepcopy(self.adj)
        graph[graph < threshold_weight] = 0

        # save if desired
        if save_fn is not None:
            utilities.save_edgelist(graph, save_fn)

        return ProteinGraph(graph, self.proteinnames, f"sparsity={pct:.3f}", self.figpath)

    ################################################################################
    def form_by_threshold(self, wmin, save_fn=None):
        """ Forms graph from weight matrix W given edge-weight threshold value.

            Args:    wmin  (float)
                     save_fn (None or str)
            
            Returns: graph   (ProteinGraph obj)

        """ 
        
        # Do some argument verification
        assert wmin < np.amax(self.adj), "Argument `wmin` should be < max(self.adj)."

        # Call any edge with weight > threshold a link
        graph = copy.deepcopy(self.adj)
        graph[graph < wmin] = 0

        # Save if desired
        if save_fn is not None:
            utilities.save_edgelist(graph, save_fn)

        return ProteinGraph(graph, self.proteinnames, f"wmin={wmin:.3f}", self.figpath)


class ProteinGraph(object):

    def __init__(self, adj, proteinnames, name, figpath):

        self.adj = adj
        self.G = nx.from_numpy_array(adj)
        self.proteinnames = proteinnames
        self.name = name
        self.figpath = figpath

        return 

    ################################################################################
    # Graph randomization methods
    def random_link_removal(self, pct, i, save_fn=None):
        """ Remove pct % of links from graph specified by adj.

            Args:    pct     (float)
                     i       (int)
                     save_fn (None or str)
            
            Returns: graph (ProteinGraph object)

        """ 
        
        # Do some argument verification
        assert pct >= 0 and pct <= 1, "Argument `pct` should be in range [0, 1]."

        # Find links
        links = np.where(self.adj > 0)

        # Choose appropriate pct of links to remove, + remove them
        ind = np.random.choice(len(links[0]), int(pct * len(links[0])), 
                replace=False)
        to_rmv = [links[0][ind], links[1][ind]]
        depleted = copy.deepcopy(adj)
        depleted[to_rmv] = 0

        # save if desired
        if save_fn is not None:
            utilities.save_edgelist(depleted, save_fn)

        return ProteinGraph(depleted, self.proteinnames, 
            self.name + f"_randremoval_pct={pct:.3f}_{i}", self.figpath)

    ################################################################################
    def weight_permutation(self, i, save_fn=None):
        """ Permute weights of graph specified by adj.
            
            Args:    i        (int)
                     save_fn  (None or str)
            
            Returns: graph (ProteinGraph obj)
        """

        # Find links
        links = np.where(self.adj > 0)

        # Permute weights
        permuted = copy.deepcopy(self.adj)
        r_weights = np.random.permutation(permuted[links])
        permuted[links] = r_weights

        # Save if desired
        if save_fn is not None:
            utilities.save_edgelist(permuted, save_fn)

        return ProteinGraph(permuted, self.proteinnames, 
            self.name + f"_wperm_{i}", self.figpath)

    ################################################################################
    def edge_permutation(self, i, save_fn=None):
        """ Permute edges of graph specified by adj.
            
            Args:    i        (int)
                     save_fn  (None or str)
            
            Returns: graph (ProteinGraph obj)
        """

        # Find links
        links = np.where(self.adj > 0)

        # Randomly link network
        new_links = np.random.choice(range(len(self.adj.flatten())), 
            len(links[0]), replace=False)
        weights = self.adj[links]
        permuted = np.zeros(len(self.adj.flatten()))
        permuted[new_links] = weights
        permuted = np.reshape(permuted, self.adj.shape)

        # save if desired
        if save_fn is not None:
            utilities.save_edgelist(permuted, save_fn)

        return ProteinGraph(permuted, self.proteinnames, 
            self.name + f"_eperm_{i}", self.figpath)

    ################################################################################
    def degree_preserving_randomization(self, i, n_swaps = 1000, save_fn=None):
        """ Perform degree-preserving randomization of graph
            specified by adj; does this by swapping link pairs
            if they form new link pairs that don't already exist
            in the network structure.
            
            Args:    i        (int)    
                     save_fn  (None or str)
            
            Returns: graph (ProteinGraph obj)
        """

        # do some argument verification
        assert type(n_swaps) == int, "Argument `n_swaps` should be int."

        # Find links
        links = np.where(self.adj > 0)
        weights = self.adj[links]
        permuted = np.zeros(self.adj.shape)

        link_set = set(zip(links[0], links[1]))

        # Swap link pairs
        swapped = 0
        while swapped < n_swaps:
            l1, l2 = np.random.choice(range(len(links[0])), 2)

            # if either of the generated link pairs exists, continue
            if ((links[0][l1], links[1][l2]) in link_set or 
                (links[0][l2], links[1][l1]) in link_set):
                continue

            # otherwise, add new edges to link set, remove old ones,
            # and update links arrays accordingly
            link_set.remove((links[0][l1], links[1][l1]))
            link_set.remove((links[0][l2], links[1][l2]))
            link_set.add((links[0][l1], links[1][l2]))
            link_set.add((links[0][l2], links[1][l1]))
            l1_targ = links[1][l1]
            l2_targ = links[1][l2]
            links[1][l1] = l2_targ
            links[1][l2] = l1_targ
            swapped += 1


        # add specified connectivity to permuted
        permuted[links] = weights

        # save if desired
        if save_fn is not None:
            utilities.save_edgelist(permuted, save_fn)

        return ProteinGraph(permuted, self.proteinnames, 
            self.name + f"_degpresrand_{i}", self.figpath)

    ################################################################################
    def save_edgelist_csv(self):

        inds_to_keep = np.where(self.adj > 0)
        with open(self.name + "_edgelist.csv", 'w') as f:
            f.write("Source,Target,Weight,Type\n")
            for x,y in zip(inds_to_keep[0], inds_to_keep[1]):

                # make sure uncharacterized proteins are labeled correctly
                src_name = self.proteinnames[x][1]
                targ_name = self.proteinnames[y][1]
                if src_name == '/':
                    src_name = self.proteinnames[x][0]
                if targ_name == '/':
                    targ_name = self.proteinnames[y][0]
                f.write(f"{src_name},{targ_name},{self.adj[x,y]},Undirected\n")

        return

    ############################################################################
    def compute_betweenness_centrality(self):
        """ Compute betweenness centrality for all nodes in the graph.
                  
            Returns: centralities (dict of dicts)
            
        """
        return betweenness_centrality(self.G)

    def plot_betweenness_centrality_distribution(self):

        return

    ############################################################################
    def compute_community_assignments(self, louvain_args=[{'resolution': 0.2}]):
        """ Compute community assignments via Louvain algorithm.

            Args:     louvain_args (list of dict)
                  
            Returns:  comm_assignments (dict of dicts)
            
        """
    
        comm_assignments = {}

        # For each cobination of arguments to community detection algorithm,
        # identify community IDs for all nodes and store        
        for i, l_args in enumerate(louvain_args):
            comm_assignments[(self.name,i)] = community.best_partition(self.G, **l_args)
        
        return comm_assignments

    ############################################################################
    def compute_degree_distribution(self):
        """ Compute node degrees.
            
            Returns: degrees (dict)
            
        """
        
        # Compute + return node degrees
        return dict(self.G.degree())

    def plot_degree_distribution(self, deg_dist):
        """ Plot degree distribution.
            
            Returns: None
        """

        # Extract + plot degree distribution
        degrees = np.array(deg_dist.values())
        fig, ax = plt.subplots(1, figsize=(3,3))
        sns.histplot(data=degrees, ax=ax, bins=degrees.max() - degrees.min(),
            discrete=True, linewidth=0, stat='count')
        ax.set(xlabel='Degree', ylabel='Count')

        plt.tight_layout()
        fig.savefig(os.path.join(self.figpath, 
            f'{self.sparsity}_degree_distribution.pdf'))

        return 

    ############################################################################