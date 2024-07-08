import numpy as np
import scipy.io as sio 
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy, os, itertools, argparse
from collections import defaultdict
import networkx as nx
import seaborn as sns
import xlsxwriter
import pandas as pd
try:
    from src import ProteinGraph as pg
except:
    import ProteinGraph as pg 


class InteractomeDataset(object):

    def __init__(self, args):

        # Bind relevant information
        self.filename              = args.filename
        self.fieldname             = args.fieldname
        self.is_sym                = 'sym' in args.fieldname

        self.resultspath           = args.resultspath
        self.figpath               = args.figpath
        self.n_perm                = args.n_perm
        self.do_nulls              = args.n_perm > 0
        self.edge_inclusion_method = args.edge_inclusion_method
        self.save_edgelists        = bool(args.save_edgelists)

        self.gene_name_fn  = args.gene_name_fn

        # Load unprocessed data
        self.raw_data = self.load_raw_data()
        self.N_NODES  = self.raw_data.shape[0]
        self.names    = self.load_names()


        # Load expression levels; if relevant information not passed,
        # None returned, and no adjustment of interactions performed
        '''
        self.names, exp_level_data = self.load_expression_levels()
        if exp_level_data is not None:
            self.raw_data = self.normalize_by_expression_levels(self.raw_data,
                exp_level_data)
        '''

    ############################################################################
    # Data loading/pre-processing methods
    def load_raw_data(self):
        """ 
        Load + return raw dataset of interactions.
        """
        return sio.loadmat(self.filename)[self.fieldname]

    def load_names(self):
        """
        Load + return gene/protein names.
        """
        return sio.loadmat(self.gene_name_fn)['names']

    def load_expression_levels(self):
        """
        Load + return expression levels.
        """

        #########################################################################
        # Deal with node label remapping
        with open(self.gene_name_fn, 'r') as f:
            names = np.array([line.strip().split(',') for line in f.readlines()])

        # Strip all whitespace internal to each name
        for i in range(names.shape[0]):
            for j in range(names.shape[1]):
                names[i, j] = "".join(names[i, j].split())
            if names[i,1] == 'SLT-1N':
                names[i,0] += "-N"
            elif names[i,1] == 'SLT-1C':
                names[i,0] += "-C"
            elif names[i,1] == 'UNC-6S':
                names[i,0] += "-S"

        # If any of the necessary filenames are None, return
        if any([self.expression_fn is None,
                self.plasmid_fn is None,
                self.gene_name_fn is None]):
            return names, None

        # Parse plasmid names
        plasmid_dict = {}
        with open(self.plasmid_fn, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.split(",")
                key = ",".join(line[1:3])

                # some corner cases in reading in
                if line[1] == '' and line[2] == '':
                    continue

                if line[2] == '':
                    key += '/'

                # Extract plasmid nums to enable later indexing
                plasmids_raw = line[0].split('/')
                plasmid_fc = "".join(plasmids_raw[0].split())
                plasmid_ap = "pCE" + plasmids_raw[1]

                # add to dict
                key = "".join(key.split())
                plasmid_dict[key] = [plasmid_fc, plasmid_ap]

        # Construct a dictionary of expression levels for each plasmid
        exp_by_plasmid = {}
        with open(self.expression_fn, 'r') as f:
            for line in f.readlines():
                line = line.split(",")
                plasmid = line[0]
                if plasmid == "":
                    break
                if line[1].startswith("/"):
                    continue

                value = float(line[1])
                exp_by_plasmid[plasmid] = value

        # Build a dictionary by protein of exp level in (FC, AP) configs
        exp_levels = {}
        for key, value in plasmid_dict.items():
            exp = []
            for plasmid in value:
                exp.append(exp_by_plasmid[plasmid])
            exp_levels[key] = exp

        

        return names, exp_levels

    def normalize_by_expression_levels(self, raw_data, exp_level_data):
        """
        Normalize raw data by expression levels.

        Key idea: if measured interaction strength is a function of
        expression level, then it will reflect expression of both bait
        and prey expression levels. 

        """
        # Unpack expression level data/names
        names, exp_levels = exp_level_data

        # Build expression product matrix
        exp_products = np.zeros_like(raw_data)

        # Loop through all interactions; take product of 
        # pair's expression, divide interaction strength
        for i in range(raw_data.shape[0]):

            # FC name, expression
            fc_name = ",".join(names[i, :])
            fc_exp = exp_levels[fc_name][0]

            for j in range(raw_data.shape[1]):

                # AP name, expression
                ap_name = ",".join(names[j, :])
                ap_exp = exp_levels[ap_name][1]

                exp_products[i, j] = fc_exp * ap_exp

        # To avoid dividing by 0 when normalizing, add the max
        # expression value to all elements; this constrains
        # the effect of the normalization to 2x
        exp_products += exp_products.max()
        
        return np.divide(raw_data, exp_products)

    ############################################################################
    # Graph creation methods
    def create_graphs(self, sparsities, thresholds):
        """ 
        Create and save graphs. 
        """
        graphs = {}

        # First - combine z-scored interactions via geometric mean to obtain one score per
        # pair

        if self.is_sym:
            merged_scores = self.raw_data
        else:
            merged_scores = np.zeros_like(self.raw_data)
            for i in range(self.N_NODES):
                for j in range(self.N_NODES):
                    s1 = self.raw_data[i,j]
                    s2 = self.raw_data[j,i]
                    if s1 < 0 or s2 < 0:
                        ms = 0
                    else:
                        ms = (s1*s2)**0.5
                    merged_scores[i,j] = ms
                    merged_scores[j,i] = ms

        base_graph = pg.ProteinGraphBase(merged_scores, self.names, self.figpath, remove_isolates=False)

        # Iterating through sparsity-formed graphs - by sparsity or threshold
        rel_var = sparsities if sparsities is not None else thresholds
        for j, level in enumerate(rel_var):
            print(f"{level:.3f}")

            results_fn = os.path.join(self.resultspath, "graphs/"
                    f"{level:.1e}_weights.txt.gz") if self.save_edgelists else None
            if results_fn is not None and os.path.exists(results_fn):
                continue

            # Generate and save graph
            if sparsities is not None:
                G = base_graph.form_by_sparsity(level,
                    results_fn)
            else:
                G = base_graph.form_by_threshold(level,
                    results_fn)
            graphs[level] = G

            # Generate and save null graphs, if specified
            if self.do_nulls:
                
                null_pref = os.path.join(self.resultspath, "nulls/",
                    f"{level:.1e}/")
                dp_pref = null_pref + "deg_pres/degree_preserved_"
                w_pref = null_pref + "w_perm/w_perm_"
                e_pref = null_pref + "e_perm/e_perm_"

                # Weight/edge permutation, deg. pres. rand
                for i in range(self.n_perm):
                    suff = f"perm_{str(i)}.txt"
                    w_perm = G.permute_weights(i, w_pref + suff)
                    e_perm = G.permute_edges(i, e_pref + suff)
                    deg_pres = G.degree_preserving_randomization(i,
                        save_fn=dp_pref + suff)

        return graphs

    def create_single_graph(self, method, val):
        """ 
        Create single graph, save edgelist for visualization in Gephi 
        """

        # Multi-Z method: combine z-scored interactions via geometric mean 
        # to obtain one score per pair
        if self.edge_inclusion_method == 'multi_z':

            merged_scores = np.zeros_like(self.raw_data)
            for i in range(self.raw_data.shape[0]):
                for j in range(self.raw_data.shape[1]):
                    s1 = self.raw_data[i,j]
                    s2 = self.raw_data[j,i]
                    if s1 < 0 or s2 < 0:
                        ms = 0
                    else:
                        ms = (s1*s2)**0.5
                    merged_scores[i,j] = ms
                    merged_scores[j,i] = ms

        # Kovacs max-ent method  (new interface)
        elif self.edge_inclusion_method == 'kovacs_maxent':
            merged_scores = self.raw_data

        # Kovacs max-ent method (old interface)
        elif self.edge_inclusion_method == 'kovacs_maxent_old':

            # Load hits directly
            edges = pd.read_csv("../data/185PPIs.csv").to_numpy()
            merged_scores = np.zeros_like(self.raw_data)

            # Loop through proteins, map to protein number, add connection weight
            for i in range(edges.shape[0]):
                source = edges[i, 0]
                dest   = edges[i, 2]
                weight = edges[i, 4]

                source_id = np.where(self.names[:,0] == source)[0]
                dest_id = np.where(self.names[:,0] == dest)[0]
                merged_scores[source_id, dest_id] = weight

        base_graph = pg.ProteinGraphBase(merged_scores, self.names, self.figpath, remove_isolates=True)

        # Generate graph
        if method == 'sparsity':
            G = base_graph.form_by_sparsity(val)
        elif method == 'threshold':
            G = base_graph.form_by_threshold(val)
        
        # Write edgelist
        G.save_edgelist_csv()

        return {(method, val): G}

    ############################################################################
    # Analysis methods
    def compute_modularity_vs_null(self, graph, resolutions, n_null=1000):

        g = list(graph.values())[0]

        # Compute modularity score for actual network across resolutions
        com_resolutions = [{'resolution': i} for i in resolutions]

        # For the provided graph, compute modularity across resolutions
        modularities = np.array(list(
            g.compute_community_assignments(com_resolutions, True).values()))
        
        # Compute null distribution of modularity values by degree-preserving 
        # randomization
        null_mods = np.zeros((n_null, len(resolutions)))
        for n in range(n_null):
            print(n)
            null = g.permute_weights_by_node(n)
            null_mods[n] = np.array(list(
                null.compute_community_assignments(com_resolutions, True).values()))


        nullmean = null_mods.mean(0)
        nullsd = null_mods.std(0)
        nullhigh = null_mods.max(0) - nullmean
        nulllow = nullmean - null_mods.min(0)

        fig, ax = plt.subplots(1)
        ax.scatter(resolutions, modularities, color='red', label='Data')
        ax.errorbar(resolutions, nullmean, yerr=[nulllow, nullhigh], color='black', label='Weight-shuffled control')
        ax.legend()
        ax.set(xlabel='Louvain resolution (a.u.)', ylabel='Modularity (a.u.)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        fig.savefig(os.path.join(self.figpath, 'modularity_true_vs_control.png'), dpi=500, bbox_inches='tight')
        plt.show()

        return

    def plot_community_graph(self, graph, resolution):
        """ 
        Plot graph of communities; print and save list of communities for later annotation.
        """
        # Identify communities in graph
        g = list(graph.values())[0]
        comm_assignments = g.compute_community_assignments([{'resolution': resolution}])
        induced_graph = g.compute_induced_graph(list(comm_assignments.values())[0])

        # Remove isolates, self loops
        induced_graph.remove_edges_from(list(nx.selfloop_edges(induced_graph)))
        induced_graph.remove_nodes_from(list(nx.isolates(induced_graph)))
        induced_graph = induced_graph.to_undirected()

        # Draw the graph, and save node assignments
        pos = nx.kamada_kawai_layout(induced_graph, scale=3)

        # Assign node colors by partition
        n_comms = induced_graph.number_of_nodes()

        cm        = sns.color_palette('nipy_spectral', as_cmap=True)
        cNorm     = mpl.colors.Normalize(vmin=0, vmax=n_comms-1)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
        colors = np.array([scalarMap.to_rgba(i) for i in range(n_comms)])
        np.random.shuffle(colors)

        # Assign node size by weighted degree
        wdegs = np.array([i[1] for i in induced_graph.degree(weight='weight')])
        
        # Assign edge size by weight
        edgeweights = np.array([i[-1] for i in list(induced_graph.edges(data='weight'))])

        node_labels = np.arange(len(list(induced_graph.nodes())))
        node_label_dict = {j: i for i, j in enumerate(list(induced_graph.nodes()))}
        rev_node_label_dict = {i: j for i,j in enumerate(list(induced_graph.nodes()))}

        fig, ax = plt.subplots(1, figsize=(8, 8))
        nx.draw_networkx_nodes(induced_graph, pos, ax=ax, node_color=colors, node_shape='s', node_size=np.log2(wdegs)**2 * 10 + 200)
        nx.draw_networkx_edges(induced_graph, pos, ax=ax, alpha=0.5, width=edgeweights/40)
        nx.draw_networkx_labels(induced_graph, pos, node_label_dict, ax=ax, font_size=18, font_family='Arial', font_weight='bold', font_color='white')
        plt.show()

        fig.savefig(os.path.join(self.figpath, "community_graph.pdf"), bbox_inches='tight', fonttype=42)



        # Print members of each community
        ca = list(comm_assignments.values())[0]
        for c in np.arange(n_comms):
            original_id_in_induced_graph = rev_node_label_dict[c]
            in_c = np.array([k for k, v in ca.items() if v == original_id_in_induced_graph])
            in_c_names = [g.proteinnames[i][0] if g.proteinnames[i][1] == '/' else g.proteinnames[i][1] for i in in_c]
            print(c, node_labels[c])
            for n in in_c_names:
                print(f"{n}")
            print()

        # Sort protein names, print community ID
        all_names = [n[0] if n[1] == '/' else n[1] for n in g.proteinnames]
        sorting_order = np.argsort(all_names)
        for k in sorting_order:
            if k in ca.keys():
                print(k, all_names[k], ca[k])

        return

    def find_canonical_communities(self, comms, p):

        # Across all versions of the graph and resolution values,
        # find every node's community
        all_neighbors = defaultdict(list)
        canonical_neighbors = {}

        # Loop through graph types/resolution values and nodes 
        for k, comm in comms.items():
            for i in list(comm.keys()):

                # In every version of the community breakdown, store the nodes
                # that belong to the same community
                fellow_community_members = [k for k, v in comm.items() if v == comm[i]]
                all_neighbors[i].append(fellow_community_members)

        # Identify canonical neighbors by evaluating nodes that frequently appear
        # in community together (greater than p percent of community assignments pair
        # them together)
        for node in list(all_neighbors.keys()):

            flattened = [item for sublist in all_neighbors[node] for item in sublist]
            counts = {i: flattened.count(i) for i in flattened}
            mult_appearing = [k for k, v in counts.items() if v / len(comms) > p]
            canonical_neighbors[node] = mult_appearing

        self.write_community_results(canonical_neighbors, all_neighbors)
        
        return

    def write_community_results(self, canonical_neighbors, all_neighbors):

        workbook = xlsxwriter.Workbook('TableS4_neighbors.xlsx')
        canonical_sheet = workbook.add_worksheet('Canonical neighbors')
        all_sheet = workbook.add_worksheet('All neighbors')

        bold = workbook.add_format({'bold': True})

        all_sheet.set_column('A:A', 20)
        canonical_sheet.set_column('A:A', 20)

        # Format the header rows
        all_sheet.write('A1','All neighbors', bold)
        all_sheet.write('A2', 'Name', bold)

        canonical_sheet.write('A1','Canonical neighbors', bold)
        canonical_sheet.write('A2', 'Name', bold)

        for i in range(self.N_NODES):
            name = self.names[i][1]
            if name == '/':
                name = self.names[i][0]

            # Label corresponding column
            all_sheet.write(1, i+1, name[0], bold)
            canonical_sheet.write(1, i+1, name[0], bold)


            all_n = list(set().union(*list(all_neighbors[i])))

            for j,ai in enumerate(all_n):

                name_j = self.names[ai][1]
                if name_j == '/':
                    name_j = self.names[ai][0]

                all_sheet.write(2+j, i+1, name_j[0])

            for j,ci in enumerate(canonical_neighbors[i]):
                name_j = self.names[ci][1]
                if name_j == '/':
                    name_j = self.names[ci][0]

                canonical_sheet.write(2+j, i+1, name_j[0])

        workbook.close()

        return

    def compute_canonical_community_assignments(self, graphs, resolutions, 
        p=0.25):

        com_resolutions = [{'resolution': i} for i in resolutions]
        com_assignments = {}

        # Loop through all graphs
        for k, g in graphs.items():

            # Loop through all resolutions for community detection
            ca = g.compute_community_assignments(com_resolutions)
            for k, v in ca.items():
                com_assignments[k] = v

        # Merge community assignments: loop through nodes,
        # count number of times every other node appears as a neighbor;
        # any node that co-occurs in at least p percent of communities
        # is added to the canonical community list
        self.find_canonical_communities(com_assignments, p)

        return

################################################################################
if __name__ == "__main__":

    # Parse args from cmdline; use these defaults otherwise
    parser = argparse.ArgumentParser('')
    parser.add_argument('--filename', type=str, default="../data/sym_z.mat") # "../data/raw_norm.mat"
    parser.add_argument('--fieldname', type=str, default="sym_z")  # "TwohrRowColumnNormalized"
    parser.add_argument('--resultspath', type=str, default="../")
    parser.add_argument('--figpath', type=str, default="../figures/")
    parser.add_argument('--gene_name_fn', type=str, default="../data/names.mat") # "../data/Gene_Names.csv"
    parser.add_argument('--n_perm', type=int, default=0)
    parser.add_argument('--edge_inclusion_method', type=str, default='kovacs_maxent') # options: ['kovacs_maxent', 'kovacs_maxent_old', 'multi_z']
    parser.add_argument('--save_edgelists', type=int, default=0)

    args = parser.parse_args()

    ############################################################################
    # Make interactome object
    interactome = InteractomeDataset(args)

    # Generate graph w/ threshold for edges + save for visualization
    threshold = 0.5 if args.edge_inclusion_method == 'kovacs_maxent' else 20
    graph = interactome.create_single_graph('threshold', threshold)

    # Fig 2a: Graph of community structure
    #interactome.plot_community_graph(graph, 0.75)

    # Generate additional versions of the graph for analysis
    sparsities = None
    thresholds = np.linspace(2, 40, 20)
    graphs = interactome.create_graphs(sparsities, thresholds)

    # Run community detection analysis
    resolutions = np.linspace(0.1, 5, 20)
    interactome.compute_canonical_community_assignments(graphs, resolutions)
