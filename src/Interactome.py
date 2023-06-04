import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
import scipy, os, itertools, argparse
from collections import defaultdict
try:
    from src import ProteinGraph as pg
except:
    import ProteinGraph as pg 


class InteractomeDataset(object):

    def __init__(self, args):

        # Bind relevant information
        self.filename              = args.filename
        self.fieldname             = args.fieldname
        self.resultspath           = args.resultspath
        self.figpath               = args.figpath
        self.n_perm                = args.n_perm
        self.do_nulls              = args.n_perm > 0
        self.edge_inclusion_method = args.edge_inclusion_method

        self.expression_fn = args.expression_fn
        self.plasmid_fn    = args.plasmid_fn
        self.gene_name_fn  = args.gene_name_fn

        # Load unprocessed data
        self.raw_data = self.load_raw_data()
        self.N_NODES  = self.raw_data.shape[0]

        # Load expression levels; if relevant information not passed,
        # None returned, and no adjustment of interactions performed
        self.names, exp_level_data = self.load_expression_levels()
        if exp_level_data is not None:
            self.raw_data = self.normalize_by_expression_levels(self.raw_data,
                exp_level_data)

    ############################################################################
    # Data loading/pre-processing methods
    def load_raw_data(self):
        """ 
        Load + return raw dataset of interactions.
        """
        return sio.loadmat(self.filename)[self.fieldname]

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

        base_graph = pg.ProteinGraphBase(merged_scores, self.names, self.figpath)

        # Iterating through sparsity-formed graphs - by sparsity or threshold
        rel_var = sparsities if sparsities is not None else thresholds
        for j, level in enumerate(rel_var):
            print(f"{level:.3f}")

            results_fn = os.path.join(self.resultspath, "graphs/"
                    f"{level:.1e}_weights.txt.gz")
            if os.path.exists(results_fn):
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
            if do_nulls:
                
                null_pref = os.path.join(self.resultspath, "nulls/",
                    f"{level:.1e}/")
                dp_pref = null_pref + "deg_pres/degree_preserved_"
                w_pref = null_pref + "w_perm/w_perm_"
                e_pref = null_pref + "e_perm/e_perm_"

                # Weight/edge permutation, deg. pres. rand
                for i in range(self.n_perm):
                    suff = f"perm_{str(i)}.txt"
                    w_perm = G.weight_permutation(i, w_pref + suff)
                    e_perm = G.edge_permutation(i, e_pref + suff)
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


        # Kovacs max-ent method:
        elif self.edge_inclusion_method == 'kovacs_maxent':

            # Load hits directly, abuse form_by_threshold w/ threshold
            # 1; for now, assign all edges the same weight
            edges = sio.loadmat("../data/maxent_edgelist.mat")['edgelist']
            merged_scores = np.zeros_like(self.raw_data)
            merged_scores[edges[:,0].astype(np.int32), edges[:,1].astype(np.int32)] = 1


        base_graph = pg.ProteinGraphBase(merged_scores, self.names, self.figpath)

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
    def find_canonical_communities(self, comms, p):

        # Across all versions of the graph and resolution values,
        # find every node's community, compute Jaccard similarity score
        all_neighbors = defaultdict(list)
        canonical_neighbors = {}
        community_similarity_scores = np.zeros(self.N_NODES)

        # Loop through graph types/resolution values and nodes 
        for k, comm in comms.items():
            for i in list(comm.keys()):

                # In every version of the community breakdown, store the nodes
                # that belong to the same community
                fellow_community_members = [k for k, v in comm.items() if v == comm[i]]
                all_neighbors[i].append(fellow_community_members)

        # Score community constancy - across different graph
        # thresholds and resolution values, what is the Jaccard similarity b/w 
        # communities?
        for node in list(all_neighbors.keys()):
            pairs = list(itertools.combinations(np.arange(len(comms)), 2))
            jaccards = np.zeros((len(pairs), len(pairs)))

            # Jaccard similarities for graph generation pairs
            for i, j in pairs:
                comm_i = set(all_neighbors[node][i])
                comm_j = set(all_neighbors[node][j])
                intersection = comm_i.intersection(comm_j)
                union = comm_i.union(comm_j)
                jaccards[i,j] = len(intersection) / len(union)
            community_similarity_scores[node] = np.mean(jaccards)

            # Identify canonical neighbors by evaluating nodes that frequently appear
            # in community together (greater than p percent of community assignments pair
            # them together)
            flattened = [item for sublist in all_neighbors[node] for item in sublist]
            counts = {i: flattened.count(i) for i in flattened}
            mult_appearing = [k for k, v in counts.items() if v / len(comms) > p]
            canonical_neighbors[node] = mult_appearing

        self.write_community_results(community_similarity_scores, canonical_neighbors,
            all_neighbors)
        
        return

    def write_community_results(self, community_similarity_scores, 
        canonical_neighbors, all_neighbors):

        # Save nodes + neighbors
        cc_dir = os.path.join(self.resultspath, f"community_members/")
        if not os.path.exists(cc_dir):
            os.makedirs(cc_dir)

        for i in range(self.N_NODES):
            name = self.names[i][1]
            if name == '/':
                name = self.names[i][0]
            with open(os.path.join(cc_dir, f"{name}.txt"), 'w') as f:
                f.write(f"{name}\n")
                f.write(f"Community constancy score: {community_similarity_scores[i]}\n\n")
                f.write("All neighbors:\n")
                all_n = list(set().union(*list(all_neighbors[i])))

                f.write("\n".join([f"{str(j)} {self.names[j][0]} {self.names[j][1]}" for j in all_n]))
                f.write("\n\n")
                f.write("Canonical neighbors:\n")
                f.write("\n".join([f"{str(j)} {self.names[j][0]} {self.names[j][1]}" for j in canonical_neighbors[i]]))

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
    parser.add_argument('--filename', type=str, default="../data/raw_norm.mat")
    parser.add_argument('--fieldname', type=str, default="TwohrRowColumnNormalized")
    parser.add_argument('--resultspath', type=str, default="../")
    parser.add_argument('--figpath', type=str, default="../figures/")
    parser.add_argument('--expression_fn', type=str, default=None) # "../data/expression-Table 1.csv"
    parser.add_argument('--plasmid_fn', type=str, default=None) # "../data/plasmids-Table 1.csv"
    parser.add_argument('--gene_name_fn', type=str, default="../data/Gene_Names.csv")
    parser.add_argument('--n_perm', type=int, default=0)
    parser.add_argument('--edge_inclusion_method', type=str, default='kovacs_maxent')

    args = parser.parse_args()

    ############################################################################
    # Make interactome object
    interactome = InteractomeDataset(args)

    # Generate graph w/ threshold for edges + save for visualization
    threshold = 0.5 if args.edge_inclusion_method == 'kovacs_maxent' else 20
    graph = interactome.create_single_graph('threshold', threshold)

    # Run community detection analysis
    resolutions = np.linspace(0.1, 10, 50)
    interactome.compute_canonical_community_assignments(graph, resolutions)
