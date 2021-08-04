import util
import networkx as nx
from pathlib import Path
from collections import Counter
import click
from multiprocessing import Pool
import pickle
import random
import networkx as nx

def create_linker(sub, year):
    linker = {}
    user_comment_count = Counter()
    for file in Path(f'data/subreddit_comments/{sub}').glob(f'{year}-*.csv'):
        log.debug(f"Reading {file}.")
        for comment in util.iter_comments(file):
            user_comment_count[comment['author']] += 1
            reply_to_id = comment['parent_id'][3:] if comment['parent_id'].startswith('t1_') else None
            linker[comment['id']] = {'author': comment['author'], 'reply_to_id': reply_to_id}
    util.save_counts(user_comment_count, f"model/community/{sub}_{year}_user_comment_counts.txt")
    return linker, user_comment_count

def create_graph(linker):
    G = nx.Graph()
    replies = (comment for  comment in linker.values() if comment['reply_to_id'])
    for comment1 in replies:
        comment2 = linker.get(comment1['reply_to_id'])
        if not comment2: # parent comment was deleted, made in a previous year, etc.
            continue
        user1, user2, = comment1['author'], comment2['author'] 
        if (user1, user2) in G.edges:
            G.edges[(user1, user2)]['interactions'] += 1
        else:
            G.add_edge(user1, user2, interactions=1)
    return G

def prune_network(G, subset_users=None):
    # Subset down to the provided nodes
    if subset_users:
        G = G.subgraph(subset_users).copy()
    # Subset down to the largest components.
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    largest_component = components[component_sizes.index(max(component_sizes))]
    G.remove_nodes_from(user for user in set(G.nodes) if not user in largest_component)
    return G, len(components)

def load_graph(sub, year):
    G = nx.read_edgelist(f'model/community/{sub}_{year}_user_interactions.txt', data=[('interactions', int)])
    return G


def compute_weak_clustering(G):
    G_weak = G.copy()
    G_weak.remove_edges_from([e for e in G.edges if G.edges[e]['interactions'] > 1])
    G_weak.remove_nodes_from(list(nx.isolates(G_weak)))
    return nx.clustering(G_weak)

def compute_strong_clustering(G):
    G_strong = G.copy()
    G_strong.remove_edges_from([e for e in G.edges if G.edges[e]['interactions'] <= 1])
    G_strong.remove_nodes_from(list(nx.isolates(G_strong)))
    return nx.clustering(G_strong)

def get_active_users(sub, year):
    user_comment_count = util.load_counts(f"model/community/{sub}_{year}_user_comment_counts.txt")
    active_users = [u for u in user_comment_count if user_comment_count[u] >= 10]
    return active_users

def process(sub, year, log):

    G = load_graph(sub, year)

    ### SHORTEST PATH

    # for e in G.edges: # define the distance between nodes as the inverse of their numeber of interactions
        # G.edges[e]['distance'] = 1/G.edges[e]['interactions']

    # size_all = len(G)
    # G, n_components = prune_network(G, active_users)
    # size_active = len(G)
    # log.info(f"{sub} {year} Original size: {size_all}.")
    # log.info(f"{sub} {year} Found {n_components} active components.")
    # log.info(f"{sub} {year} Pruned to {size_active} active users.")

    # user_sample = random.sample(list(G.nodes),1677) # size of smallest active community network (r/exjw)
    # user_sample = pickle.load(open(f"model/community/{sub}_{year}_avg_path_length_active_unweighted.pickle", 'rb')).keys()
    # avg_path_length = {}
    # for i, node in enumerate(user_sample):
        # if (i % 100 == 0):
            # log.info(f"{sub} {year} path length progress: {i}/{len(user_sample)}.")
        # path_length=nx.single_source_dijkstra_path_length(G, node, weight='distance')
        # avg_path_length[node] = sum(path_length.values()) / len(path_length)
    # with open(f'model/community/{sub}_{year}_avg_path_length_active_weighted.pickle', 'wb') as f:
        # pickle.dump(avg_path_length, f)
    ### END SHORTEST PATH

    ### CLUSTERING COEFF

    log.debug(f"{sub:<15} | computing clustering (full)")
    cluster_coeff_full   = nx.clusterig(G)
    # with open(f'model/community/{sub}_{year}_clustering_full.pickle', 'wb') as f:
        # pickle.dump(cluster_coeff_full, f)
    avg_cluster_full = sum(cluster_coeff_full.values()) / len(cluster_coeff_full)

    log.debug(f"{sub:<15} | computing clustering (weak)")
   
    cluster_coeff_weak = clustering_weak(G)
    with open(f'model/community/{sub}_{year}_clustering_weak.pickle', 'wb') as f:
        pickle.dump(cluster_coeff_weak, f)
    avg_cluster_weak = sum(cluster_coeff_weak.values()) / len(cluster_coeff_weak)

    log.debug(f"{sub:<15} | computing clustering (strong)")
    cluster_coeff_strong = compute_strong_clustering(G) 
    with open(f'model/community/{sub}_{year}_clustering_strong_corrected.pickle', 'wb') as f:
        pickle.dump(cluster_coeff_strong, f)
    avg_cluster_strong = sum(cluster_coeff_strong.values()) / len(cluster_coeff_strong)

    log.info(f"{sub:<15} | strong: {avg_cluster_strong:0.4f} weak: {avg_cluster_weak:0.4f} full: {avg_cluster_full:0.4f}")
    ### END CLUSTERING COEFF

@click.command()
@click.option('--chosen-subs-file', type=str, default='chosen_subs.txt',
        help="File to save the list of chosen subs to.")
@click.option('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")
@click.option('--debug/--no-debug', default=False,
        help="Print debugging info.")
def cli(chosen_subs_file, n_processes, debug):
    log = util.create_logger(f"network", 'network.log', debug)
    subs = util.get_subs(chosen_subs_file)
    years = (2015,)

    args = [(sub, year, log) for sub in subs for year in years]
    with Pool(processes=n_processes) as p:
        p.starmap(process, args)

if __name__ == '__main__':
    cli()

