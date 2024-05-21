import torch
from torch_geometric.datasets import ZINC
import networkx as nx
from torch_geometric.utils import to_networkx
from multiprocessing import Pool, cpu_count

# Define the directory to store the dataset (if it's not already downloaded)
dataset_directory = 'data/ZINC'

# Download the dataset if it's not already downloaded, and load it
dataset = ZINC(dataset_directory)

def count_cycles_in_graph(g):
    """Counts the number of cycles in a single graph."""
    cycle_basis = nx.cycle_basis(g)
    return len(cycle_basis)

def count_cycles_in_graph_parallel(data):
    """Counts the number of cycles in a single graph in parallel."""
    g = to_networkx(data, to_undirected=True)
    return count_cycles_in_graph(g)

def count_total_cycles_parallel(dataset):
    """Counts the total number of cycles in the dataset using parallel processing."""
    # Get the number of CPU cores available
    num_cores = cpu_count()

    # Create a Pool of worker processes
    with Pool(num_cores) as p:
        # Map the dataset to worker processes for parallel processing
        cycle_counts = p.map(count_cycles_in_graph_parallel, dataset)

    # Sum up the cycle counts from all graphs
    total_cycles = sum(cycle_counts)
    return total_cycles

# Count the total number of cycles in the ZINC dataset using parallel processing
total_cycles = count_total_cycles_parallel(dataset)
print(f"Total number of cycles in the ZINC dataset: {total_cycles}")
