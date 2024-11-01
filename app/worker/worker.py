import os
import uuid
import io
from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

import redis
from rq import Worker, Queue, Connection, get_current_job

listen = ['default']
redis_url = os.environ.get('REDIS_HOST')
conn = redis.from_url(redis_url)


def plot_homo_molecule(graph):
    from simg.config import COLOR_MAPPER
    # returns bytes

    f, (ax) = plt.subplots(1, 1, figsize=(10, 10))

    G = nx.Graph()
    G.add_nodes_from([
        (i, {"color": COLOR_MAPPER[symbol]}) for i, symbol in enumerate(graph['symbol'])
    ])

    colors = [G.nodes[n]['color'] for n in G.nodes()]

    if type(graph.edge_index) is not list:
        edges = [tuple(t) for t in graph.edge_index.T.tolist()]
    else:
        edges = graph.edge_index

    G.add_edges_from(edges)
    nx.draw_kamada_kawai(G, node_color=colors, with_labels=True, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(f)
    return buf.getvalue()


def plot_interaction_matrix(preds, symbol, index):
    f, (ax) = plt.subplots(1, 1, figsize=(15, 15))

    lp_lp = deepcopy(preds)
    bnd_lp = deepcopy(preds)
    lp_bnd = deepcopy(preds)
    bnd_bnd = deepcopy(preds)

    mat = np.array(
        [[f"{a}->{b}" for b in symbol] for a in symbol]
    )

    # set all values that are not LP to None
    lp_lp[mat != "LP->LP"] = None
    bnd_lp[mat != "LP->BND"] = None
    lp_bnd[mat != "BND->LP"] = None
    bnd_bnd[mat != "BND->BND"] = None

    extent = [
        index[0],
        index[-1],
        index[-1],
        index[0],
    ]
    ax.imshow(lp_lp, cmap="Reds", extent=extent)
    ax.imshow(bnd_lp, cmap="Purples", extent=extent)
    ax.imshow(lp_bnd, cmap="Purples", extent=extent)
    ax.imshow(bnd_bnd, cmap="Blues", extent=extent)
    # ticks
    # array of x-axis ticks method.index_1 from 10 to 10
    index_new = np.arange(
        index[0], index[-1], int(len(index) // 6)
    )
    ax.set_xticks(index_new, fontsize=1)
    ax.set_yticks(index_new, fontsize=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(f)

    return buf.getvalue()


def submit_request(param):
    from simg.model_utils import pipeline
    from simg.data import get_connectivity_info

    job = get_current_job(conn)

    job.meta['status'] = 'Running'
    job.save_meta()

    if '\n' not in param:
        path = uuid.uuid4().hex
        xyz_path = f"{path}.xyz"
        smi_path = f"{path}.smi"

        with open(smi_path, "w") as f:
            f.write(param + '\n')

        os.system(f'obabel -i smi {smi_path} -o xyz -O {xyz_path} --gen3d >/dev/null 2>&1')

        with open(xyz_path, "r") as f:
            xyz = f.read()

        job.meta['status'] = 'Convered to xyz'
        job.save_meta()

        os.remove(xyz_path)
        os.remove(smi_path)
    else:
        xyz = param

    xyz_data = [l + '\n' for l in xyz.split('\n')[2:-1]]
    symbols = [l.split()[0] for l in xyz_data]
    coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
    connectivity = get_connectivity_info(xyz_data)

    job.meta['status'] = 'Running pipeline'
    job.save_meta()

    results = pipeline(symbols, coordinates, connectivity, use_threshold=False)

    job.meta['status'] = 'Completed'
    job.save_meta()

    print(results[2][0])

    return {
        "smiles": param if '\n' not in param else '.xyz file',
        "graph": plot_homo_molecule(results[0]),
        "node_level": results[3][1],
        "is_atom": results[0].is_atom.bool().numpy(),
        "is_lp": results[0].is_lp.bool().numpy(),
        "is_bond": results[0].is_bond.bool().numpy(),
        "interactions": plot_interaction_matrix(*results[2])
    }


if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
