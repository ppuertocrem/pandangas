# -*- coding: utf-8 -*-

"""Linear simulation module."""

from math import pi
import numpy as np
import networkx as nx

import fluids
from thermo.chemical import Chemical

import pandangas.topology as top
from pandangas.utilities import get_index

# TODO: MOVE TO SPECIFIC FILE (utilities.py ?) ++++++++++
def _scaled_loads_as_dict(net):
    """
    Maps sinks (loads and lower pressure stations) name to scaled load
    """
    loads = {row[1]: round(row[2] * row[4] / net.LHV, 6) for _, row in net.load.iterrows()}  # kW to kg/s
    stations = {}
    for _, row in net.res_station.iterrows():
        idx_stat = get_index(row[0], net.station)
        stations[net.station.at[idx_stat, "bus_high"]] = round(row[1], 6)
    loads.update(stations)
    return loads


def _operating_pressures_as_dict(net):
    """
    Map sources (feeders and higher pressure stations) name to operating pressure
    """
    feed = {row[1]: row[3] for _, row in net.feeder.iterrows()}
    stat = {row[2]: row[4] for _, row in net.station.iterrows()}
    feed.update(stat)
    return feed


def create_incidence(graph):
    """
    Create oriented incidence matrix of the given graph
    """
    return nx.incidence_matrix(graph, oriented=True).toarray()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++


def create_k(graph, fluid):
    """
    Create factor for mass flow in pressure losses equation
    in P_j - P_i + k * m_ij = 0 -> with k = 2⁷*L*mu/(D⁴*pi*rho)
    """
    mu = fluid.mu
    rho = fluid.rho

    L = np.array([d["L_m"] for u, v, d in graph.edges(data=True)])
    D = np.array([d["D_m"] for u, v, d in graph.edges(data=True)])

    k = 2 ** 7 * L * D * mu / (D ** 4 * pi * rho)

    return k


def weird(m):
    """
    weird([1, 0, 0, 1, 0, 1]) = [[1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 1]]
    """
    r = []
    for idx, x in enumerate(m):
        if x == 1:
            rr = [0] * len(m)
            rr[idx] = 1
            r.append(rr)
    return np.array(r)


def create_a(graph, fluid):
    """
    Create the A matrix (for solving A.X = B)
    """

    nbr_pipes = len(graph.edges)
    nbr_nodes = len(graph.nodes)

    # P_j - P_i + k * m_ij = 0 for (i, j) in pipes ---------------------------------------------------------------------
    a00 = create_incidence(graph).T
    a01 = create_k(graph, fluid) * np.eye(nbr_pipes)
    a02 = np.zeros((nbr_pipes, nbr_nodes))

    a0 = np.concatenate((a00, a01, a02), axis=1)

    # sum(m_ki) - sum(m_ik) - m_i = 0 for i in nodes -------------------------------------------------------------------
    a10 = np.zeros((nbr_nodes, nbr_nodes))
    a11 = create_incidence(graph)
    a12 = -1 * np.eye(nbr_nodes)

    a1 = np.concatenate((a10, a11, a12), axis=1)

    # m_i = 0 for i in nodes (passive) & m_i = -c for i in nodes (sink) ------------------------------------------------
    idx_nodes_pass = np.array([1 if d["type"] == "NODE" else 0 for n, d in graph.nodes(data=True)])
    idx_nodes_sink = np.array([1 if d["type"] == "SINK" else 0 for n, d in graph.nodes(data=True)])

    nbr_nodes_pass = sum(idx_nodes_pass)
    nbr_nodes_sink = sum(idx_nodes_sink)

    a20 = np.zeros((nbr_nodes_pass + nbr_nodes_sink, nbr_nodes))
    a21 = np.zeros((nbr_nodes_pass + nbr_nodes_sink, nbr_pipes))
    try:
        a22 = np.concatenate((weird(idx_nodes_pass), weird(idx_nodes_sink)), axis=0)
    except ValueError as e:
        a22 = weird(idx_nodes_sink)

    a2 = np.concatenate((a20, a21, a22), axis=1)

    # P_i = P_nom for i in nodes (source) ------------------------------------------------------------------------------
    idx_nodes_srce = np.array([1 if d["type"] == "SRCE" else 0 for n, d in graph.nodes(data=True)])
    nbr_nodes_srce = sum(idx_nodes_srce)

    a30 = weird(idx_nodes_srce)
    a31 = np.zeros((nbr_nodes_srce, nbr_pipes))
    a32 = np.zeros((nbr_nodes_srce, nbr_nodes))

    a3 = np.concatenate((a30, a31, a32), axis=1)

    # Complete A -------------------------------------------------------------------------------------------------------
    a = np.concatenate((a0, a1, a2, a3))
    return a


def create_b(graph, scaled_loads, op_pressures):
    """
    Create the B matrix (for solving A.X = B)
    """
    nbr_pipes = len(graph.edges)
    nbr_nodes = len(graph.nodes)

    # P_j - P_i + k * m_ij = 0 -----------------------------------------------------------------------------------------
    b0 = np.zeros(nbr_pipes)

    # sum(m_ki) - sum(m_ik) - m_i = 0 for i in nodes -------------------------------------------------------------------
    b1 = np.zeros(nbr_nodes)

    # m_i = 0 for i in nodes (passive) & m_i = -c for i in nodes (sink) ------------------------------------------------
    idx_nodes_pass = np.array([1 if d["type"] == "NODE" else 0 for n, d in graph.nodes(data=True)])
    idx_nodes_sink = np.array([1 if d["type"] == "SINK" else 0 for n, d in graph.nodes(data=True)])

    nbr_nodes_pass = sum(idx_nodes_pass)
    nbr_nodes_sink = sum(idx_nodes_sink)

    b20 = np.zeros(nbr_nodes_pass)
    b21 = np.array([scaled_loads[n] for n, d in graph.nodes(data=True) if d["type"] == "SINK"])

    # P_i = P_nom for i in nodes (source) ------------------------------------------------------------------------------
    b3 = np.array([op_pressures[n] for n, d in graph.nodes(data=True) if d["type"] == "SRCE"])

    # Complete B -------------------------------------------------------------------------------------------------------
    b = np.concatenate([b0, b1, b20, b21, b3])
    return b


def solve(a, b):
    """
    Solve A.X = B using numpy.linalg.solve and check that the solution is correct
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.solve.html
    """
    x = np.linalg.solve(a, b)
    assert np.allclose(np.dot(a, x), b)
    return x


def run_one_level(net, level):
    """

    """
    g = top.graphs_by_level_as_dict(net)[level]

    gas = Chemical("natural gas", T=net.T_GRND, P=net.LEVELS[level])
    loads = _scaled_loads_as_dict(net)
    p_ops = _operating_pressures_as_dict(net)

    a = create_a(g, gas)
    b = create_b(g, loads, p_ops)

    x = solve(a, b)

    p_nodes = x[: len(g.nodes)]
    m_dot_pipes = x[len(g.nodes) : len(g.nodes) + len(g.edges)]
    m_dot_nodes = x[len(g.nodes) + len(g.edges) :]

    return p_nodes, m_dot_pipes, m_dot_nodes, gas
