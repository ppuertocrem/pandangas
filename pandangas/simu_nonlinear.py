# -*- coding: utf-8 -*-

"""Non-linear simulation module."""

from math import pi
import numpy as np
import networkx as nx
from scipy.optimize import fsolve

import fluids
import fluids.vectorized as fvec
from thermo.chemical import Chemical

import pandangas.topology as top
from pandangas.utilities import get_index
from pandangas.simu_linear import run_one_level as run_linear

M_DOT_REF = 1e-3

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


def _dp_from_m_dot_vec(m_dot_ad, l, d, e, fluid):
    m_dot = m_dot_ad * M_DOT_REF
    a = np.pi * (d / 2) ** 2
    v = m_dot / a / fluid.rho
    re = fvec.core.Reynolds(v, d, fluid.rho, fluid.mu)
    fd = fvec.friction_factor(re, eD=e / d)
    k = fvec.K_from_f(fd=fd, L=l, D=d)
    return fvec.dP_from_K(k, rho=fluid.rho, V=v) / fluid.P


def _eq_m_dot_sum(m_dot_pipes, m_dot_nodes, i_mat):
    return np.matmul(i_mat, m_dot_pipes) - m_dot_nodes


def _eq_pressure(p_nodes, m_dot_pipes, i_mat, l, d, e, fluid):
    return np.matmul(p_nodes, i_mat) + _dp_from_m_dot_vec(m_dot_pipes, l, d, e, fluid)


def _eq_m_dot_node(m_dot_nodes, gr, loads):
    bus_load = np.array(
        [
            m_dot_nodes[i] - loads[node] / M_DOT_REF
            for i, (node, data) in enumerate(gr.nodes(data=True))
            if data["type"] == "SINK"
        ]
    )
    bus_node = np.array([m_dot_nodes[i] for i, (node, data) in enumerate(gr.nodes(data=True)) if data["type"] == "NODE"])
    return np.concatenate((bus_load, bus_node))


def _eq_p_feed(p_nodes, gr, p_nom, p_ref):
    p_feed = np.array(
        [
            p_nodes[i] - p_nom[node] / p_ref
            for i, (node, data) in enumerate(gr.nodes(data=True))
            if data["type"] == "SRCE"
        ]
    )
    return p_feed


def _eq_model(x, *args):
    mat, gr, lengths, diameters, roughness, fluid, loads, p_nom, p_ref = args
    p_nodes = x[: len(gr.nodes)]
    m_dot_pipes = x[len(gr.nodes) : len(gr.nodes) + len(gr.edges)]
    m_dot_nodes = x[len(gr.nodes) + len(gr.edges) :]

    return np.concatenate(
        (
            _eq_m_dot_sum(m_dot_pipes, m_dot_nodes, mat),
            _eq_pressure(p_nodes, m_dot_pipes, mat, lengths, diameters, roughness, fluid),
            _eq_m_dot_node(m_dot_nodes, gr, loads),
            _eq_p_feed(p_nodes, gr, p_nom, p_ref),
        )
    )


def run_one_level(net, level):
    """

    """
    g = top.graphs_by_level_as_dict(net)[level]

    gas = Chemical("natural gas", T=net.T_GRND, P=net.LEVELS[level])
    loads = _scaled_loads_as_dict(net)
    p_ops = _operating_pressures_as_dict(net)

    p_nodes_i, m_dot_pipes_i, m_dot_nodes_i, gas = run_linear(net, level)
    x0 = np.concatenate((p_nodes_i, m_dot_pipes_i, m_dot_nodes_i))

    i_mat = create_incidence(g)

    leng = np.array([data["L_m"] for _, _, data in g.edges(data=True)])
    diam = np.array([data["D_m"] for _, _, data in g.edges(data=True)])

    materials = np.array([data["mat"] for _, _, data in g.edges(data=True)])
    eps = np.array([fluids.material_roughness(m) for m in materials])

    res = fsolve(_eq_model, x0, args=(i_mat, g, leng, diam, eps, gas, loads, p_ops, gas.P))

    p_nodes = res[: len(g.nodes)] * gas.P
    m_dot_pipes = res[len(g.nodes) : len(g.nodes) + len(g.edges)] * M_DOT_REF
    m_dot_nodes = res[len(g.nodes) + len(g.edges) :] * M_DOT_REF

    return p_nodes, m_dot_pipes, m_dot_nodes, gas
