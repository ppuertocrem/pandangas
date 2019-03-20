#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Implementation of the simulation results gathering methods.

    Usage:

    >>> import pandangas as pg

    >>>

"""

# TODO: proper usage of node VS bus

from math import pi
import operator

from geopandas import GeoDataFrame

import pandangas.topology as top
import pandangas.simu_linear as sim_ln
import pandangas.simu_nonlinear as sim_nl
from pandangas.utilities import get_index


def _v_from_m_dot(diam, m_dot, fluid):
    q = m_dot / fluid.rho
    a = pi * diam ** 2 / 4
    return q / a


def runpp(net, t_grnd=10 + 273.15, method="NON-LINEAR"):

    # Reset results data-frames
    net.res_bus.drop(net.res_bus.index, inplace=True)
    net.res_pipe.drop(net.res_pipe.index, inplace=True)
    net.res_feeder.drop(net.res_feeder.index, inplace=True)
    net.res_station.drop(net.res_station.index, inplace=True)

    # Set results for pipes not in service
    for pipe in net.pipe.loc[net.pipe["in_service"] == False, "name"].values:
        idx = get_index(pipe, net.pipe)
        net.res_pipe.loc[idx] = [pipe, 0.0, 0.0, 0.0, 0, pipe["geometry"]]

    # Run simulation by pressure level (from lower to higher)
    sorted_levels = sorted(net.LEVELS.items(), key=operator.itemgetter(1))
    for level, value in sorted_levels:
        # Check if level exists
        # TODO: Why not looping over net.bus["level"].unique() ???
        if level in net.bus["level"].unique():
            g = top.graphs_by_level_as_dict(net)
            graph = g[level]
            p_nodes, m_dot_pipes, m_dot_nodes, fluid = {"NON-LINEAR": sim_nl, "LINEAR": sim_ln}[method].run_one_level(
                net, level
            )

            # Set p_node value in results
            for (node, data), p in zip(graph.nodes(data=True), p_nodes):
                net.res_bus.loc[data["index"]] = [
                    node,
                    round(p),
                    round(p * 1e-5, 2),
                    data["geometry"]
                ]

            # Set m_dot_pip value in results
            data_pipes = [data for u, v, data in graph.edges(data=True) if data["type"] == "PIPE"]
            for data, m_dot in zip(data_pipes, m_dot_pipes):
                v = _v_from_m_dot(data["D_m"], m_dot, fluid)
                net.res_pipe.loc[data["index"]] = [
                    data["name"],
                    m_dot,
                    round(v, 2),
                    round(m_dot * net.LHV, 1),
                    round(abs(100 * v / net.V_MAX), 1),
                    data["geometry"]
                ]

            # Set m_dot_pip value in results
            for node, m_dot in zip(graph.nodes, m_dot_nodes):
                if node in net.station["bus_low"].unique():
                    idx_stat = get_index(node, net.station, col="bus_low")
                    stat = net.station.at[idx_stat, "name"]
                    idx = get_index(stat, net.res_station)
                    net.res_station.loc[idx] = [
                        stat,
                        -m_dot,
                        -m_dot * net.LHV,
                        round(abs(-100 * m_dot * net.LHV / net.station.at[idx_stat, "p_lim_kW"]), 1),
                    ]

                if node in net.feeder["bus"].unique():
                    idx_feed = get_index(node, net.feeder, col="bus")
                    feed = net.feeder.at[idx_feed, "name"]
                    idx = get_index(feed, net.res_feeder)
                    net.res_feeder.loc[idx] = [
                        feed,
                        m_dot,
                        m_dot * net.LHV,
                        round(abs(100 * m_dot * net.LHV / net.feeder.at[idx_feed, "p_lim_kW"]), 1),
                    ]


def geospatialize(net, crs='epsg:2056'):
    for df in ["bus", "pipe"]:
        setattr(net, df, GeoDataFrame(getattr(net, df), crs=crs))
        setattr(net, "res_"+df, GeoDataFrame(getattr(net, "res_"+df), crs=crs))
