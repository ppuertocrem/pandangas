#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `simu_linear` package."""

import pytest

import numpy as np
import fluids
from thermo.chemical import Chemical

from pandangas import simu_nonlinear as sim
from pandangas import topology as top
from fixtures import simple_network


def test_dp_from_m_dot():
    gas = Chemical("natural gas", T=10 + 273.15, P=1.022e5)
    material = fluids.nearest_material_roughness("steel", clean=True)
    eps = fluids.material_roughness(material)
    m_dot_ref = 1e-3
    m_dot_ad = np.array([0.95, 1.0, 1.05]) * m_dot_ref
    l = np.array([1e4, 0.8e4, 1.2e4])
    d = np.array([0.05, 0.04, 0.03])
    res = gas.P * sim._dp_from_m_dot_vec(m_dot_ad, l, d, eps, gas)
    assert res.round().tolist() == [1.0, 2.0, 10.0]  # Pa


def test_run_one_level_BP_shape(simple_network):
    net = simple_network
    g = top.graphs_by_level_as_dict(net)
    graph = g["BP"]
    p_nodes, m_dot_pipes, m_dot_nodes, gas = sim.run_one_level(net, "BP")
    assert p_nodes.shape == (len(graph.nodes),)
    assert m_dot_pipes.shape == (len(graph.edges),)
    assert m_dot_nodes.shape == (len(graph.nodes),)


def test_run_one_level_BP_values(simple_network):
    net = simple_network
    g = top.graphs_by_level_as_dict(net)
    graph = g["BP"]
    p_nodes, m_dot_pipes, m_dot_nodes, gas = sim.run_one_level(net, "BP")

    assert p_nodes.round().tolist() == [102200.0, 101991.0, 101965.0, 102063.0, 101998.0, 102200.0]
    assert m_dot_pipes.round(5).tolist() == [2.1e-04, 2.4e-04, 3.0e-05, 7.0e-05, -1.4e-04, 7.0e-05, -2.0e-04, 1.0e-05]
    assert m_dot_nodes.round(5).tolist() == [-0.00045, 0.00026, 0.00026, 0.0, 0.00026, -0.00034]

# TODO: non-linear method do not like (ZeroDivisionError) null mass flows (in dead-end pipes)?
def test_run_with_dead_end_pipes():
    pass
