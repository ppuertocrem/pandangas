#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `simu_linear` package."""

import pytest

import numpy as np
from thermo.chemical import Chemical

from pandangas import simu_linear as sim
from pandangas import topology as top
from fixtures import simple_network


def test_solve():
    # 3 * x0 + x1 = 9 and x0 + 2 * x1 = 8 <=> x0 = 2, x1 = 3
    a = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    assert np.array_equal(sim.solve(a, b), np.array([2.0, 3.0]))


def test_weird():
    a = np.array([1, 0, 0, 1, 0, 1])
    waited = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
    assert np.array_equal(sim.weird(a), waited)


def test_create_a(simple_network):
    gas = Chemical("natural gas", T=10 + 273.15, P=1.022e5)
    net = simple_network
    g = top.graphs_by_level_as_dict(net)
    graph = g["BP"]
    a = sim.create_a(graph, gas)
    assert a.shape == (20, 20)


def test_create_k(simple_network):
    gas = Chemical("natural gas", T=10 + 273.15, P=1.022e5)
    net = simple_network
    g = top.graphs_by_level_as_dict(net)
    graph = g["BP"]
    k = sim.create_k(graph, gas)
    assert k.shape == (len(graph.edges),)
    for ik in k:
        assert int(ik) == 49975


def test_create_b(simple_network):
    net = simple_network
    loads = sim._scaled_loads_as_dict(net)
    p_ops = sim._operating_pressures_as_dict(net)
    g = top.graphs_by_level_as_dict(net)
    graph = g["BP"]
    b = sim.create_b(graph, loads, p_ops)
    assert b.shape == (20,)


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

    assert p_nodes.round().tolist() == [102200.0, 102190.0, 102188.0, 102193.0, 102190.0, 102200.0]
    assert m_dot_pipes.round(5).tolist() == [2.1e-04, 2.4e-04, 3.0e-05, 7.0e-05, -1.4e-04, 7.0e-05, -2.0e-04, 1.0e-05]
    assert m_dot_nodes.round(5).tolist() == [-0.00045, 0.00026, 0.00026, 0.0, 0.00026, -0.00034]
