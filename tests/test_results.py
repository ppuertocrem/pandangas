#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `results` package."""

import pytest

from pandangas import results as res

from fixtures import simple_network


def test_runpp_nonlinear(simple_network):
    net = simple_network
    res.runpp(net, method="NON-LINEAR")
    print(set(net.res_bus["p_Pa"].values.tolist()))
    assert set(net.res_bus["p_Pa"].values.tolist()) == {
        89988.0,
        101991.0,
        101965.0,
        101998.0,
        102063.0,
        90000.0,
        102200.0,
        89983.0,
    }
    assert set(net.res_pipe["p_kW"].values.tolist()) == {8.0, 9.0, 1.0, 2.5, -5.2, 2.7, -7.7, 0.2, 13.0, 29.9}
    assert set(net.res_pipe["v_m/s"].values.tolist()) == {0.15, 0.17, 0.02, 0.05, -0.1, 0.05, -0.15, 0.0, 0.13, 0.29}


def test_runpp_linear(simple_network):
    net = simple_network
    res.runpp(net, method="LINEAR")
    print(set(net.res_bus["p_Pa"].values.tolist()))
    assert set(net.res_bus["p_Pa"].values.tolist()) == {102188.0, 102190.0, 89999.0, 90000.0, 102193.0, 102200.0}
    assert set(net.res_pipe["p_kW"].values.tolist()) == {8.0, 9.0, 1.0, 2.5, -5.2, 2.7, -7.7, 0.2, 13.0, 29.9}
    assert set(net.res_pipe["v_m/s"].values.tolist()) == {0.15, 0.17, 0.02, 0.05, -0.1, 0.05, -0.15, 0.0, 0.13, 0.29}


def test_columns_of_created_df_nonlinear(simple_network):
    net = simple_network
    res.runpp(net, method="NON-LINEAR")
    assert set(net.res_bus.columns) == {"name", "p_Pa", "p_bar"}
    assert set(net.res_pipe.columns) == {"name", "m_dot_kg/s", "v_m/s", "p_kW", "loading_%"}
    assert set(net.res_feeder.columns) == {"name", "m_dot_kg/s", "p_kW", "loading_%"}
    assert set(net.res_station.columns) == {"name", "m_dot_kg/s", "p_kW", "loading_%"}
    assert len(net.res_bus) == 9
    assert len(net.res_pipe) == 10
    assert len(net.res_feeder) == 1
    assert len(net.res_station) == 2


def test_columns_of_created_df_linear(simple_network):
    net = simple_network
    res.runpp(net, method="LINEAR")
    assert set(net.res_bus.columns) == {"name", "p_Pa", "p_bar"}
    assert set(net.res_pipe.columns) == {"name", "m_dot_kg/s", "v_m/s", "p_kW", "loading_%"}
    assert set(net.res_feeder.columns) == {"name", "m_dot_kg/s", "p_kW", "loading_%"}
    assert set(net.res_station.columns) == {"name", "m_dot_kg/s", "p_kW", "loading_%"}
    assert len(net.res_bus) == 9
    assert len(net.res_pipe) == 10
    assert len(net.res_feeder) == 1
    assert len(net.res_station) == 2
