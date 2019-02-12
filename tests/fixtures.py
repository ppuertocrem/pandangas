import pytest

import networkx as nx

from pandangas import pandangas as pg

@pytest.fixture()
def fix_create():
    net = pg.create_empty_network()

    busf = pg.create_bus(net, level="MP", name="BUSF")
    bus0 = pg.create_bus(net, level="MP", name="BUS0")

    bus1 = pg.create_bus(net, level="BP", name="BUS1")
    bus2 = pg.create_bus(net, level="BP", name="BUS2")
    bus3 = pg.create_bus(net, level="BP", name="BUS3")

    pg.create_load(net, bus2, p_kW=10.0, name="LOAD2")
    pg.create_load(net, bus3, p_kW=15.0, name="LOAD3")

    pg.create_pipe(net, busf, bus0, length_m=1000, diameter_m=0.05, name="PIPE0")
    pg.create_pipe(net, bus1, bus2, length_m=4000, diameter_m=0.05, name="PIPE1")
    pg.create_pipe(net, bus1, bus3, length_m=5000, diameter_m=0.05, name="PIPE2")
    pg.create_pipe(net, bus2, bus3, length_m=3000, diameter_m=0.05, name="PIPE3")

    pg.create_station(net, bus0, bus1, p_lim_kW=50, p_Pa=1.022e5, name="STATION")
    pg.create_feeder(net, busf, p_lim_kW=50, p_Pa=4.5e5, name="FEEDER")

    return net

@pytest.fixture()
def simple_network():
    g = nx.graph_atlas(150)
    h = nx.OrderedDiGraph()
    for u, v in g.edges():
        h.add_edge(u, v)

    net = pg.create_empty_network()

    for n in h.nodes:
        pg.create_bus(net, level="BP", name="BUS{}".format(n))

    for n in [0, 3, "F"]:
        pg.create_bus(net, level="MP", name="BUSMP{}".format(n))

    for u, v in h.edges:
        pg.create_pipe(
            net, "BUS{}".format(u), "BUS{}".format(v), length_m=1E4, diameter_m=0.05, name="PIPE{}-{}".format(u, v)
        )

    for i in [2, 4, 5]:
        pg.create_load(net, "BUS{}".format(i), p_kW=10.0, name="LOAD{}".format(i))

    for i in [0, 3]:
        bus_mp = "BUSMP{}".format(i)
        bus_bp = "BUS{}".format(i)
        pg.create_station(net, bus_mp, bus_bp, p_lim_kW=50, p_Pa=1.022e5, name="STATION{}".format(i))

    pg.create_pipe(net, "BUSMP0", "BUSMP3", length_m=300, diameter_m=0.05, name="PIPEMP0-3")
    pg.create_pipe(net, "BUSMPF", "BUSMP0", length_m=300, diameter_m=0.05, name="PIPEMPF-0")
    pg.create_feeder(net, "BUSMPF", p_lim_kW=50, p_Pa=0.9e5, name="FEEDER")

    return net
