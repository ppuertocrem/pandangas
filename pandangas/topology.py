import networkx as nx


def create_nxgraph(net, only_in_service=True):
    """
    Convert a given network into a NetworkX MultiGraph

    :param net: the given network
    :param only_in_service: if True, convert only the pipes that are in service (default: True)
    :return: a MultiGraph
    """

    g = nx.OrderedDiGraph()

    for idx, row in net.bus.iterrows():
        g.add_node(
            row["name"], 
            index=idx,
            level=row["level"],
            zone=row["zone"],
            type=row["type"],
            geometry=row["geometry"]
        )

    pipes = net.pipe
    if only_in_service:
        pipes = pipes.loc[pipes["in_service"] != False]

    for idx, row in pipes.iterrows():
        g.add_edge(
            row["from_bus"],
            row["to_bus"],
            name=row["name"],
            index=idx,
            L_m=row["length_m"],
            D_m=row["diameter_m"],
            mat=row["material"],
            type="PIPE",
            geometry=row["geometry"],
        )

    for idx, row in net.station.iterrows():
        g.add_edge(
            row["bus_high"],
            row["bus_low"],
            name=row["name"],
            index=idx,
            p_lim_kw=row["p_lim_kW"],
            p_Pa=row["p_Pa"],
            type="STATION",
        )

    return g


def graphs_by_level_as_dict(net):
    levels = net.bus["level"].unique()
    g = create_nxgraph(net)
    g_dict = {}
    for l in levels:
        nodes = [n for n, data in g.nodes(data=True) if data["level"] == l]
        g_dict[l] = g.subgraph(nodes)
    return g_dict
