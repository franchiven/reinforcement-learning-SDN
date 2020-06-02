from __future__ import print_function
import networkx as nx
from networkx.readwrite import json_graph
import json
import pickle

def computeL2FwdTables(g):
    """ Given a network described by the graph g returns a dictionary of forwarding
     tables indexed by switch name. Each forwarding table is itself a dictionary
     indexed by a destination MAC address.
    """
    # Figure out which nodes are hosts or switches
    nodes = g.nodes()
    hosts = [n for n in nodes if g.node[n]['type'] == 'host']
    switches = [n for n in nodes if g.node[n]['type'] == 'switch']
    #print(hosts, switches) #prints hosts and swicthes H1... S1...
    print()
    
    # Create the switch to host mapping, i.e., lists of hosts associated with
    # switches
    switch_host_map = {}
    for s in switches:
        switch_host_map[s] = []
        #print(switch_host_map) #places switches in the dict
    for h in hosts:
        hedges = list(g.edges(h)) # Modification for NetworkX 2.0
        if len(hedges) != 1:
            raise Exception("Hosts must be connected to only one switch in this model")
        other = hedges[0][1]  # Should be the other side of the link
        if not other in switches:
            raise Exception("Hosts must be connected only with a switch in this model")
        switch_host_map[other].append(h)  #Okay add the host to the switch map
        #print(switch_host_map) # u'S11' : [] --> u'S11':[H11] for each host switch pair. 

    # Get switch only subgraph and compute all the shortest paths with NetworkX
    g_switches = g.subgraph(switches)
    g_switchesOut = g_switches.edges()
    print(g_switchesOut)

#The Following section was added by Francesco Venerandi -------------------------


    with open('g_switchesOut.pickle', 'wb') as fp:
        pickle.dump(g_switchesOut, fp)
    

    # compute all the shortest paths, result is a dictionary index by two nodes
    # and returning a list of nodes. From this we can get the next hop link to
    # any destination switch
    
    #spaths = nx.shortest_path(g_switches, weight='weight')
    import net_NP
    spaths = net_NP.computeShortestPaths()
    #{u'S9': {u'S9': [u'S9'], u'S8': [u'S9', u'S8']  --> Start at S9. To get to S8 from S8 --> S9-S8-...


#---------------------------------------------------------------------------------

    
    # Compute next hop port forwarding table for switches
    next_hop_port = {}
    for s_src in switches:
        for s_dst in switches:
            if s_src != s_dst:
                path = spaths[s_src][s_dst] #[u'S9', u'S7', u'S4', u'S1'] --> from S9 to S1
                #print(path) 
                next_hop = path[1]  # Get the next hop along path from src to dst. ^ S7
                #print(path[1])
                #print()
                port = g_switches.get_edge_data(s_src,next_hop)["ports"][s_src]
                #print(port)
                next_hop_port[(s_src, s_dst)] = port #Which ports they leave and arrive on. 
                #print(port)

    # Create MAC based forwarding table for each switch from previous table
    # and direct switch to host links
    mac_fwd_table = {}
    for s_src in switches:
        mac_fwd_table[s_src] = {}  # Initialize forwarding table for each source switch
        for s_dst in switches:
            if s_src != s_dst:
                for h in switch_host_map[s_dst]:
                    h_mac = g.node[h]['mac']
                    #print(h_mac)
                    mac_fwd_table[s_src][h_mac] = next_hop_port[(s_src, s_dst)]
                   # print(mac_fwd_table)
            else:  # Host is directly connected to the switch
                for h in switch_host_map[s_dst]:
                    port = g.get_edge_data(s_src,h)["ports"][s_src]
                    h_mac = g.node[h]['mac']
                    mac_fwd_table[s_src][h_mac] = port
                    #print(port) #Prints all port 1s.
    #print(mac_fwd_table)
    return mac_fwd_table

# A small adjustment in port representation from the JSON needed prior
# to converting to NetworkX's internal format.
def in_adjust_ports(gnl_dict):
    """ Converts from ports {"srcPort": num1, "trgPort": num2} format to
        ports {"SrcNodeId": num1, "TrgNodeId": num2} format.
    """
    for link in gnl_dict["links"]:
        if "ports" in link:
            ports = link["ports"]
            new_ports = {
                gnl_dict['nodes'][link["source"]]['id']: ports["srcPort"],
                gnl_dict['nodes'][link["target"]]['id']: ports["trgPort"]}
            link["ports"] = new_ports
    return gnl_dict

# Adjust for NetworkX version. If version 2 run this on old files
# NetworkX 2.0 uses node ids for link source and target rather than
# position in node list, i.e., and integer index.
def adjust_nodeIds(gnl_dict):
    nodes = gnl_dict["nodes"]
    for link in gnl_dict["links"]:
        link["source"] = nodes[link["source"]]["id"] # Gets source id
        link["target"] = nodes[link["target"]]["id"]
    return gnl_dict


if __name__ == '__main__':
    gnl = in_adjust_ports(json.load(open("PanEuroNet.json"))) #####Place own topology here
    gnl = adjust_nodeIds(gnl) # for reading old files with NetworkX 2.0
    g = json_graph.node_link_graph(gnl)
    fwdTable = computeL2FwdTables(g)
    for s in list(fwdTable.keys()):
        print("Switch {} forwarding table:".format(s))
        print(fwdTable[s])
