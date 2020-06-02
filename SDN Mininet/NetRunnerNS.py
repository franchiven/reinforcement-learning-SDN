import argparse
import json
from mininet.cli import CLI
from mininet.log import lg
from mininet.node import RemoteController
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from networkx.readwrite import json_graph

# Turns a string, like the node name,or array of bytes into a long
# from http://stackoverflow.com/questions/25259947/convert-variable-sized-byte-array-to-a-integer-long
def bytes_to_int(bytes):
  return int(bytes.encode('hex'), 16)
  
# Takes a int or long and turns it into a hex string without the leading
# 0x or the sometimes trailing L.
def hex_strip(n):
    hexString = hex(n)
    plainString = hexString.split("0x")[1] # Gets rid of the Ox of the hex string
    return plainString.split("L")[0] #Gets rid of the trailing L if any

class GraphTopoFixedAddrPorts(Topo):
    """ Creates a Mininet topology based on a NetworkX graph object where
        hosts have assigned MAC and IP addresses, links ends have assigned port
        numbers, bandwidth and delay. Where the delay comes from the link weight."""
    def __init__(self, graph, **opts):
        listenPort = 6634
        Topo.__init__(self, **opts)
        nodes = graph.nodes()
        node_names = {}
        for node in nodes: # node is the unicode string name of the node
            tmp_node = graph.node[node]
            if tmp_node['type'] == 'switch':
                # Creates a datapath id based on the name as an string of ascii bytes
                # Mininet wants this as a hex string without the 0x or L
                our_dpid = hex_strip(bytes_to_int(node.encode('ascii'))) 
                print "Node: {} dpid: {}".format(node, our_dpid)
                switch = self.addSwitch(node.encode('ascii'), listenPort=listenPort, 
                    dpid=our_dpid)
                listenPort += 1
                node_names[node.encode('ascii')] = switch
            else:
                # The following line also sets MAC and IP addresses
                host = self.addHost(node.encode('ascii'), **tmp_node)
                node_names[node.encode('ascii')] = host
        edges = graph.edges()
        for edge in edges:
            props = graph.get_edge_data(edge[0], edge[1])
            delay = str(props['weight']) + "ms"
            bw = props['capacity']
            port1 = props['ports'][edge[0]]
            port2 = props['ports'][edge[1]]
            self.addLink(node_names[edge[0]],node_names[edge[1]],port1=port1, port2=port2,
                         delay=delay, bw=bw)
    @staticmethod
    def from_file(filename):
        """Creates a Mininet topology from a given JSON filename."""
        f = open(filename)
        tmp_graph = json_graph.node_link_graph(json.load(f))
        f.close()
        print(tmp_graph.nodes(data=True))
        #print(tmp_graph.links())
        #exit()
        return GraphTopoFixedAddrPorts(tmp_graph)



if __name__ == '__main__':
    fname = "PanEuroNet.json"  # You can put your default file here
    remoteIP = " "      # Put your default remote IP here
    # Using the nice Python argparse library to take in optional arguments
    # for file name and remote controller IP address
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", help="network graph file name")
    parser.add_argument("-ip", "--remote_ip", help="IP address of remote controller")
    args = parser.parse_args()
    if not args.fname:
        print "fname not specified using: {}".format(fname)
    else:
        fname = args.fname
    if not args.remote_ip:
        print "remote controller IP not specified using: {}".format(remoteIP)
    else:
        remoteIP = args.remote_ip
    topo = GraphTopoFixedAddrPorts.from_file(fname)
    lg.setLogLevel('info')
    network = Mininet(controller=RemoteController, autoStaticArp=True, link=TCLink)
    network.addController(controller=RemoteController, ip=remoteIP)
    network.buildFromTopo(topo=topo)
    network.start()
    CLI(network)
    network.stop()
