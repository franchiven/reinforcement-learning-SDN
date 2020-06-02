import json
from networkx.readwrite import json_graph
import ShortestPathBridgeNet_NP as spbN_NP
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls
from ryu.cmd import manager # For directly starting Ryu
import sys  # For getting command line arguments and passing to Ryu
import eventlet
from eventlet import backdoor  # For telnet python access
from ryu.ofproto import ofproto_v1_0 # This code is OpenFlow 1.0 specific

if __name__ == "__main__": # Stuff to set additional command line options
    from ryu import cfg
    CONF = cfg.CONF
    CONF.register_cli_opts([
        cfg.StrOpt('netfile', default=None, help='network json file'),
        cfg.BoolOpt('widest_paths', default=False,
                    help='Use widest path.'),
        cfg.BoolOpt('notelnet', default=False,
                    help='Telnet based debugger.')
    ])

class L2DestForwardStatic(app_manager.RyuApp):
    """
    Waits for OpenFlow switches to connect and then fills their forwarding
    tables based on the network topology json file from the command line.
    """
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        """
        Compute the forwarding tables based on the network description file,
        and whether shortest or widest paths should be used.
        """
        super(L2DestForwardStatic, self).__init__(*args, **kwargs)
        self.netfile = self.CONF.netfile
        self.widest_paths = self.CONF.widest_paths
        self.logger.info("Computing forwarding table for: {}".format(self.netfile))
        self.logger.info("Using widest paths: {}".format(self.widest_paths))
        self.g = json_graph.node_link_graph(json.load(open(self.netfile)))
        if self.widest_paths:
            pass
        else:
            self.fwdTable = spbN_NP.computeL2FwdTables(self.g)
        if not self.CONF.notelnet:
            eventlet.spawn(backdoor.backdoor_server,
                           eventlet.listen(('localhost', 3000)))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures)
    def handle_switchUp(self, event):
        """ This method gets called at the end of switch-controller handshaking.
            We fill in the forwarding tables for the switch here.
        """
        msg = event.msg
        dp = msg.datapath
        self.logger.info("Switch {} came up".format(dp.id))
        self.load_fwd_table(dp)

    def load_fwd_table(self, datapath):
        """ This method does the nitty gritty of creating and sending the
            OpenFlow messages based on the pre-computed forwarding tables."""
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        sname = dpidDecode(dpid)
        self.logger.info("setting up forwarding table for switch {}".format(sname))
        table = self.fwdTable[sname]
        for mac in list(table.keys()):
            self.logger.info("Mac address string {}".format(mac))
            actions = [parser.OFPActionOutput(table[mac])]
            match = parser.OFPMatch(dl_dst=mac)
            mod = parser.OFPFlowMod(
                datapath=datapath, match=match, cookie=0,
                command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
                priority=ofproto.OFP_DEFAULT_PRIORITY,
                flags=ofproto.OFPFF_SEND_FLOW_REM, actions=actions)
            datapath.send_msg(mod)


def dpidDecode(aLong):
    try:
        myBytes = bytearray.fromhex('{:8x}'.format(aLong)).strip()
        return myBytes.decode()
    except ValueError:
        return str(aLong)

if __name__ == "__main__":
    manager.main(args=sys.argv)
