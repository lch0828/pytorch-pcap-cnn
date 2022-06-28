from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from joblib import Parallel, delayed #not yet implemented
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.utils import rdpcap
from scapy.compat import raw
from scapy.layers.l2 import Ether
from scapy.layers.ipsec import ESP
from prefix import PREFIX_TO_TRAFFIC_ID, ID_TO_TRAFFIC


def get_packet_layers(packet):
    counter = 0
    while True:
        layer = packet.getlayer(counter)
        if layer is None:
            break

        yield layer
        counter += 1
        
def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'
    if TCP in packet:
        packet[TCP].remove_payload()
    elif UDP in packet:
        packet[UDP].remove_payload()

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def packet_to_sparse_array(packet, max_length=100):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255

    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    arr = sparse.csr_matrix(arr)
    return arr

def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True
    
    if TCP not in packet and UDP not in packet:
        return True

    return False

def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)

    return arr

class pre_PCAP():
    def __init__(self, data_root):
        filename = ''
        BASE_PATH = '/pre'
        classnum = np.zeros(shape=(10)) #number of sample per class
        for traffic in os.listdir(data_root):
            self.data=[]
            self.label=[]
            print(traffic)
            traffic_path = os.path.join(data_root, traffic)
            if traffic.endswith('.pcap'):
                filename = traffic[:-5]
                path = Path(traffic_path)
                packets = rdpcap(str(path))
                flag = 0
                traffic_label = 0
                for i,j in enumerate(packets):
                    arr = transform_packet(j)
                    if arr is not None:
                        prefix = path.name.split('.')[0].lower()
                        traffic_label = PREFIX_TO_TRAFFIC_ID.get(prefix)
                        pkg = arr.todense().tolist()[0]
                        self.label.append(traffic_label)
                        self.data.append(pkg)
                        flag += 1
                    classnum[traffic_label] += flag
                cwd = os.getcwd()
                print(flag)
                self.label = np.array(self.label)
                self.data = np.array(self.data)
                if flag != 0:
                    np.savez(os.path.join(cwd+BASE_PATH, filename+'.npz'), data=self.data, label=self.label)
                print('saved')
        #for i in classnum:
        #  print(i)

