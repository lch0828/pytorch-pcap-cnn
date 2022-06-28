from preprocess import pre_PCAP
import sys


if __name__ == '__main__':
    path = sys.argv[1]
    train_dataset = pre_PCAP(path) #file is saved to ./pre
