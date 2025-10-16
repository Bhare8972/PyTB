"""
Simple version of initial TB-reading test script

Currently expects a tcpdump .txt file as input,
and produces an output HDF5 file + some debug plots

"""
import argparse
import re
import time
import read

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='Input tcpdump txt file(s)')
    parser.add_argument('--debug', action='store_true', help='Produce debug plots')

    args = parser.parse_args()

    filename = time.strftime("%Y%M%dT%H%M%S") + '.hdf5'
    # to find UDP packets in the tcpdump file,
    # we use regex to match to the start of a packet
    packet_start_re = re.compile(
        '\s+\d+\.\d+\.\d+\.\d+\.\d+ > \d+\.\d+\.\d+\.\d+\.\d+.*UDP.*')

    writer = read.HDF5Writer(debug=args.debug)

    for inputfile in args.input:
        with open(inputfile, 'r') as f:
            lines = f.readlines()

        current_packet = bytes()
        found_packet = False
        for l in lines:
            if not found_packet:
                found_packet = bool(packet_start_re.match(l))
                continue
            else: # we're reading a packet!
                # convert the tcpdump line to bytes
                # We assume that if this fails, we're at the end of a packet
                try:
                    l_stripped = l.split(': ')[1].split('  ')[0].replace(' ', '')
                    current_packet += bytes(
                        [
                            int(l_stripped[2*i:2*i+2], 16)
                            for i in range(len(l_stripped) // 2)])
                except Exception: # we've probably reached the end of the packet
                    writer.parse(current_packet)
                    current_packet = bytes()
                    found_packet = False

    writer.write(filename=filename)


