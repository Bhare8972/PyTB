"""
Module to read in transient-buffer (TB) raw data packets

"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.time
import h5py
import os
import struct
import logging
from collections import defaultdict

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s:%(message)s", datefmt="%H:%M:%S")

logger = logging.getLogger('PyTB.read')

# convert 1-bit f_adc to sampling rate in GHz
sampling_rate_from_f_adc = {0: 0.16, 1: 0.2}

# constant arrays to convert tightly-packed 14-bit data to 16-bits
sign_conversion = np.array([0, 2**13], dtype=np.int16)
bit_multiplier = np.array([0,] + [2**k for k in range(12, -1, -1)], dtype=np.int16)

def unpack_source_info(source_info: int) -> dict:
    """
    Unpack source info into dictionary

    Unpacks the information contained in the long integer
    source info into its labelled components, as described in
    section 3.9.3 L3-SDP to L2-CEP: Transient raw data
    of the LOFAR2.0 STAT to CEP Interface Control Document

    Parameters
    ----------
    source_info : int
        Short (2-byte) integer from raw data packet header

    Returns
    -------
    dict
        Dictionary with the following entries:

        - 'gn_index': global node index (8-bit, used for debugging)
        - 'sample_width': number of bits per sample (4-bit, usually 14);
          note that a 16-bit sample width is stored as 0
        - 'f_adc': bool encoding the sampling rate used
          (0 for 160 MHz, 1 for 200 MHz)
        - 'nyquist_zone_index': index of the Nyquist zone (2-bit)
          0 indicates first Nyquist zone, 1 second Nyquist zone, etc.
        - 'antenna_band_index': 1-bit, 0 for Low Band, 1 for High Band

    """
    # extract bits n+m:n (inclusive) by dividing by 2**n,
    # then take only m+1 bits by using modulo 2**(m+1)
    source_info_dict = dict(
        gn_index = source_info % 2**8,
        sample_width = source_info // 2**8 % 2**4,
        f_adc = source_info // 2**11 % 2,
        nyquist_zone_index = source_info // 2**12 % 4,
        antenna_band_index = source_info // 2**14
    )

    return source_info_dict

def unpack_raw_header(header : bytes, assert_valid : bool = True) -> dict:
    """
    Unpack transient-buffer (TB) header to Python dict.

    Parameters
    ----------
    header : bytes
        The 24-byte transient raw data header.
    assert_valid : bool, default True
        If True, assert that the 'marker', 'version_id', and 'nof_raw_data_per_packet'
        conform to expected values for TB packets.

    Returns
    -------
    dict
        A dict containing the information in the raw data header,
        with the following keys:

        - 'marker' (should be b'r' for raw TB data)
        - 'version_id'
        - 'observation_id'
        - 'station_info'
        - 'source_info' (can be further unpacked using `unpack_source_info`)
        - 'antenna_input_index'
        - 'nof_raw_data_per_packet'
        - 'RSN' (raw sequence number)

    """
    hdr_dict = dict(zip(
        [
            'marker', 'version_id', 'observation_id', 'station_info',
            'source_info', 'antenna_input_index',
            'nof_raw_data_per_packet', 'RSN'],
        struct.unpack('>cBIHHxxxBHQ', header)))

    if assert_valid:
        assert hdr_dict['marker'] == b'r', f"Expect marker 'b'r' but found {hdr_dict['marker']}."
        assert hdr_dict['version_id'] == 2, f"Expect version_id '2' but found {hdr_dict['version_id']}."
        assert hdr_dict['nof_raw_data_per_packet'] == 2000, f"Expect 2000 samples per packet but header claims {hdr_dict['nof_raw_data_per_packet']}."

    return hdr_dict


def unpack_raw_payload(payload: bytes) -> np.ndarray:
    """
    Unpack tightly-packed transient-buffer bytes into numpy int16 array

    Parameters
    ----------
    payload : bytes
        The transient buffer payload (7000 bytes)

    Returns
    -------
    np.ndarray
        The data in the array, as signed 16-bit integers,
        in a numpy array of shape (2000, 2) = (samples, polarizations)

    Notes
    -----
    The raw transient-buffer (TB) data consists of 7000 bytes
    of tightly-packed 14-bit waveform data. This function takes
    the raw bytes as an input, and extracts the 14-bit data by
    treating sets of 4 samples (7 bytes + 1 zero-padding byte)
    as 64-bit unsigned integers, and applying appropriate shifting
    / masking to return the 2000 x 2 (samples x polarizations).
    """
    bit_array = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).reshape((-1, 2, 14))
    values = np.dot(bit_array, bit_multiplier) - sign_conversion[bit_array[..., 0]]

    return values


## This function allocates large arrays three times. (np.unpackbits, np.dot, and subtraction)
## in order to be fast enough this function needs to be written so that large arrays are not allocated. There are two ways to do this. 1) use in-place operators ( -= for example), 
## and 2) allow the user to pass-in scratch space (e.g.  see the 'out' parameter of np.dot)


def parse_packet(packet : bytes) -> dict:
    """
    Convert UDP transient raw data packet

    Parameters
    ----------
    packet : bytes
        The 7066-byte transient raw data packet

    Returns
    -------
    dict
        Dictionary containing the raw transient data
        header and data
    """

    assert len(packet) == 7066, f"Expect packet of length 7066 bytes, but input has length {len(packet)}"

    results_dict = dict(
        header = unpack_raw_header(packet[42:66]),
        data = unpack_raw_payload(packet[66:])
    )
    return results_dict

## you will need a funciton that ONLy parses the header. Thus, like this but doesn't unpack the data, 


def rsn_to_time(rsn : int, sampling_rate: float = 0.2) -> astropy.time.Time:
    """
    Convert the raw sequence number (RSN) to an astropy Time object

    Parameters
    ----------
    rsn : int
        The raw sequence number (RSN). See
        https://plm.astron.nl/polarion/#/project/LOFAR2System/workitem?id=LOFAR2-8860
    sampling_rate : float, optional
        The sampling rate, in GHz. Default is 0.2 (200 MHz sampling)

    Returns
    -------
    astropy.time.Time

    Notes
    -----
    The RSN (as an integer) encodes the time of the first sample with sub-ns
    precision; this precision is not currently guaranteed to remain in the astropy
    object.
    """

    t = astropy.time.Time(rsn / sampling_rate * 1e9, format='utc')
    return t

## this has two issues I do not like. 
## 1) this single function adds an entire dependence to astropy. But this dependency is not at all needed ( astropy is hardly needed for keeping track of time )
##               Note this also means that all downstream-code that uses your code will ALSO need astropy now just to handle the result of this one function
## 2) the precision of astropy is a problem.
## the general fix is as follows. Return instead two integers. The first integer is a unix time stamp (seconds past the epoc) and the second is number of samples past that second. 
##    Reason we do this is becouse 1) it is very general,  2) it is very easy to convert this info into whatever format you need   3) it is easy to understand



## this writer needs improvment
## 1) It is sequential when it ought to stream.
##      what I mean is that currently you assume all packets are provided, and then you write them all to disk.
##      instead assume that the packets are streamed to you. That is, they are not available all at once and are not in-order.
##      This changes the order of operations. First operation is to open the HDF5 file and hold it open for the duration of class lifespan.
##      The class keeps track of the "next" packet that is needed to write to disk 
##      then you have a function, I will name it "inject" for now. Inject takes one packet and if it is the packet you need, then you write it to disk, if not you put it on a que
##      The que should have a maximum seqential size I call "Seq_Max". 
##      If there are  Seq_Max number of sequential packets (i.e. no spaces between them) on teh que, then you assume the next packet you need was lost and you write zeros instead (and note it was lost)
##
## 2) this reader opens up the data from all packets. This will require FAR more ram than is available. Instead, ONLY open the data you need. 
##      For example, read the packet header ONLY and then store the file-pointer for time being. Do not keep read the raw data unless you need it.
##      When you need to write the data to disk, use the stored file-pointer to read the actual data array.
##      At any point it time your code should only have 1 array of packet data available.Seq_Max
##
## 3) This file is named "read". But these next two objects have nothign to do with reading (writing and plotting instead).
##      Writing and plotting ought to be their own python files
 

class HDF5Writer:

    def __init__(self, debug : bool = False):
        """
        Write transient-buffer data to HDF5

        Parameters
        ----------
        debug : bool, default False
            If True, produce some debug plots.
        """
        self._packets = dict()
        self._debug = debug
        pass

    def parse(self, packet : bytes):
        """
        Parse a UDP transient raw data packet

        """

        try:
            results_dict = parse_packet(packet)
            header = results_dict['header']
            station_info = header['station_info']
            antenna_input_index = header['antenna_input_index']
            if (station_info, antenna_input_index) not in self._packets:
                self._packets[(station_info, antenna_input_index)] = dict(
                    headers=[], data=[])
            self._packets[(station_info, antenna_input_index)]['headers'].append(header)
            self._packets[(station_info, antenna_input_index)]['data'].append(results_dict['data'])
            return 0

        except Exception as e:
            logger.error("Failed to parse UDP packet", exc_info=e)
            return 1


    def write(self, filename : str):
        """
        Write data to HDF5 output file

        Parameters
        ----------
        filename : str
            Filename to write data to

        """

        if os.path.exists(filename):
            logger.warning(f'Previous file exists at {filename}, data may be overwritten...')

        with h5py.File(filename, 'a') as f:
            n_packets = 0
            for key, d in self._packets.items():
                hdrs = d['headers']
                rsn, idx = np.unique([hdr['RSN'] for hdr in hdrs], return_index=True)

                # check for duplicates (same timestamp)
                duplicate_packets = len(hdrs) - len(rsn)
                if duplicate_packets:
                    logger.error(
                        f'Station, antenna {key} contains {duplicate_packets} packets with duplicate timestamps.'
                    )

                # check that packets come in 2000 sample intervals
                dt = np.diff(rsn)
                dt_expected = dt == 2000
                if not all(dt_expected):
                    weird_dt = dt % 2000
                    missing = np.sum((dt - 2000) // 2000)

                    if any(weird_dt):
                        logger.error(
                            f"Station, antenna {key} contains {np.sum(weird_dt.astype(bool))} timestamps"
                            f" that differ by {np.unique(weird_dt)[:25]} samples from the expectation (2000)."
                        )

                    if missing:
                        logger.error(
                            f'Station, antenna {key} is missing {missing} packets.'
                        )

                # check that all packets come with the same source_info
                # (= sampling rate, ...)
                # TODO: double check that gn_index should always be constant
                src_info, src_counts = np.unique(
                    [hdr['source_info'] for hdr in hdrs],
                    return_counts=True)

                if len(src_info) > 1:
                    logger.error(
                        f"{key} got different values for property 'source_info':"
                        f" {src_info[:25]} occurring {src_counts[:25]} times..."
                    )
                # TODO - exclude packets with 'different' source info rather than
                # pretending everything is okay?
                source_info = unpack_source_info(src_info[np.argmax(src_counts)])
                sampling_rate = sampling_rate_from_f_adc[source_info['f_adc']]
                times = (rsn - min(rsn)) / sampling_rate # time in ns since first packet

                # combine the packets, sort according to timestamp
                data = np.vstack([d['data'][i] for i in idx])

                # write the data to file - we create one group per antenna
                group = f.create_group('{}/{}'.format(*key))
                group.create_dataset('times[ns]', data=times)
                group.create_dataset('data', data=data)
                group.attrs['sampling_rate[GHz]'] = sampling_rate
                group.attrs['antenna_band_index'] = source_info['antenna_band_index']
                group.attrs['nyquist_zone_index'] = source_info['nyquist_zone_index']
                group.attrs['rsn'] = min(rsn)
                n_packets += len(times)

                if self._debug:
                    try:
                        make_debug_plots(times, data, sampling_rate, save_fig='./{}_{}.pdf'.format(*key))
                    except Exception as e:
                        logger.error(
                            'An exception occurred while trying to produce debug plots.',
                            exc_info=e)

        logger.info(f'Wrote {n_packets} packets to {filename}')


def make_debug_plots(
        times, data, sampling_rate,
        save_fig : str = './debug_output.pdf'):
    """
    Make some debug plots
    """
    blocks = data.reshape(-1, 2000, 2)
    std = np.std(blocks, axis=1)
    abs_spectra = np.abs(np.fft.rfft(blocks, axis=1))
    mean_spectra = np.mean(abs_spectra, axis=0)
    spectrum_quantiles = np.quantile(abs_spectra, [0.05, 0.16, 0.84, 0.95], axis=0)

    freqs = np.fft.rfftfreq(2000, 1/sampling_rate)

    fig, axs = plt.subplots(
        2, 3, sharex='col', layout='constrained', figsize=(12, 6),
        width_ratios=(2, 1, 2)
        )

    for i in range(2): # both polarizations
        axs[i, 0].plot(times, std[:, i], ls='', marker='.', alpha=.2)
        axs[i, 1].hist(std[:, i], bins=max(int(len(std) / 100), 10), orientation='horizontal')
        axs[i, 2].plot(freqs, mean_spectra[:,i])
        for quantile in spectrum_quantiles:
            axs[i, 2].plot(freqs, quantile[:,i], color='grey', lw=.5)

        axs[i, 0].set_ylabel('XY'[i] + ' STD [ADC]')
        axs[i, 2].set_ylabel('XY'[i] + ' Spectrum')

    axs[-1, 0].set_xlabel('Time [ns]')
    axs[-1, 1].set_xlabel('N')
    axs[-1, 2].set_xlabel('Frequency [GHz]')

    plt.savefig(save_fig)
    plt.close()
