import os.path
import csv
from ast import literal_eval
from collections import namedtuple

from logging import getLogger
_log = getLogger(__name__)


def read_spatiocyte(pathto, tstart, tend, interval, max_count=None):
    (count_array, index_array_size, index0) = spatiocyte_time_arrays(tstart, tend, interval)
    data = read_spatiocyte_data(pathto, count_array, max_count=max_count)

    SpatiocyteDataSet = namedtuple('SpatiocyteDataSet', ('data', 'index_array_size', 'index0'))
    return SpatiocyteDataSet(data, index_array_size=index_array_size, index0=index0)

def spatiocyte_time_arrays(start_time, end_time, interval):
    # set count arrays by spatiocyte interval
    interval = self.spatiocyte_interval
    N_count = int(round((end_time - start_time)/interval))
    c0 = int(round(start_time/interval))

    delta_array = numpy.zeros(shape=(N_count))
    delta_array.fill(interval)
    time_array  = numpy.cumsum(delta_array) + start_time
    count_array = numpy.array([c + c0 for c in range(N_count)])

    # set index arrays by exposure time
    exposure = self.detector_exposure_time
    N_index = int(round((end_time - start_time)/exposure))
    i0 = int(round(start_time/exposure))
    index_array = numpy.array([i + i0 for i in range(N_index)])

    return (count_array, len(index_array), index_array[0])

def read_spatiocyte_shape(filename):
    cell_shape = numpy.genfromtxt(filename, delimiter=',')
    cell_shape = numpy.array(cell_shape.tolist())  #XXX: == cell_shape?
    return cell_shape

def read_spatiocyte_data(pathto, shutter_count_array=None, max_count=None):
    # set data-array
    data = []

    # read lattice file
    for i in range(len(shutter_count_array)):
        csv_file_path = os.path.join(pathto, 'pt-{:09d}.0.csv'.format(shutter_count_array[i]))
        if not os.path.isfile(csv_file_path):
            _log.err('{} not found'.format(csv_file_path))
            #XXX: raise an axception

        with open(csv_file_path, 'r') as csv_file:
            particles = []
            t = None
            for row in csv.reader(csv_file):
                t_ = float(row[0])
                coordinate = (float(row[1]), float(row[2]), float(row[3]))
                # radius = float(row[4])
                id1 = literal_eval(row[5])
                assert isinstance(id1, tuple) and len(id1) == 2
                id2 = literal_eval(row[6])
                assert isinstance(id2, tuple) and len(id2) == 2

                if len(row) >= 9:
                    p_state, cyc_id = float(row[7]), float(row[8])
                else:
                    p_state, cyc_id = 1.0, float('inf')

                particles.append((coordinate, id1[0], id1[1], id2[1], p_state, cyc_id))
                if t is None:
                    t = t_
                elif t != t_:
                    raise RuntimeError('File [{}] contains multiple time'.format(csv_file_path))

            # Just for debugging
            if max_count is not None and len(particles) > max_count:
                particles = particles[: max_count]

            _log.debug('File [{}] was loaded. [t={}, #particles={}]'.format(csv_file_path, t, len(particles)))
            data.append([t, particles])

    data.sort(key=lambda x: x[0])
    return data
