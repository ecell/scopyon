import os.path
import csv
from ast import literal_eval

from logging import getLogger
_log = getLogger(__name__)


def read_spatiocyte_shape(self, filename):
    cell_shape = numpy.genfromtxt(filename, delimiter=',')
    cell_shape = numpy.array(cell_shape.tolist())  #XXX: == cell_shape?
    return cell_shape

def read_spatiocyte_data(csv_file_directory, shutter_count_array=None, max_count=None):
    # set data-array
    data = []

    # read lattice file
    for i in range(len(shutter_count_array)):
        csv_file_path = os.path.join(csv_file_directory, 'pt-{:09d}.0.csv'.format(shutter_count_array[i]))
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

