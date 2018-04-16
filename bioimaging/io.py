import os.path
import warnings
import re
import csv
import copy
import glob
from ast import literal_eval
from collections import namedtuple, defaultdict

import numpy

from logging import getLogger
_log = getLogger(__name__)


def is_float(val):
    """Check if the given string can be converted to float or not.
    This method is fast when returning True

    Args:
        val (str): The input

    Returns:
        bool: Return True if the input can be converted to float

    """
    try:
        float(val)
        return True
    except ValueError:
        return False

def read_spatiocyte(tstart, tend, pathto, observable=None, max_count=None):
    spatiocyte_input = read_spatiocyte_input(os.path.join(pathto, 'pt-input.csv'))
    species_ids = spatiocyte_input['species_ids']
    lengths = spatiocyte_input['lengths']
    if isinstance(observable, str):
        observables = [sp_idx for sp_idx, (sp_id, radius) in species_ids.items()
                       if re.search(observable, sp_id) is not None]
    else:
        observables = None

    filenames = glob.glob(os.path.join(pathto, 'pt-*.0.csv'))
    data = read_inputs(filenames, tstart, tend, observables, max_count)

    SpatiocyteDataSet = namedtuple('SpatiocyteDataSet', ('data', 'lengths'))
    return SpatiocyteDataSet(data, lengths=lengths)

def read_spatiocyte_input(filename):
    with open(filename, 'r') as f:
        header = f.readline().rstrip().split(',')
    assert len(header) >= 5

    interval, lz, ly, lx, voxel_radius = [float(_) for _ in header[: 5]]
    species_info = header[5:  ]
    lengths = numpy.array([lx, ly, lz], dtype=numpy.float64) * 2 * voxel_radius / 1e-9  # nanometers

    species_ids = {}
    for sp in species_info:
        # ex) [/Cell/Membrane:S1][0]=2e-08
        mobj = re.fullmatch(r'\[(.*)\]\[(\d+)\]=(.*)', sp)
        if mobj is None:
            warnings.warn('An invalid species info [{}] was given. Check [{}]'.format(sp, filename))
            continue
        sp_id = mobj.group(1)
        sp_idx = int(mobj.group(2))
        radius = float(mobj.group(3))
        if re.fullmatch(r'(\/|(\/[^\/]+)+):([^\/:]+)', sp_id) is None:
            warnings.warn('A species ID is invalid [{}]. Ignored'.format(sp))
            continue
        _log.info("    Species [{}] => ID={}, Radius={}".format(sp_idx, sp_id, radius))
        species_ids[sp_idx] = (sp_id, radius)

    _log.info('    Time Interval = {} sec'.format(interval))
    _log.info('    Voxel radius  = {} m'.format(voxel_radius))
    _log.info('    Compartment lengths: {} nm x {} nm x {} nm'.format(lengths[0], lengths[1], lengths[2]))
    _log.info('    Species IDs: {}'.format(', '.join([sp_id for sp_id, _ in species_ids.values()])))

    return dict(interval=interval, lengths=lengths, voxel_radius=voxel_radius, species_ids=species_ids)

# def read_spatiocyte_shape(filename):
#     cell_shape = numpy.genfromtxt(filename, delimiter=',')
#     cell_shape = numpy.array(cell_shape.tolist())  #XXX: == cell_shape?
#     return cell_shape

def read_time(filename):
    with open(filename, 'r') as fin:
        for row in csv.reader(fin):
            if len(row) == 0 or not is_float(row[0]):
                return None
            else:
                return float(row[0])

def read_inputs(filenames, tstart, tend, observables=None, max_count=None):
    times = []
    for filename in filenames:
        if not os.path.isfile(filename):
            raise RuntimeError('File [{}] was not found'.format(filename))

        t = read_time(filename)
        if t is None:
            continue
        times.append((t, filename))

    time_array = []
    filenames = []
    for t, filename in sorted(times, key=lambda x: x[0]):
        time_array.append(t)
        filenames.append(filename)
    time_array = numpy.asarray(time_array)

    start_idx = numpy.searchsorted(time_array, tstart, side='left')
    end_idx = numpy.searchsorted(time_array, tend, side='left')
    time_array = time_array[start_idx: end_idx]
    filenames = filenames[start_idx: end_idx]

    # set data-array
    data = []

    for t, filename in zip(time_array, filenames):
        with open(filename, 'r') as csv_file:
            particles = []
            for row in csv.reader(csv_file):
                t_ = float(row[0])
                coordinate = (float(row[1]), float(row[2]), float(row[3]))
                # radius = float(row[4])
                # Molecule ID and its state
                id1 = literal_eval(row[5])
                assert isinstance(id1, tuple) and len(id1) == 2
                # Fluorophore ID and compartment ID
                id2 = literal_eval(row[6])
                assert isinstance(id2, tuple) and len(id2) == 2

                if len(row) >= 9:
                    p_state, cyc_id = float(row[7]), float(row[8])
                else:
                    p_state, cyc_id = 1.0, float('inf')

                if observables is None or int(id1[1]) in observables:
                    particles.append((coordinate, id1[0], id1[1], id2[1], p_state, cyc_id))

                if t != t_:
                    raise RuntimeError('File [{}] contains multiple time'.format(csv_file_path))

        # Just for debugging
        if max_count is not None and len(particles) > max_count:
            particles = particles[: max_count]

        _log.debug('File [{}] was loaded. [t={}, #particles={}]'.format(filename, t, len(particles)))
        data.append([t, particles])

    return data

def read_catalog(filename):
    if not os.path.exists(filename):
        raise IOError('Catalog file [{}] was not found'.format(filename))

    catalog_data = defaultdict(list)
    with open(filename, 'r') as fin:
        for _ in range(5):
            line = fin.readline()
            line = line.rstrip()
            _log.debug('     {}'.format(line))

        reader = csv.reader(fin)

        row = next(reader)
        if len(row) != 1 or is_float(row[0]):
            raise RuntimeError('A catalog in invalid format was given [{}]'.format(filename))
        key = row[0]

        for row in reader:
            if len(row) == 1 and not is_float(row[0]):
                key = row[0]
            elif len(row) != 0:
                catalog_data[key].append(row)
    return catalog_data

def read_fluorophore_catalog(fluorophore_type):
    filename = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'catalog/fluorophore/', fluorophore_type + '.csv')
    catalog_data = read_catalog(filename)
    return catalog_data['Excitation'], catalog_data['Emission']

def read_dichroic_catalog(dm):
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/dichroic/', dm + '.csv')
    return read_catalog(filename)['wavedataset']

def read_excitation_catalog(excitation):
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/excitation/', excitation + '.csv')
    return read_catalog(filename)['wavedataset']

def read_emission_catalog(emission):
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/emission/', emission + '.csv')
    return read_catalog(filename)['wavedataset']

