"""
Most of the functions needed by the web service are here. 
In seekpath_app.py we just keep the main web logic.
"""
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
# from builtins import str
# from builtins import zip
import os


import copy
import numpy as np
import time, datetime

from ase.data import chemical_symbols
import json
import io

import jinja2
import spglib  # Mainly to get its version

import sys
sys.path.insert(0,'/local/git/glosim2/')
sys.path.insert(0,'/home/app/git/glosim2/')
sys.path.insert(0,'/Users/iMac/git/glosim2/')


from libmatch.soap import get_Soaps
import cPickle as pck
from ase.io import write,read
from ase import Atoms


Shift_ML_version = '0.0.1'


MAX_NUMBER_OF_ATOMS = 1000

############################
def load_model_data(sp):
    '''
    Load the model data at the start of the app
    :return: 
    '''
    dataPath = '../data/'
    # TODO dataPath = '/home/app/data/'
    # ref_desc = np.load(dataPath + 'test_soap.npy')
    with open(dataPath + 'model_sp_{}.pck'.format(z2symb[sp]), 'rb') as f:
        model = pck.load(f)
    return model

# ref_desc, model = load_model_data()

class FlaskRedirectException(Exception):
    """
    Class used to return immediately with a flash message and a redirect.
    """
    pass

def logme(logger, *args, **kwargs):
    """
    Log information on the passed logger. 

    See docstring of generate_log for more info on the
    accepted kwargs.

    :param logger: a valid logger. If you pass `None`, no log is output.
    """
    if logger is not None:
        logger.debug(generate_log(*args, **kwargs))


def generate_log(filecontent,
                 fileformat,
                 request,
                 call_source,
                 reason,
                 extra={}):
    """
    Given a string with the file content, a file format, a Flask request and 
    a string identifying the reason for logging, stores the 
    correct logs.

    :param filecontent: a string with the file content
    :param fileformat: string with the file format
    :param request: a Flask request
    :param call_source: a string identifying who called the function
    :param reason: a string identifying the reason for this log
    :param extra: additional data to add to the logged dictionary. 
        NOTE! it must be JSON-serializable
    """
    # I don't know the fileformat
    data = {'filecontent': filecontent, 'fileformat': fileformat}

    logdict = {
        'data': data,
        'reason': reason,
        'request': str(request.headers),
        'call_source': call_source,
        'source': request.headers.get('X-Forwarded-For', request.remote_addr),
        'time': datetime.datetime.now().isoformat()
    }
    logdict.update(extra)
    return json.dumps(logdict)


def get_json_for_visualizer(frame):
    # TODO get a working jmol
    response = {}
    response['numbers'] = frame.get_atomic_numbers().tolist()
    response['positions'] = frame.get_positions().tolist()
    response['cell'] = frame.get_cell().tolist()
    response['pbc'] = frame.get_pbc().tolist()

    return response


def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)

spkitMax = {8: 88, 1: 132, 6: 148, 7: 112}

z2symb = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
          8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
          14: 'Si', 15: 'P',
          16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc',
          22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co',
          28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As',
          34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',
          40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
          46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb',
          52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
          58: 'Ce', 59: 'Pr', 60: 'Nd', 62: 'Sm', 63: 'Eu', 64: 'Gd',
          65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
          71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os',
          77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
          83: 'Bi'
}

def get_Nsoap(spkitMax, nmax, lmax):
    Nsoap = 0
    for sp1 in spkitMax:
        for sp2 in spkitMax:
            if sp1 == sp2:
                Nsoap += nmax * (nmax + 1) * (lmax + 1) / 2
            elif sp1 > sp2:
                Nsoap += nmax ** 2 * (lmax + 1)
    return Nsoap + 1


def get_fingerprints(frames, soap_params, nprocess, sp=None):
    fings = get_Soaps(frames, nprocess=nprocess, **soap_params)
    if sp is not None:
        N = 0
        for cc in frames:
            numbers = cc.get_atomic_numbers()
            N += len(numbers[numbers == sp])
    else:
        N = 0
        for cc in frames:
            numbers = cc.get_atomic_numbers()
            N += len(numbers)

    Nsoap = get_Nsoap(spkitMax, soap_params['nmax'], soap_params['lmax'])
    soaps = np.zeros((N, Nsoap))
    if sp is not None:
        ii = 0
        for iframe, fing in enumerate(fings):
            numbers = fing.get_atomic_numbers()
            n = len(numbers[numbers == sp])
            centers = [fing[z2symb[sp] + str(it)] for it in range(n)]
            for icenter, center in enumerate(centers):
                soaps[ii] = center
                ii += 1
    else:
        ii = 0
        for iframe, fing in enumerate(fings):
            numbers = fing.get_atomic_numbers()
            n = len(numbers[numbers == sp])
            centers = fing.values()
            for icenter, center in enumerate(centers):
                soaps[ii] = center
                ii += 1
    return soaps

def predict(kernel, alpha, csi):
    return np.dot(alpha, np.power(kernel, csi, out=kernel)).reshape((-1))


def process_structure_core(filecontent, fileformat, call_source="", logger=None, flask_request=None):
    """
    The main function that generates the data to be sent back to the view.
    
    :param fileformat: The file format (string), among the accepted formats
    :param call_source: a string identifying the source (i.e., who called
       this function). This is a string, mainly for logging reasons.
    :param logger: if not None, should be a valid logger, that is used
       to output useful log messages.
    :param flask_request: if logger is not None, pass also the flask.request
       object to help in logging.

    :return: this function calls directly flask methods and returns flask 
        objects

    :raise: FlaskRedirectException if there is an error that requires
        to redirect the the main selection page. The Exception message
        is the message to be flashed via Flask (or in general shown to
        the user).
    """
    from structure_importers import get_structure, UnknownFormatError

    start_time = time.time()
    fileobject = io.StringIO(unicode(filecontent))
    form_data = dict(flask_request.form)

    try:
        frame = get_structure(
            fileobject, fileformat, extra_data=form_data)
    except UnknownFormatError:
        logme(
            logger,
            filecontent,
            fileformat,
            flask_request,
            call_source,
            reason='unknownformat',
            extra={
                'form_data': form_data,
            })
        raise FlaskRedirectException("Unknown format '{}'".format(fileformat))
    except Exception:
        import traceback
        logme(
            logger,
            filecontent,
            fileformat,
            flask_request,
            call_source,
            reason='codeexception',
            extra={
                'traceback': traceback.extract_stack(),
                'form_data': form_data,
            })
        raise

        # There was an exception...
        # import traceback
        # logme(
        #     logger,
        #     filecontent,
        #     fileformat,
        #     flask_request,
        #     call_source,
        #     reason='exception',
        #     extra={
        #         'traceback': traceback.format_exc(),
        #         'form_data': form_data,
        #     })
        # raise FlaskRedirectException(
        #     "I tried my best, but I wasn't able to load your "
        #     "file in format '{}'...".format(fileformat))

    if frame.get_number_of_atoms() > MAX_NUMBER_OF_ATOMS:
        ## Structure too big
        logme(
            logger,
            filecontent,
            fileformat,
            flask_request,
            call_source,
            reason='toolarge',
            extra={
                'number_of_atoms': frame.get_number_of_atoms(),
                'form_data': form_data,
            })
        raise FlaskRedirectException(
            "Sorry, this online visualizer is limited to {} atoms "
            "in the input cell, while your structure has {} atoms."
            "".format(MAX_NUMBER_OF_ATOMS, frame.get_number_of_atoms()))



    for sp in np.unique(frame.get_atomic_numbers()) :
        if sp not in [1, 6, 7, 8]:
            ## Structure has wrong species
            logme(
                logger,
                filecontent,
                fileformat,
                flask_request,
                call_source,
                reason='wrongspecies',
                extra={
                    'atomic_numbers': frame.get_atomic_numbers(),
                    'form_data': form_data,
                })
            raise FlaskRedirectException(
                "Sorry, Shift-ML is limited to the predictions of "
                "isotropic chemical shift of 1H, 13C, 15N and 17O."
                "The atomic species you provided are {}"
                "".format(np.unique(frame.get_atomic_numbers())))

    # Log the content in case of valid structure
    logme(
        logger,
        filecontent,
        fileformat,
        flask_request,
        call_source,
        reason='OK',
        extra={
            'number_of_atoms': frame.get_number_of_atoms(),
            'form_data': form_data,
        })

    try:
        in_json_data = {
            'cell': [ v.tolist() for v in frame.get_cell()],
            'scaled_coords': [ v.tolist() for v in frame.get_scaled_positions()],
            'atomic_numbers': frame.get_atomic_numbers().tolist()
        }


        inputstructure_cell_vectors = [[
            idx, coords[0], coords[1], coords[2]
        ] for idx, coords in enumerate(in_json_data['cell'], start=1)]
        inputstructure_symbols = [
            chemical_symbols[num] for num in in_json_data['atomic_numbers']
        ]
        inputstructure_positions_cartesian = np.dot(
            np.array(in_json_data['scaled_coords']),
            np.array(in_json_data['cell']),
        ).tolist()

        inputstructure_atoms_scaled = [
            [label, coords[0], coords[1], coords[2]]
            for label, coords in zip(inputstructure_symbols, in_json_data[
                'scaled_coords'])
        ]
        inputstructure_atoms_cartesian = [
            [label, coords[0], coords[1], coords[2]]
            for label, coords in zip(inputstructure_symbols,
                                     inputstructure_positions_cartesian)
        ]
##############################################################################

        shielding = np.zeros((len(frame),))
        sps = list(set(frame.get_atomic_numbers()).intersection([1,6,7,8]))
        for sp in sps:
            atoms = ase2qp(frame)
            model = load_model_data(sp)
            ll = list(sps)
            ll.remove(sp)
            soap_params = dict(
                centerweight=1.,  cutoff_transition_width=0.5, spkitMax=spkitMax,
                nocenters=ll, is_fast_average=False, chem_channels=False, dispbar=True
            )
            soap_params.update(model['soap_params'])

            descriptor = get_fingerprints([atoms], soap_params, nprocess=1, sp=sp)

            kernel_test = np.dot(model['soap_ref'], descriptor.T)

            y_pred = predict(kernel_test, model['model'].alpha, model['model'].csi)

            shielding[frame.get_atomic_numbers()==sp] = y_pred

        np.round(shielding,decimals=2,out=shielding)
        ##############################################################################

        compute_time = time.time() - start_time
    except Exception:
        import traceback
        logme(
            logger,
            filecontent,
            fileformat,
            flask_request,
            call_source,
            reason='codeexception',
            extra={
                'traceback': traceback.extract_stack(),
                'form_data': form_data,
            })
        raise

    out_json_data = get_json_for_visualizer(frame)

    raw_code_dict = copy.copy(out_json_data)

    raw_code_dict.update({'IsotropicChemicalShielding':shielding.tolist()})

    raw_code = json.dumps(raw_code_dict, indent=2)
    ## I manually escape it to then add <br> and pass it to a filter with
    ## |safe. I have to 'unicode' it otherwise it keeps escaping also the
    ## next replaces
    raw_code = str(jinja2.escape(raw_code)).replace('\n', '<br>').replace(
        ' ', '&nbsp;')

    xyz = str(jinja2.escape(get_xyz(frame, shielding))).replace(
        '\n', '<br>').replace(' ', '&nbsp;')

    magres = str(jinja2.escape(get_magres(frame, shielding))).replace(
        '\n', '<br>').replace(' ', '&nbsp;')

    return dict(
        magres=magres,
        xyz=xyz,
        jsondata=json.dumps(out_json_data),
        raw_code=raw_code,
        inputstructure_cell_vectors=inputstructure_cell_vectors,
        inputstructure_atoms_scaled=inputstructure_atoms_scaled,
        inputstructure_atoms_cartesian=inputstructure_atoms_cartesian,
        compute_time=compute_time,
        Shift_ML_version=Shift_ML_version,
    )



class DummyFileObject(object):
    def __init__(self):
        self.data = ''
    def write(self,text):
        self.data += text
    def read(self):
        return self.data

def get_xyz(frame,shielding):
    from ase.io.extxyz import write_xyz
    numbers = frame.get_atomic_numbers()
    positions = frame.get_positions()
    cell = frame.get_cell()
    pbc = frame.get_pbc()
    new_frame = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    new_frame.set_array('ChemicalShielding'.encode('ascii', 'ignore'), shielding)
    # prop_ref = frame[0].get_array('CS')[np.where(numbers == 1)[0]]

    # print new_frame.get_array('ChemicalShielding')
    # write(fn,new_frame,format='extxyz')
    f_obj = DummyFileObject()

    write_xyz(f_obj, new_frame)
    return f_obj.read()

def get_magres(frame,shielding):
    # TODO change import to ase version when ase==3.16.0
    from structure_importers.magres_io import write_magres
    numbers = frame.get_atomic_numbers()
    positions = frame.get_positions()
    cell = frame.get_cell()
    pbc = frame.get_pbc()
    new_frame = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    new_frame.set_array('iso_cs'.encode('ascii', 'ignore'), shielding)
    new_frame.info = {'ml_units': {'iso_cs': 'ppm'},
                      'magresblock_calculation': {'calc_code': [['ML_NMR']],
                                                  'calc_code_version': [['svnunknown']],
                                                  'calc_name': [['gpr']]}
                      }

    f_obj = DummyFileObject()

    write_magres(f_obj, new_frame)
    return f_obj.read()



def get_qe_pw(raw_data, out_json_data):
    """
    Return the data in format of the QE pw.x input
    """
    lines = []

    lines.append("&CONTROL")
    lines.append("    calculation = 'bands'")
    lines.append("    <...>")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append("    ibrav = 0")
    lines.append("    nat = {}".format(len(raw_data["primitive_symbols"])))
    lines.append("    ntyp = {}".format(
        len(set(raw_data["primitive_symbols"]))))
    lines.append("    <...>")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append("    <...>")
    lines.append("/")
    lines.append("ATOMIC_SPECIES")
    for s in sorted(set(raw_data["primitive_symbols"])):
        lines.append("{:4s} <MASS_HERE> <PSEUDO_HERE>.UPF".format(s))

    lines.append("ATOMIC_POSITIONS angstrom")
    for s, p in zip(raw_data["primitive_symbols"],
                    raw_data["primitive_positions_cartesian"]):
        lines.append("{:4s} {:16.10f} {:16.10f} {:16.10f}".format(
            s, p[0], p[1], p[2]))

    lines.append("K_POINTS crystal")
    kplines = []
    for kp in out_json_data['explicit_kpoints_rel']:
        kplines.append("{:16.10f} {:16.10f} {:16.10f} 1".format(*kp))
    lines.append("{}".format(len(kplines)))
    lines += kplines

    lines.append("CELL_PARAMETERS angstrom")
    for v in raw_data['primitive_lattice']:
        lines.append("{:16.10f} {:16.10f} {:16.10f}".format(v[0], v[1], v[2]))

    return "\n".join(lines)


def get_qe_matdyn(raw_data, out_json_data):
    """
    Return the data in format of the QE matdyn.x input
    """
    return "Not implemented yet, sorry..."


def get_cp2k(raw_data):
    """
    Return the data in format of a CP2K input
    """
    template = jinja2.Template("""
&GLOBAL
   RUN_TYPE ENERGY
   <...>
&END GLOBAL

&FORCE_EVAL
   <...>

   &DFT
      <...>

      &QS
         EXTRAPOLATION USE_GUESS ! required for K-Point sampling
      &END QS

      &POISSON
         PERIODIC XYZ
      &END POISSON

      &PRINT
         &BAND_STRUCTURE
            FILE_NAME <NAME_HERE>.bs

            {% for s in raw.path -%}
            &KPOINT_SET
               UNITS B_VECTOR
               SPECIAL_POINT {{ '%5s' % s[0] }} {%- for c in raw.kpoints_rel[s[0]] %} {{ '%16.10f' % c }}{% endfor %}
               SPECIAL_POINT {{ '%5s' % s[1] }} {%- for c in raw.kpoints_rel[s[1]] %} {{ '%16.10f' % c }}{% endfor %}
               NPOINTS 10
            &END KPOINT_SET
            {% endfor %}
         &END BAND_STRUCTURE
      &END PRINT

      &KPOINTS
         <...>
      &END KPOINTS
   &END DFT

   &SUBSYS
      ! Important: use the cell and coordinates provided here instead of the original ones
      &CELL
         {%- for s, v in zip(['A', 'B', 'C'], raw.primitive_lattice) %}
         {{ '%4s' % s }} [angstrom] {%- for c in v %} {{ '%16.10f' % c }}{% endfor %}
         {%- endfor %}
         PERIODIC XYZ
      &END CELL
      &COORD
         {%- for s, v in zip(raw.primitive_symbols, raw.primitive_positions_cartesian) %}
         {{ '%4s' % s }} {%- for c in v %} {{ '%16.10f' % c }}{% endfor %}
         {%- endfor %}
      &END COORD

      {% for s in set(raw.primitive_symbols) -%}
      &KIND {{ s }}
         ELEMENT {{ s }}
         BASIS_SET <BS_NAME_HERE>
         POTENTIAL <PP_NAME_HERE>
      &END KIND
      {% endfor %}
   &END SUBSYS
&END FORCE_EVAL
""")
    return template.render(
        raw=raw_data,  # export the complete raw data to the template
        zip=zip,
        set=set)  # add zip and set to the template environment for some loops
    