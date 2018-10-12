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

from structure_importers import get_structure, UnknownFormatError, WrongAtomFormat, WrongCellFormat,WrongDensity
import copy
import numpy as np
import time, datetime

from ase.data import chemical_symbols
import json
import io

import jinja2
#import spglib  # Mainly to get its version

import sys
#sys.path.insert(0,'/local/git/glosim2/')
#sys.path.insert(0,'/home/app/git/glosim2/')
#sys.path.insert(0,'/Users/iMac/git/glosim2/')
#sys.path.insert(0,'/Users/fedeparuzzo/git/glosim2/')



#from libmatch.soap import get_Soaps
import cPickle as pck
from scipy.spatial.distance import pdist

from ase import Atoms
import celery


from seekpath_app import PROGRESS_BAR_TIMEOUT,is_fast_dev,is_debug,dataPath,MAX_NUMBER_OF_ATOMS



############################



def load_all_models_data():
    '''
    Load the model data at the start of the app
    :return:
    '''
    dataPath = '../data/'
    # TODO	dataPath = '/home/app/data/'
    # ref_desc = np.load(dataPath + 'test_soap.npy')
    models = {}
    for sp in [1,6,7,8]:
        with open(dataPath + 'model_sp_{}.pck'.format(z2symb[sp]), 'rb') as f:
            models[sp] = pck.load(f)
    return models



class FlaskRedirectException(Exception):
    """
    Class used to return immediately with a flash message and a redirect.
    """
    pass

def logme( *args, **kwargs):
    """
    Log information on the passed logger.

    See docstring of generate_log for more info on the
    accepted kwargs.

    :param logger: a valid logger. If you pass `None`, no log is output.
    """
    from seekpath_app import logger
    if logger is not None:
        logger.debug(generate_log(*args, **kwargs))


def generate_log(filecontent,
                 fileformat,
                 request_info,
                 call_source,
                 reason,
                 extra={}):
    """
    Given a string with the file content, a file format, a Flask request and
    a string identifying the reason for logging, stores the
    correct logs.

    :param filecontent: a string with the file content
    :param fileformat: string with the file format
    :param request_info: a Flask request info
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
        'request': request_info['request'],
        'call_source': call_source,
        'source': request_info['source'],
        'time': datetime.datetime.now().isoformat()
    }
    logdict.update(extra)
    return json.dumps(logdict)


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


def read_test_frame(filecontent,fileformat,call_source,request_info, extra_data=""):
    fileobject = io.StringIO(unicode(filecontent))
    form_data = request_info['form']
    try:
        frame = get_structure(
            fileobject, fileformat, extra_data=extra_data)
    except UnknownFormatError:
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='unknownformat',
            extra={
                'form_data': form_data,
            })
        raise FlaskRedirectException("Unknown format '{}'".format(fileformat))

    except WrongAtomFormat:
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='unknownformat',
            extra={
                'form_data': form_data,
            })
        raise FlaskRedirectException("Problem with the parsing of the atoms")

    except WrongCellFormat:
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='unknownformat',
            extra={
                'form_data': form_data,
            })
        raise FlaskRedirectException("Problem with the parsing of the cell")

    except WrongDensity:
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='wrongdensity',
            extra={
                'form_data': form_data,
            })
        raise FlaskRedirectException("The density of the structure seems wrong")

    except Exception:


        import traceback
        print traceback.format_exc()
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='exception',
            extra={
                'traceback': traceback.format_exc(),
                'form_data': form_data,
            })
        raise FlaskRedirectException(
            "I tried my best, but I wasn't able to load your "
            "file in format '{}'...".format(fileformat))

    test_frame(frame, filecontent, fileformat, call_source, request_info, extra_data)


def test_frame(frame,filecontent, fileformat,call_source,request_info, extra_data=""):

    form_data = request_info['form']
    if frame.get_number_of_atoms() > MAX_NUMBER_OF_ATOMS:
        ## Structure too big
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='toolarge',
            extra={
                'number_of_atoms': frame.get_number_of_atoms(),
                'form_data': form_data,
            })
        raise FlaskRedirectException(
            "Sorry, this online app is limited to {} atoms "
            "in the input cell, while your structure has {} atoms."
            "".format(MAX_NUMBER_OF_ATOMS, frame.get_number_of_atoms()))

    for sp in np.unique(frame.get_atomic_numbers()) :
        if sp not in [1, 6, 7, 8, 16, 17]:
            ## Structure has wrong species
            logme(
                filecontent,
                fileformat,
                request_info,
                call_source,
                reason='wrongspecies',
                extra={
                    'atomic_numbers': frame.get_atomic_numbers().tolist(),
                    'form_data': form_data,
                })
            raise FlaskRedirectException(
                "Sorry, ALPHA-ML is limited to the predictions of "
                " polarizabilities of molecules containing H, C, N, O, S, Cl."
                "The atomic species you provided are {}"
                "".format(np.unique( [ str(z2symb[sp]) for sp in np.unique(frame.get_atomic_numbers().tolist())])))

    positions = frame.get_positions()
    dist = pdist(positions)

    if dist[dist < 0.5].size > 0:
        ## overlapping atoms
        logme(
            filecontent,
            fileformat,
            request_info,
            call_source,
            reason='toolarge',
            extra={
                'form_data': form_data,
            })
        raise FlaskRedirectException(
            "There are some overlapping atoms. Please check your input file."
        )


    # Log the content in case of valid structure
    logme(
        filecontent,
        fileformat,
        request_info,
        call_source,
        reason='OK',
        extra={
            'number_of_atoms': frame.get_number_of_atoms(),
            'form_data': form_data,
        })


class DummyFileObject(object):
    def __init__(self):
        self.data = ''
    def write(self,text):
        self.data += text
    def read(self):
        return self.data

def get_xyz(frame,shielding, slopes, intercepts):
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

    #add disclaimer as comment in 2nd line
    disclaimer_str = ''
    if slopes!=None:
        tmp_slopes=json.loads(slopes)
        disclaimer_str += ' Slope="1H '+tmp_slopes['1']+', 13C '+tmp_slopes['6']+ ', 15N '+tmp_slopes['7']+ ', 17O '+tmp_slopes['8']+'"'
    if intercepts!=None:
        tmp_intercepts=json.loads(intercepts)
        disclaimer_str += ' Intercept="1H '+tmp_intercepts['1']+', 13C '+tmp_intercepts['6']+ ', 15N '+tmp_intercepts['7']+ ', 17O '+tmp_intercepts['8']+'"'
    disclaimer_str += ' Comments="The isotropic chemical shieldings/shifts were generated by ShiftML v' + Shift_ML_version + ' (https://shiftml.epfl.ch).'
    disclaimer_str += ' They are subject to the terms of use (https://shiftml.epfl.ch).'
    disclaimer_str += ' Please read and cite the following:'
    disclaimer_str += ' Federico M. Paruzzo, Albert Hofstetter, Felix Musil, De Sandip, Michele Ceriotti, and Lyndon Emsley. Chemical Shifts in Molecular Solids by Machine Learning (submitted)."'
    xyz_file = f_obj.read()
    xyz_idx = xyz_file.find('\n')
    xyz_idx += xyz_file[xyz_idx+1:-1].find('\n')
    xyz_file = xyz_file[0:xyz_idx+1]+disclaimer_str+xyz_file[xyz_idx+1:-1]

    return xyz_file

def get_magres(frame,shielding, slopes, intercepts):
    # TODO change import to ase version when ase==3.16.0
    from structure_importers.magres_io import write_magres
    numbers = frame.get_atomic_numbers()
    positions = frame.get_positions()
    cell = frame.get_cell()
    pbc = frame.get_pbc()
    shielding_tensor =np.array([np.eye(3)*ss for ss in shielding])
    new_frame = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    new_frame.set_array('ms'.encode('ascii', 'ignore'), shielding_tensor)
    new_frame.info = {'magres_units': {'ms': 'ppm'},
                      'magresblock_calculation': {'calc_code': [['ML_NMR']],
                                                  'calc_code_version': [['svnunknown']],
                                                  'calc_name': [['gpr']]}
                      }

    f_obj = DummyFileObject()
    write_magres(f_obj, new_frame, Shift_ML_version, slopes, intercepts)
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
