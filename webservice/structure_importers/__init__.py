__init__ = ['xyz_io','magres_io']

import ase.io
from ase.data import atomic_numbers
import pymatgen
import qe_tools
import numpy as np
from ase import Atoms

class UnknownFormatError(ValueError):
    pass


def get_atomic_numbers(symbols):
    """
    Given a list of symbols, return the corresponding atomic numbers.

    :raise ValueError: if the symbol is not recognized
    """
    retlist = []
    for s in symbols:
        try:
            retlist.append(atomic_numbers[s])
        except KeyError:
            raise ValueError("Unknown symbol '{}'".format(s))
    return retlist


def tuple_from_ase(asestructure):
    """
    Given a ASE structure, return a structure tuple as expected from seekpath

    :param asestructure: a ASE Atoms object
    
    :return: a structure tuple (cell, positions, numbers) as accepted
        by seekpath.
    """
    atomic_numbers = get_atomic_numbers(asestructure.get_chemical_symbols())
    structure_tuple = (asestructure.cell.tolist(),
                       asestructure.get_scaled_positions().tolist(),
                       atomic_numbers)
    return structure_tuple





def get_structure(fileobject, fileformat, extra_data=None):
    """
    Given a file-like object (using StringIO or open()), and a string
    identifying the file format, return a structure tuple as accepted
    by seekpath.

    :param fileobject: a file-like object containing the file content
    :param fileformat: a string with the format to use to parse the data

    :return: a structure tuple (cell, positions, numbers) as accepted
        by seekpath.
    """
    ase_fileformats = {
        'vasp-ase': 'vasp',
        'xsf-ase': 'xsf',
        'castep-ase': 'castep-cell',
        'pdb-ase': 'proteindatabank',
        'xyz-ase': 'extxyz',
        'cif-ase':
        'cif',  # currently broken in ASE: https://gitlab.com/ase/ase/issues/15
    }
    if fileformat in ['xyz-ase']:
        from xyz_io import read_xyz
        # problem with the encoding for np.dtypes (it does not accept unicode but that's how it has to be)
        for cc in read_xyz(fileobject):
            asestructure = cc
            break # reads only the first if there are more
        is_proper_crystal(asestructure)
        return asestructure
    elif fileformat in ase_fileformats.keys():
        from ase.io import read
        asestructure = read(fileobject,index='0',format=ase_fileformats[fileformat])
        is_proper_crystal(asestructure)
        return asestructure

    elif fileformat == "cif-pymatgen":
        from pymatgen.io.cif import CifParser
        from pymatgen.io.ase import AseAtomsAdaptor
        # Only get the first structure, if more than one
        pmgstructure = CifParser(fileobject).get_structures()[0]
        asestructure = AseAtomsAdaptor.get_atoms(pmgstructure)
        is_proper_crystal(asestructure)
        return asestructure


    raise UnknownFormatError(fileformat)


class WrongCellFormat(ValueError):
    pass

class WrongAtomFormat(ValueError):
    pass

def is_proper_crystal(frame):
    cell = frame.get_cell()
    sps = frame.get_atomic_numbers()
    Natoms = frame.get_number_of_atoms()

    if Natoms == 0 or len(sps) == 0:
        raise WrongAtomFormat
    elif np.isclose(cell,np.zeros((3,3))) is False:
        raise WrongCellFormat
    else:
        return None