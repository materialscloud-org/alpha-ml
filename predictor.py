
import numpy as np
import quippy as qp
import sys
import cPickle as pck


sys.path.insert(0,'/home/app/git/glosim2/')

from libmatch.soap import get_Soaps

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

def test():

    dataPath = '../data/'
    frames = qp.AtomsList(dataPath+'test.xyz')

    soap_params = dict(
        centerweight=1., gaussian_width=0.3, cutoff=4, cutoff_transition_width=0.5, spkitMax=spkitMax,
        nmax=9, lmax=9, nocenters=[8, 6, 7], is_fast_average=False, chem_channels=False, dispbar=True
                       )
    descriptor = get_fingerprints(frames, soap_params, nprocess=1, sp=1)
    ref_desc = np.load(dataPath+'test_soap.npy')
    kernel_test = np.dot(ref_desc, descriptor.T)
    with open(dataPath+'test_model.pck', 'rb') as f:
        model = pck.load(f)

    numbers = frames[0].get_atomic_numbers()
    prop_ref = frames[0].get_array('CS')[np.where(numbers == 1)[0]]

    print predict(kernel_test, **model) - prop_ref

if __name__ == '__main__':
    test()