import ase
import numpy as np
from scipy import special
from sympy.physics.wigner import wigner_3j

#############################################################################################

def complex_to_real_transformation(lval):
    """Transformation matrix from complex to real spherical harmonics"""

    if lval==0:
        mat = 1.0

    if lval==1:
        mat = np.array([[1.0j,0.0,1.0j],
                       [0.0,np.sqrt(2.0),0.0],
                       [1.0,0.0,-1.0]
                       ], dtype=complex) / np.sqrt(2.0)

    if lval==2:
        mat = np.array([[1.0j,0.0,0.0,0.0,-1.0j],
                        [0.0,1.0j,0.0,1.0j,0.0],
                        [0.0,0.0,np.sqrt(2.0),0.0,0.0],
                        [0.0,1.0,0.0,-1.0,0.0],
                        [1.0,0.0,0.0,0.0,1.0]      ],dtype=complex) / np.sqrt(2.0)

    if lval==3:
        mat =np.array([[1.0j,0.0,0.0,0.0,0.0,0.0,1.0j],
                       [0.0,1.0j,0.0,0.0,0.0,-1.0j,0.0],
                       [0.0,0.0,1.0j,0.0,1.0j,0.0,0.0],
                       [0.0,0.0,0.0,np.sqrt(2.0),0.0,0.0,0.0],
                       [0.0,0.0,1.0,0.0,-1.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0,0.0,1.0,0.0],
                       [1.0,0.0,0.0,0.0,0.0,0.0,-1.0]   ], dtype=complex) / np.sqrt(2.0)

    if lval==4:
        mat =np.array([[1.0j,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0j],
                       [0.0,1.0j,0.0,0.0,0.0,0.0,0.0,1.0j,0.0],
                       [0.0,0.0,1.0j,0.0,0.0,0.0,-1.0j,0.0,0.0],
                       [0.0,0.0,0.0,1.0j,0.0,1.0j,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,np.sqrt(2.0),0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,1.0,0.0,-1.0,0.0,0.0,0.0],
                       [0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0],
                       [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]], dtype=complex) / np.sqrt(2.0)

    return mat

#############################################################################################

def setup_orthomatrix(nmax,rc,rk):

    sigma = np.zeros(nmax,float)
    dsigma = 0.2*rc/float(nmax)
    sigma[0] = 0.2*rc/float(nmax)
    for i in xrange(1,nmax):
        sigma[i] = sigma[i-1] + dsigma

    overlap = np.zeros((nmax,nmax),float)
    for n1 in xrange(nmax):
        for n2 in xrange(nmax):
            overlap[n1,n2] = (0.5/(sigma[n1])**2 + 0.5/(sigma[n2])**2)**(-0.5*(3.0 +n1 +n2)) \
                             /(sigma[n1]**n1 * sigma[n2]**n2)*\
                              special.gamma(0.5*(3.0 + n1 + n2))/ (  \
                    (sigma[n1]*sigma[n2])**1.5 * np.sqrt(special.gamma(1.5+n1)*special.gamma(1.5+n2)) )    

    eigenvalues, unitary = np.linalg.eig(overlap)
    sqrteigen = np.sqrt(eigenvalues) 
    diagoverlap = np.diag(sqrteigen)
    newoverlap = np.dot(np.conj(unitary),np.dot(diagoverlap,unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)

    return [orthomatrix,sigma]

##################################################################################################################

def get_power_spectrum(lam,frame):


    ### THIS PART CAN BE CLEANED UP SIGNIFICANTLY!

    atom_valence = {"H": 1,"He": 2,"Li": 3,"Be": 4,"B": 5,"C": 6,"N": 7,"O": 8,"F": 9,"Ne": 10,"Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15,"S": 16,"Cl": 17,"Ar": 18,"K": 19,"Ca": 20,"Sc": 21,"Ti": 22,"V": 23,"Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,"Cu": 29,"Zn": 30}
    atom_symbols = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'}
    nsmax = 26

    cen = [atom_symbols[num] for num in list(set([atom_valence[centre] for centre in ["C","Cl","H","N","O","S"]]))]
    all_names = frame.get_chemical_symbols()
    unique_names = list(set([name for name in cen]))
    nspecies = len(unique_names)
    all_species = []
    for k in unique_names:
        all_species.append(atom_valence[k])
    coords = frame.get_positions()

    atom_indexes = [[] for j in xrange(nsmax)]
    natmax = len(all_names)
    for ii in xrange(natmax):
        for j in all_species:
            if all_names[ii] == atom_symbols[j]:
                atom_indexes[j].append(ii)

    centers = centers = [atom_valence[i] for i in cen]

    ###

    # Read in sparsification options from external files
    sparse_options=[np.load('/home/app/data/SPARSE_' + typestr + str(lam) + '.npy') for typestr in ['Amat','fps']]
#    sparse_options = ['','']

    params = dict( lmax=6, nmax=8, sg=0.3, cw=1.0,rk=1e-10,rc=4.0, nnmax=35 )

    lmax = params['lmax']
    nmax = params['nmax']
    sg = params['sg']
    cw = params['cw']
    rk = params['rk']
    rc = params['rc']
    nnmax = params['nnmax']

    llmax = 0
    lvalues = {}
    if lam > 0: # Needed for tensorial stuff
        for l1 in xrange(lmax+1):
            for l2 in xrange(lmax+1):
                if abs(l2-lam) <= l1 and l1 <= (l2+lam):
                    lvalues[llmax] = [l1,l2]
                    llmax += 1

    [orthomatrix,sigma] = setup_orthomatrix(nmax,rc,rk)

    if lam == 0:

        featsize = nspecies*nspecies*nmax**2*(lmax+1)
        do_list = np.array(sparse_options[1],dtype=int)
        PS = np.zeros((1,natmax,len(do_list)),dtype=complex)

        [nat,omega,harmonic,orthoradint] = fill_arrays(nspecies,nnmax,lmax,nmax,centers,atom_indexes,all_species,coords,rc,rk,cw,sigma,sg,orthomatrix) 
    
        # PRECOMPUTE OMEGA CONJUGATE
        omegatrue = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
        omegaconj = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
        for l in xrange(lmax+1):
            for im in xrange(2*l+1):
                omegatrue[:,:,:,l,im] = omega[:,:,:,l,im]/np.sqrt(np.sqrt(2*l+1))
                omegaconj[:,:,:,l,im] = np.conj(omega[:,:,:,l,im])/np.sqrt(np.sqrt(2*l+1))
    
        # COMPUTE POWER SPECTRUM
        power = np.einsum('asblm,adnlm->asdbnl',omegatrue,omegaconj)
        power = power.reshape(nat,featsize)
        for iat in xrange(nat):
            inner = np.real(np.dot(power[iat],np.conj(power[iat])))
            power[iat] /= np.sqrt(inner)

        PS[0,:nat] = power[:,do_list]
    
    else:
      
        # PRECOMPUTE WIGNER-3J SYMBOLS
        w3j = np.zeros((2*lam+1,lmax+1,lmax+1,2*lmax+1),float)
        for l1 in xrange(lmax+1):
            for l2 in xrange(lmax+1):
                for m in xrange(2*l1+1):
                    for mu in xrange(2*lam+1):
                        w3j[mu,l1,l2,m] = wigner_3j(lam,l2,l1,mu-lam,m-l1-mu+lam,-m+l1) * (-1.0)**(m-l1)

        featsize = nspecies*nspecies*nmax**2*llmax
        do_list = np.array(sparse_options[1],dtype=int)
        PS = np.zeros((1,natmax,2*lam+1,len(do_list)),dtype=complex)

        # COMPUTE HARMONIC CONJUGATE, omega conjugate and tensorial power spectrum
        # reduce l dimensionality keeping the nonzero elements
        [nat,omega,harmonic,orthoradint] = fill_arrays(nspecies,nnmax,lmax,nmax,centers,atom_indexes,all_species,coords,rc,rk,cw,sigma,sg,orthomatrix) 
        power = np.zeros((nat,2*lam+1,nspecies,nspecies,nmax,nmax,lmax+1,lmax+1),complex)
        p2 = np.zeros((nat,2*lam+1,nspecies,nspecies,nmax,nmax,llmax),complex)
        omegaconj = np.zeros((nat,nspecies,2*lam+1,nmax,lmax+1,lmax+1,2*lmax+1),complex)
        harmconj = np.zeros((nat,nspecies,lmax+1,lmax+1,2*lmax+1,2*lam+1,nnmax),dtype=complex)

        for lval in xrange(lmax+1):
            for im in xrange(2*lval+1):
                for lval2 in xrange(lmax+1):
                    for mu in xrange(2*lam+1):
                        if abs(im-lval-mu+lam) <= lval2:
                            harmconj[:,:,lval2,lval,im,mu,:] = np.conj(harmonic[:,:,lval2,lval2+im-lval-mu+lam,:])
        for iat in xrange(nat):
            for ispe in xrange(nspecies): 
                omegaconj[iat,ispe] = np.einsum('lnh,lkmvh->vnklm',orthoradint[iat,ispe],harmconj[iat,ispe])
        for iat in xrange(nat):
            for ia in xrange(nspecies):
                for ib in xrange(nspecies):
                    power[iat,:,ia,ib] = np.einsum('nlv,xmlkv,xlkv->xnmlk',omega[iat,ia],omegaconj[iat,ib],w3j)

        for l in xrange(llmax):
            p2[:,:,:,:,:,:,l] = power[:,:,:,:,:,:,lvalues[l][0],lvalues[l][1]]
        p2 = p2.reshape(nat,2*lam+1,featsize)
        # Normalize power spectrum
        for iat in xrange(nat):
            inner = np.zeros((2*lam+1,2*lam+1),complex)
            for mu in xrange(2*lam+1):
                for nu in xrange(2*lam+1):
                    inner[mu,nu] = np.dot(p2[iat,mu],np.conj(p2[iat,nu]))
            p2[iat] /= np.sqrt(np.real(np.linalg.norm(inner)))

        PS[0,:nat] = p2[:,:,do_list]

    # Multiply by A_matrix.
    PS = np.dot(PS,sparse_options[0])

    return [PS,featsize]

##################################################################################################################

def fill_arrays(nspecies,nnmax,lmax,nmax,centers,atom_indexes,all_species,coords,rc,rk,cw,sigma,sg,orthomatrix):

    alpha = 1.0/(2*sg**2)
    sg2 = sg**2

    nat = 0
    for k in centers:
        nat += len(atom_indexes[k])

    length = np.zeros((nat,nspecies,nnmax), dtype=float )
    efact = np.zeros((nat,nspecies,nnmax), dtype=float )
    nneigh = np.zeros((nat,nspecies), dtype=int )

    omega = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
    harmonic = np.zeros((nat,nspecies,lmax+1,2*lmax+1,nnmax), dtype=complex )
    orthoradint = np.zeros((nat,nspecies,lmax+1,nmax,nnmax),float)
    radint = np.zeros((nat,nspecies,nnmax,lmax+1,nmax),float)
    nat    = 0

    def switch(r):
       if r <= rc-rk:
           f = 1.0
       elif rc-rk < r < rc:
           f = (1.0 + np.cos(np.pi*(r-(rc-rk))/rk))/2.0
       elif r >= rc:
           f = 0.0
       return f

    iat = 0
    # Loop over species to center on
    for k in centers:
        nat += len(atom_indexes[k])
        # Loop over centers of that species
        for l in atom_indexes[k]:
            # Loop over all the spcecies to use as neighbours 
            ispe = 0
            for ix in all_species:
                n=0
                # Loop over neighbours of that species
                for m in atom_indexes[ix]:
                    rrm = coords[m] 
                    rr  = rrm - coords[l]
                    # Is it a neighbour within the spherical cutoff ?
                    if np.dot(rr,rr) <= rc**2: 
                        # Central atom?
                        if m == l:
                            length[iat,ispe,n]  = 0.0
                            efact[iat,ispe,n]   = cw
                            harmonic[iat,ispe,0,0,n] = special.sph_harm(0,0,0,0)
                            nneigh[iat,ispe] += 1
                            n += 1
                        else:
                            length[iat,ispe,n]  = np.linalg.norm(rr)
                            theta   = np.arccos(rr[2]/length[iat,ispe,n])
                            phi     = np.arctan2(rr[1],rr[0])
                            efact[iat,ispe,n]   = np.exp(-alpha*length[iat,ispe,n]**2)
                            efact[iat,ispe,n]  *= switch(length[iat,ispe,n])
                            for lval in xrange(lmax+1): 
                                mxrange = np.arange(-lval,lval+1)
                                harm = np.conj(special.sph_harm(mxrange,lval,phi,theta))
                                for im in xrange(2*lval+1):
                                    harmonic[iat,ispe,lval,im,n] = harm[im]
                            nneigh[iat,ispe] += 1
                            n += 1
                ispe += 1
            iat +=1

    for n in xrange(nmax):        
        normfact = np.sqrt(2.0/(special.gamma(1.5+n)*sigma[n]**(3.0+2.0*n)))
        sigmafact = (sg2**2+sg2*sigma[n]**2)/sigma[n]**2
        for l in xrange(lmax+1):
            radint[:,:,:,l,n] = efact[:,:,:] \
                                * 2.0**(-0.5*(1.0+l-n)) \
                                * (1.0/sg2 + 1.0/sigma[n]**2)**(-0.5*(3.0+l+n)) \
                                * special.gamma(0.5*(3.0+l+n))/special.gamma(1.5+l) \
                                * (length[:,:,:]/sg2)**l \
                                * special.hyp1f1(0.5*(3.0+l+n), 1.5+l, 0.5*length[:,:,:]**2/sigmafact)   
        radint[:,:,:,:,n] *= normfact

    for iat in xrange(nat):
        for ispe in xrange(nspecies):
            for neigh in xrange(nneigh[iat,ispe]): 
                for l in xrange(lmax+1):
                    orthoradint[iat,ispe,l,:,neigh] = np.dot(orthomatrix,radint[iat,ispe,neigh,l])

    for iat in xrange(nat):
        for ispe in xrange(nspecies):
            omega[iat,ispe] = np.einsum('lnh,lmh->nlm',orthoradint[iat,ispe],harmonic[iat,ispe])

    return [nat,omega,harmonic,orthoradint]

##################################################################################################################

def split_power_spectra(PS,frame):

    natom = len(frame.get_chemical_symbols())
    ps0_new = np.zeros((natom,1,len(PS[0][0,0])),dtype=complex)
    ps2_new = np.zeros((natom,1,5,len(PS[1][0,0,0])),dtype=complex)
    k = 0
    for i in xrange(natom):
        ps0_new[k,0,:] = PS[0][0,k,:]
        ps2_new[k,0,:,:] = PS[1][0,k,:,:]
        k += 1

    return [ps0_new,ps2_new]

#############################################################################################

def get_kernel(lam,PS,frame,PS0):

    # Get training and testing number of atoms
    nat1 = np.load('/home/app/data/natoms.npy')
    all_names = frame.get_chemical_symbols()
    nat2 = len(all_names)

    # Read in the appropriate training-set power spectrum

    if (lam==0):
        # Compute nonlinear scalar kernel of order 2
        # Firstly, read in training power spectrum
        pstrain = np.load('/home/app/data/PStrain_lam0.npy')
        ntrain = len(pstrain)
        kreal = np.zeros((ntrain,nat2),dtype=float)
        # Now combine with testing power spectrum
        for i in xrange(ntrain):
            for j in xrange(nat2):
                for iat in xrange(nat1[i]):
#                for jat in xrange(nat2):
                    kreal[i,j] += np.real(np.dot(pstrain[i,iat],np.conj(PS[j,0])))**2
                kreal[i,j] /= (nat1[i]*1)

    else:
        # Compute nonlinear L=2 kernel of order 2
        # Firstly, read in training power spectra
#        pstrain  = np.load('/home/app/data/PStrain_lam2.npy')
        pstrain0 = np.load('/home/app/data/PStrain_lam0.npy')
        ntrain = len(pstrain0)
        kreal = np.zeros((ntrain,nat2,2*lam+1,2*lam+1),dtype=float)
        # Get transformation matrices between real and complex spherical harmonics
        CR = complex_to_real_transformation(lam)
        CC = np.conj(CR)
        CT = np.transpose(CR)
        # Now combine with testing power spectrum
        for i in xrange(ntrain):
            pstrain = np.load('/home/app/data/train_lam2/ps2_' + str(i) +'.npy')
            for j in xrange(nat2):
                kerloc  = np.einsum('amk,bnk->abmn',pstrain,np.conj(PS[j]))
                kerloc0 = np.real(np.dot(pstrain0[i],np.conj(PS0[j].T)))
#                print "HERE",len(kerloc),len(kerloc[0]),len(kerloc[0,0]) # N 1 5
                kernel = np.zeros((2*lam+1,2*lam+1),complex)
                for iat in xrange(nat1[i]):
#                    for jat in xrange(nat2):
                    kernel += kerloc[iat,0]*kerloc0[iat,0]
                kernel /= float(nat1[i]*1)
                kreal[i,j] = np.array(np.real(np.dot(np.dot(CC,kernel),CT)),float)

    return kreal

#############################################################################################

def get_pol(kernels,type_str):

    # Get polarizability

    folder = '/home/app/data/' + type_str + '-weights/'
#    folder = '/home/app/data/'

    # First read in weights
#    weights0 = [np.load('/home/app/data/weights_lm0.npy'),np.load('/home/app/data/weights_lm0_mean.npy')]
#    weights2 = np.load('/home/app/data/weights_lm2.npy')
    weights0 = [np.load(folder + 'weights_lm0.npy'),np.load(folder + 'weights_lm0_mean.npy')]
    weights2 = np.load(folder + 'weights_lm2.npy')

    # Multiply weights vector by kernels to give predictions
    output0 = (np.dot(weights0[0].T,kernels[0]) + weights0[1])[0]
    output2 = np.dot(weights2.T,kernels[1].transpose(0,2,1,3).reshape(len(kernels[1])*len(kernels[1][0,0,0]),len(kernels[1][0])*len(kernels[1][0,0])))

    # Transform to Cartesian tensors and return
    outvec = [[np.array(output0[i]),output2[i*5:(i+1)*5]] for i in xrange(len(kernels[1][0]))]

    predcart = [spherical_to_cartesian(outvec[i]) for i in xrange(len(kernels[1][0]))]

    return predcart

#############################################################################################

def spherical_to_cartesian(vector):

    # Convert spherical tensors to Cartesian

    mask1 = [1.0,2.0**0.5,2.0**0.5,1.0,2.0**0.5,1.0]
    mask2 = [[0],[1,3],[2,6],[4],[5,7],[8]]

    sizes = [1,5]
    ns = 1
    rank = 2
    CS = get_CS_matrix(rank,mask1,mask2)
    CR = [complex_to_real_transformation(sz) for sz in [0,2]]

    outvecs = []
    outsphr = []
    sphe = np.zeros((ns,sum(sizes)),dtype=complex)
    cart = np.zeros((ns,sum(sizes)),dtype=float)
    for j in xrange(len(sizes)):
        outvecs.append(vector[j].reshape((ns,sizes[j])))
        outsphr.append(np.zeros((ns,sizes[j]),dtype=complex))
    for i in xrange(ns):
        for j in xrange(len(sizes)):
            outsphr[j][i] = np.dot(np.conj(CR[j]).T,outvecs[j][i])
        sphe[i] = np.concatenate([outsphr[j][i] for j in xrange(len(sizes))])
        cart[i] = np.real(np.dot(sphe[i],np.conj(CS).T))

        predcart = []
        for i in xrange(ns):
            crt = np.zeros(len(np.concatenate(mask2)),dtype=float)
            for j in xrange(len(mask2)):
                for k in xrange(len(mask2[j])):
                    crt[mask2[j][k]] = cart[i][j] / mask1[j]
            predcart.append(crt)
        predcart = np.concatenate(predcart)

    return predcart

#############################################################################################

def get_CS_matrix(rank,mask1,mask2):
    # Get Cartesian to spherical tensor transformation matrix
    if (rank==0):
        CS = np.array([1.0],dtype=complex)
    elif (rank==1):
        CS = np.array([[1.0,0.0,-1.0],[-1.0j,0.0,-1.0j],[0.0,np.sqrt(2.0),0.0]],dtype = complex) / np.sqrt(2.0)
    elif (rank==2):
        CS = np.array([[-1.0/np.sqrt(3.0),0.5,0.0,-1.0/np.sqrt(6.0),0.0,0.5],[0.0,-0.5j,0.0,0.0,0.0,0.5j],[0.0,0.0,0.5,0.0,-0.5,0.0],[-1.0/np.sqrt(3.0),-0.5,0.0,-1.0/np.sqrt(6.0),0.0,-0.5],[0.0,0.0,-0.5j,0.0,-0.5j,0.0],[-1.0/np.sqrt(3.0),0.0,0.0,2.0/np.sqrt(6.0),0.0,0.0]],dtype = complex)
    elif (rank==3):
        CS = np.array([[-3.0/np.sqrt(30.0),0.0,3.0/np.sqrt(30.0),1.0/np.sqrt(8.0),0.0,-3.0/np.sqrt(120.0),0.0,3.0/np.sqrt(120.0),0.0,-1.0/np.sqrt(8.0)],[1.0j/np.sqrt(30.0),0.0,1.0j/np.sqrt(30.0),-1.0j/np.sqrt(8.0),0.0,1.0j/np.sqrt(120.0),0.0,1.0j/np.sqrt(120.0),0.0,-1.0j/np.sqrt(8.0)],[0.0,-1.0/np.sqrt(15.0),0.0,0.0,1.0/np.sqrt(12.0),0.0,-1.0/np.sqrt(10.0),0.0,1.0/np.sqrt(12.0),0.0],[-1.0/np.sqrt(30.0),0.0,1.0/np.sqrt(30.0),-1.0/np.sqrt(8.0),0.0,-1.0/np.sqrt(120.0),0.0,1.0/np.sqrt(120.0),0.0,1.0/np.sqrt(8.0)],[0.0,0.0,0.0,0.0,-1.0j/np.sqrt(12.0),0.0,0.0,0.0,1.0j/np.sqrt(12.0),0.0],[-1.0/np.sqrt(30.0),0.0,1.0/np.sqrt(30.0),0.0,0.0,4.0/np.sqrt(120.0),0.0,-4.0/np.sqrt(120.0),0.0,0.0],[3.0j/np.sqrt(30.0),0.0,3.0j/np.sqrt(30.0),1.0j/np.sqrt(8.0),0.0,3.0j/np.sqrt(120.0),0.0,3.0j/np.sqrt(120.0),0.0,1.0j/np.sqrt(8.0)],[0.0,-1.0/np.sqrt(15.0),0.0,0.0,-1.0/np.sqrt(12.0),0.0,-1.0/np.sqrt(10.0),0.0,-1.0/np.sqrt(12.0),0.0],[1.0j/np.sqrt(30.0),0.0,1.0j/np.sqrt(30.0),0.0,0.0,-4.0j/np.sqrt(120),0.0,-4.0j/np.sqrt(120),0.0,0.0],[0.0,-3.0/np.sqrt(15.0),0.0,0.0,0.0,0.0,2.0/np.sqrt(10.0),0.0,0.0,0.0]],dtype=complex)
    else:
        print "This L value is not yet covered by get_CS_matrix!"
        sys.exit(0)
    for i in xrange(len(mask1)):
        CS[i] = CS[i] * mask1[i]

    return CS

#############################################################################################

def reorder_polarizabilities(pols,frame):

    atom_valence = {"H": 1,"He": 2,"Li": 3,"Be": 4,"B": 5,"C": 6,"N": 7,"O": 8,"F": 9,"Ne": 10,"Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15,"S": 16,"Cl": 17,"Ar": 18,"K": 19,"Ca": 20,"Sc": 21,"Ti": 22,"V": 23,"Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,"Cu": 29,"Zn": 30}
    atom_symbols = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'}
    nsmax = 26

    cen = [atom_symbols[num] for num in list(set([atom_valence[centre] for centre in ["C","Cl","H","N","O","S"]]))]
    all_names = frame.get_chemical_symbols()
    unique_names = list(set([name for name in cen]))
    all_species = []
    for k in unique_names:
        all_species.append(atom_valence[k])

    atom_indexes = [[] for j in xrange(nsmax)]
    natmax = len(all_names)
    for ii in xrange(natmax):
        for j in all_species:
            if all_names[ii] == atom_symbols[j]:
                atom_indexes[j].append(ii)

#    print atom_indexes
#    print len(atom_indexes)

#    reorder_data = []
#    for i in xrange(len(atom_indexes)):
#        if (len(atom_indexes[i]) > 0):
#            reorder_data.append(np.concatenate(atom_indexes[i]).astype(int))

    reorder_data = np.concatenate(atom_indexes).astype(int)
    pols2 = [None for i in xrange(len(pols))]
    for i in xrange(len(pols)):
        pols2[reorder_data[i]] = pols[i]

    return pols2
