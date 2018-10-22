#!/usr/bin/env python
"""
Main Flask python function that manages the server backend

If you just want to try it out, just run this file and connect to
http://localhost:5000 from a browser. Otherwise, read the instructions
in README_DEPLOY.md to deploy on a Apache server.
"""



from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()

# https://github.com/ContinuumIO/anaconda-issues/issues/6678
# there are some warnings about numpy that are useless
import warnings
warnings.filterwarnings("always")

import os,sys,flask,datetime,logging, logging.handlers
import numpy as np
import cPickle as pck
import json
from celery import Celery
from structure_importers import get_structure
from ase import Atoms
from ase.io import write
import datetime
from sagpr import *

logger = logging.getLogger("seekpath_server")
logHandler = logging.handlers.TimedRotatingFileHandler(
    os.path.join(
        os.path.split(os.path.realpath(__file__))[0], 'logs', 'requests.log'),
    when='midnight')
formatter = logging.Formatter(
    '[%(asctime)s]%(levelname)s-%(funcName)s ^ %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.DEBUG)



ALPHA_ML_version = '1.0 Beta'

MAX_NUMBER_OF_ATOMS = 2000

PROGRESS_BAR_TIMEOUT = 300 # every 1 sec the progress bar is updated

if os.environ.get('DEBUG') == 'false':
    is_debug = False
else:
    is_debug = True

if os.environ.get('FAST_DEV') == 'false':
    is_fast_dev = False
else:
    is_fast_dev = True

if os.environ.get('MSK') == 'false':
    is_msk = False
else:
    is_msk = True

NB_OF_WORKERS = int(os.environ.get('NB_OF_WORKERS'))

dataPath = '/home/app/data/'
if is_debug:
    dataPath = '../data/'



# This (undocumented) flag changes the style of the webpage (CSS, etc.)
# and decides whether some of the headers (e.g. the SeeK-path title) and the
# description of what seekpath can do should appear or not
#
# Options:
# - 'lite': simple version, not title, no info description, different CSS
# - anything else: default
#
# How to pass: with Apache, when forwarding, in a ReverseProxy section, add
#   RequestHeader set X-Seekpath-Style lite
def get_style_version(request):
    return request.environ.get("HTTP_X_SEEKPATH_STYLE", "")



class ConfigurationError(Exception):
    pass


valid_examples = {
    "Caffeine": ("Caffeine"),
    "Aciclovir": ("Aciclovir"),
    "Cytosine": ("Cytosine"),
    "Methionine": ("Methionine"),
}

if __name__ == "__main__":
    # If run manually (=> debug/development mode),
    # use the local version of seekpath, not the installed one
    import sys
    sys.path.insert(0, os.path.join(os.path.split(__file__)[0], os.pardir))


static_folder = os.path.join(
    os.path.split(os.path.realpath(__file__))[0], 'static')

view_folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'view')
app = flask.Flask(__name__, static_folder=static_folder)
app.use_x_sendfile = True
directory = os.path.split(os.path.realpath(__file__))[0]

#### CELERY
# Expects the default redis configuration
if is_debug:
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
else:
    REDIS_PORT = os.environ.get('REDIS_PORT')
    app.config['CELERY_BROKER_URL'] = 'redis://:devpassword@redis:{}/0'.format(REDIS_PORT)
    app.config['CELERY_RESULT_BACKEND'] = 'redis://:devpassword@redis:{}/0'.format(REDIS_PORT)

app.config['CELERYD_CONCURRENCY'] = NB_OF_WORKERS # number of worker in the task pool
app.config['CELERYD_HIJACK_ROOT_LOGGER'] = True


celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

try:
    with open(os.path.join(directory, 'SECRET_KEY')) as f:
        app.secret_key = f.readlines()[0].strip()
        if len(app.secret_key) < 16:
            raise ValueError
except Exception:
    raise ConfigurationError(
        "Please create a SECRET_KEY file in {} with a random string "
        "of at least 16 characters".format(directory))


class ReverseProxied(object):
    '''Wrap the application in this middleware and configure the
    front-end server to add these headers, to let you quietly bind
    this to a URL other than / and to an HTTP scheme that is
    different than what is used locally.

    Inspired by  http://flask.pocoo.org/snippets/35/

    In apache: use the following reverse proxy (adapt where needed)
    <Location /proxied>
      ProxyPass http://localhost:4444/
      ProxyPassReverse http://localhost:4444/
      RequestHeader set X-Script-Name /proxied
      RequestHeader set X-Scheme http
    </Location>

    :param app: the WSGI application
    '''

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        scheme = environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        server = environ.get('HTTP_X_FORWARDED_HOST', '')
        if server:
            environ['HTTP_HOST'] = server
        return self.app(environ, start_response)


app.wsgi_app = ReverseProxied(app.wsgi_app)

def get_visualizer_select_template(request):
    if get_style_version(request) == 'lite':
        return 'visualizer_select_lite.html'
    else:
        return 'visualizer_select.html'

def get_visualizer_template(request):
    if get_style_version(request) == 'lite':
        return 'visualizer_lite.html'
    else:
        return 'visualizer.html'

def get_info_from_request(request):
    request_info = {'request': str(request.headers),
            'source': request.headers.get('X-Forwarded-For', request.remote_addr),
            'form':dict(request.form),

    }
    return request_info


logger.debug("Start")

# From http://arusahni.net/blog/2014/03/flask-nocache.html
## Add @nocache right between @app.route and the 'def' line
from functools import wraps, update_wrapper


def nocache(view):

    @wraps(view)
    def no_cache(*args, **kwargs):
        response = flask.make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.datetime.now()
        response.headers[
            'Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


@app.route('/')
def index():
    """
    Main view, redirect to input_structure
    """
    return flask.redirect(flask.url_for('input_structure'))

def home_link():
    """
    Main view, redirect to input_structure
    """
    return flask.redirect(flask.url_for('input_structure'))

@app.route('/termsofuse/')
def termsofuse():
    """
    View for the terms of use
    """
    return flask.send_from_directory(view_folder, 'termsofuse.html')


@app.route('/input_structure/')
def input_structure():
    """
    Input structure selection
    """
    return flask.render_template(get_visualizer_select_template(flask.request))

# use template js for the submit buttons
@app.route("/submit_js/<name_path>")
def submit_js(name_path):
    # print name_path
    name, path = name_path.split('-')
    url = flask.url_for(path)
    inp = dict(form_name=name,url=url,timeout=PROGRESS_BAR_TIMEOUT)
    return flask.render_template("/js/ajax_submit.js",**inp)


@app.route('/process_structure', methods=['GET', 'POST'])
def process_structure():
    import uuid
    from flask import request,jsonify,url_for,redirect
    from seekpath_web_module import read_test_frame,FlaskRedirectException

    if request.method == 'POST':
        # Submit user defined structure
        slopes=None
        intercepts=None
        if 'structurefile' in request.files:
            structurefile = flask.request.files['structurefile']
            fileformat = flask.request.form.get('fileformat', 'unknown')
            outputformat = flask.request.form.get('outputformat', 'unknown')
            #email = flask.request.form.get('email','example@example.com')
            #write_email(email)
            if outputformat=='shifts_intref':
                 slopes=json.dumps({1:1,6:1,7:1,8:1})
                 intercepts=json.dumps({1:30.8,6:169.5,7:185,8:248})
            elif outputformat=='shifts_extref':                                     ###TODO: THIS SHOULD BE TIDIED UP.
                 Hs=flask.request.form.get('Hs','unknown')
                 Hi=flask.request.form.get('Hi','unknown')
                 Cs=flask.request.form.get('Cs','unknown')
                 Ci=flask.request.form.get('Ci','unknown')
                 Ns=flask.request.form.get('Ns','unknown')
                 Ni=flask.request.form.get('Ni','unknown')
                 Os=flask.request.form.get('Os','unknown')
                 Oi=flask.request.form.get('Oi','unknown')
                 slopes={1:Hs,6:Cs,7:Ns,8:Os}
                 intercepts={1:Hi,6:Ci,7:Ni,8:Oi}
                 slopes=json.dumps(slopes)
                 intercepts=json.dumps(intercepts)

            filecontent = structurefile.read()
            call_source = "process_structure"

        # Use example structures
        elif 'examplestructure' in request.form:

            example_name = request.form.get('examplestructure')
            try:
                example_name = valid_examples[example_name]
            except:
                flask.flash("This is not a valid example: {}".format(example_name))
                return flask.redirect(flask.url_for('input_structure'))

            fileformat = 'xyz-ase'
            call_source = "process_example_structure"

            # I expect that the valid_examples dictionary already filters only
            # existing files, so I don't try/except here
            file_name = '/home/app/code/webservice/file_examples/' + example_name + '.xyz'
            if is_debug:
                file_name = './file_examples/' + example_name + '.xyz'

            with open(file_name, 'r') as f:
                filecontent = f.read()
        else:
            return redirect(url_for('input_structure'))

        request_info = get_info_from_request(request)

        try:
            read_test_frame(filecontent,fileformat,call_source,request_info)

        except FlaskRedirectException as e:
            print e.message
            # flask.flash(e.message)
            # return flask.redirect(flask.url_for('input_structure'))
            return jsonify(message=e.message),500

        task = predict_polarizabilities.apply_async(
                kwargs=dict(filecontent=filecontent, fileformat=fileformat,
                		     slopes=slopes, intercepts=intercepts,
                			call_source=call_source, request_info=request_info))


        urls = {'Location': url_for('taskstatus',task_id=task.id),
                'Redirect': url_for('longtask_completed',task_id=task.id)}

        return jsonify({}), 202, urls
    else:  # GET Request
        return flask.redirect(flask.url_for('input_structure'))


@app.route('/Polarizability_Predictions/')
@app.route('/Polarizability_Predictions/<task_id>')
def longtask_completed(task_id):
    from flask import render_template
    task = predict_polarizabilities.AsyncResult(task_id)
    #print task.info['data']
    return render_template('visualizer.html', **task.info['data'])


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = predict_polarizabilities.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state in ['STARTED','PROGRESS']:
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']

    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return flask.jsonify(response)

@app.context_processor
def utility_processor():
    def format_shifts(shifts):
        ss = []
        for ms in shifts:
            ss.append('{0:.3f}'.format(ms))
        return ' '.join(ss)
    return dict(format_shifts=format_shifts)



########################################
# THE MAIN FUNCTIONALITY

@celery.task(bind=True)
def predict_polarizabilities(self, filecontent, fileformat, slopes, intercepts, call_source="", request_info=None):

    # from structure_importers import get_structure
    import io,time,copy, json, jinja2
    from ase.data import chemical_symbols
    from celery.utils.log import get_task_logger

    start_time = time.time()
    fileobject = io.StringIO(unicode(filecontent))
    # form_data = dict(request_info.form)
    form_data = request_info['form']
    LOG = get_task_logger(__name__)

    frame = get_structure(
        fileobject, fileformat, extra_data=form_data)


    in_json_data = {
        'cell': [v.tolist() for v in frame.get_cell()],
        'scaled_coords': [v.tolist() for v in frame.get_scaled_positions()],
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

    # Now calculate the polarizability for this molecule
    # Compute power spectra for this frame
    lm = [0,2]
    types_list = ['ccsd','b3lyp','scan0','delta-b3lyp-ccsd','delta-scan0-ccsd']
    all_PS  = [np.array(get_power_spectrum(lam,frame)[0]) for lam in lm]
    # Split power spectrum into atomic spectra
    all_PS = split_power_spectra(all_PS,frame)
    # Combine with training set spectra
    all_ker = [get_kernel(lm[i],all_PS[i],frame,all_PS[0]) for i in xrange(len(lm))]
    # Apply weights to predict polarizability
    all_pols = {}
    all_pol2 = {}
    all_polarizability = {}
    for typ in types_list:
        pols = np.array(get_pol(all_ker,typ))
        # Reorder the atom-centred polarizabilities
        pols = np.array(reorder_polarizabilities(pols,frame))
        pol2 = (pols * 1e3).astype(int)
        polarizability = sum(pols)
        all_pols[typ] = pols
        all_pol2[typ] = pol2
        all_polarizability[typ] = polarizability

    # For the atom-centred polarizabilities, we use the user-chosen model
    select_type = form_data['modeltype'][0]
    #print 'select type=',select_type
    pols = all_pols[select_type]
    pol2 = all_pol2[select_type]
    polarizability = all_polarizability[select_type]

    compute_time = time.time() - start_time

    out_json_data = get_json_for_visualizer(frame,pols)

    raw_code_dict = copy.copy(out_json_data)

    raw_code = json.dumps(raw_code_dict, indent=2)
    ## I manually escape it to then add <br> and pass it to a filter with
    ## |safe. I have to 'unicode' it otherwise it keeps escaping also the
    ## next replaces
    raw_print = str(raw_code)
    raw_code = str(jinja2.escape(raw_code)).replace('\n', '<br>').replace(
        ' ', '&nbsp;')

    xyz = str(jinja2.escape(get_xyz(frame,polarizability,pols,all_polarizability,types_list,select_type))).replace(
        '\n', '<br>').replace(' ', '&nbsp;')
    xyz_print = str(get_xyz(frame,polarizability,pols,all_polarizability,types_list,select_type))

    jsmol_pos,jsmol_cell = get_json(frame)

    # Get scale factor for ellipsoids
#    eigmax = 0.0
#    for i in xrange(len(frame)):
#        [eigenvals,eigenvecs] = np.linalg.eigh(np.reshape(pols[i],(3,3)))
#        eigmax = max(eigmax,max(eigenvals))
#    scalefac = 4.7 / np.sqrt(eigmax)
    scalefac = 0.25
    eig_threshold = 0.45
    if 'delta' in select_type:
        scalefac = 0.6
#        eig_threshold = 0.1
#        eig_threshold = 0.25
##        eig_threshold = 0.15
#        eig_threshold = (0.25 / scalefac) * 0.45
#    print eig_threshold

    # Get ellipsoids
    colour_dict = {'H': 'xC0C0C0', 'C': 'x505050', 'N': 'x0000CC', 'O': 'xCC0000', 'S': 'xFFC832', 'Cl': 'x00CC00'}
    jsmol_ellipsoid = ''
    for i in xrange(len(frame)):
        [eigenvals,eigenvecs] = np.linalg.eigh(np.reshape(pols[i],(3,3)))
        eigenvecs = eigenvecs.T
#        if i == 2:
#            print eigenvals
#        axes = [eigenvecs[j] * np.sqrt(np.abs(max(eig_threshold,eigenvals[j]))) for j in xrange(3)]
        axes = [eigenvecs[j] * np.sqrt(np.abs(eigenvals[j])) for j in  xrange(3)]
        axis_str =  ' '.join(str(e) for e in ['{' + ('{:16.12f}'*3).format(*axes[j]) + '}' for j in xrange(3)])
#        if i == 2:
#            print [np.sqrt(np.abs(max(eig_threshold,eigenvals[j]))) for j in xrange(3)]
#            print axis_str
#            print '[[' + ', '.join(str(e) for e in eigenvecs[0]) + '], [' + ', '.join(str(e) for e in eigenvecs[1]) + '], [' + ', '.join(str(e) for e in eigenvecs[2]) + ']]'
        jsmol_ellipsoid += 'ellipsoid ID ' + str(i) + ' AXES ' + axis_str +' CENTER {' + ' '.join(str(e) for e in frame.get_positions()[i]) + '} OPTIONS \"ball\" COLOR [' + colour_dict[frame.get_chemical_symbols()[i]] + '];'
#    scalefac = 0.25
    for i in xrange(len(frame)):
        jsmol_ellipsoid += 'ellipsoid ID ' + str(i) + ' SCALE ' + str(scalefac) + ';'

    if 'delta' in select_type:
        eigmin = 1.e10
        for i in xrange(len(frame)):
            [eigenvals,eigenvecs] = np.linalg.eigh(np.reshape(pols[i],(3,3)))
            eigmin = min(eigmin,min(eigenvals))
        print "MINIMUM EIGENVAL",eigmin

#    if 'delta' in select_type:
#        eig_threshold = 0.15
#        jsmol_ellipsoid = ''
##        for i in xrange(len(frame)):
#        for i in xrange(2):
#            [eigenvals,eigenvecs] = np.linalg.eigh(np.reshape(pols[i],(3,3)))
#            eigenvecs = eigenvecs.T
#            axes = [eigenvecs[j] * np.sqrt(np.abs(max(eig_threshold,eigenvals[j]))) for j in xrange(3)]
#            axis_str =  ' '.join(str(e) for e in ['{' + ('{:8.4f}'*3).format(*axes[j]) + '}' for j in xrange(3)])
#            print axis_str
#            jsmol_ellipsoid += 'ellipsoid ID ' + str(i) + ' AXES ' + axis_str +' CENTER {' + ' '.join(str(e) for e in frame.get_positions()[i]) + '} OPTIONS \"ball\" COLOR [' + colour_dict[frame.get_chemical_symbols()[i]] + '];'

#    # What is the largest eigenvector?
#    eigmax = 0.0
#    for i in xrange(len(frame)):
#        [eigenvals,eigenvecs] = np.linalg.eigh(np.reshape(pols[i],(3,3)))
#        eigmax = max(eigmax,max(eigenvals))
#    print "EIGMAX",0.25 * eigmax

    # Get principal axes for ellipsoids, along with the sign of the eigenvalue

    # Get PDB file with ANISOU records
    pdb_print = 'MODEL     1	 1\n'
    for typ in types_list:
        pdb_print += 'REMARK ' + typ + '_polarizability="' + (' {:f}'*9).format(*all_polarizability[typ]) + '"\n'
    pdb_print += 'REMARK ' + select_type + ' polarizability given per-atom'
    for i in xrange(len(frame)):
        pdb_print += 'ATOM  ' + '{:>5d}'.format(i) + '{:>5s}'.format(frame.get_chemical_symbols()[i]) + ' MOL     1    ' + ('{:8.3f}'*3).format(*(frame.get_positions()[i])) + '  1.00  0.00' + '{:>12s}'.format(frame.get_chemical_symbols()[i]) + '\n'
        pdb_print += 'ANISOU' + '{:>5d}'.format(i) + '{:>5s}'.format(frame.get_chemical_symbols()[i]) + ' MOL     1  ' + ('{:>6d}'*6).format(pol2[i][0],pol2[i][4],pol2[i][8],pol2[i][1],pol2[i][2],pol2[i][7]) + '{:>14s}'.format(frame.get_chemical_symbols()[i]) + '\n'
    pdb_print += 'ENDMDL'

    pdb = str(jinja2.escape(pdb_print)).replace('\n','<br>').replace(' ', '&nbsp;')

    data = dict(
        xyz=xyz,
        xyz_print=xyz_print,
        pdb=pdb,
        pdb_print=pdb_print,
        jsmol_pos=jsmol_pos,
        jsmol_ellipsoid=jsmol_ellipsoid,
        jsondata=json.dumps(out_json_data),
        raw_code=raw_code,
        raw_print=raw_print,
        inputstructure_cell_vectors=inputstructure_cell_vectors,
        inputstructure_atoms_scaled=inputstructure_atoms_scaled,
        inputstructure_atoms_cartesian=inputstructure_atoms_cartesian,
        compute_time=compute_time,
        ALPHA_ML_version=ALPHA_ML_version,
    )


    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42, 'data': data }

def load_model_data(sp):
    '''
    Load the model data at the start of the app       # DMW: THIS NEEDS TO BE CHANGED MAYBE
    :return:
    '''
    from seekpath_app import dataPath

    with open(dataPath + 'model_sp_{}.pck'.format(z2symb[sp]), 'rb') as f:
        model = pck.load(f)
    return model

def load_model_params(sp,cutoff):
    '''
    Load the model data at the start of the app
    :return:
    '''
    from seekpath_app import dataPath

    with open(dataPath + 'model_sp_{}_c_{}-model_params.json'.format(z2symb[sp],cutoff), 'r') as f:
        model_params = json.load(f)
    return model_params

def load_model_ref(sp,cutoff):
    from seekpath_app import dataPath
    fn = dataPath+ 'model_sp_{}_c_{}-soap_ref.npy'.format(z2symb[sp],cutoff)
    return np.load(fn,mmap_mode='r')

def get_json_for_visualizer(frame,pols):
    # TODO get a working jmol
    response = {}
    response['positions'] = frame.get_positions().tolist()
    # Put in polarizabilities
    response['polarizabilities'] = [pols[i].tolist() for i in xrange(len(pols))]

    return response

class DummyFileObject(object):
    def __init__(self):
        self.data = ''
    def write(self,text):
        self.data += text
    def read(self):
        return self.data
'''
def write_email(email):
    with open("static/request_log.txt", "a") as myfile:
        myfile.write(email + '\t' + str(datetime.datetime.now()) + '\n')
'''
def get_xyz(frame,polarizability,pols,all_polarizability,type_list,select_type):
    from ase.io.extxyz import write_xyz
    numbers = frame.get_atomic_numbers()
    positions = frame.get_positions()
    symbols = frame.get_chemical_symbols()
    cell = frame.get_cell()
    pbc = frame.get_pbc()
    new_frame = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    f_obj = DummyFileObject()

    write_xyz(f_obj, new_frame)

    #add disclaimer as comment in 2nd line
    disclaimer_str = ''
#    disclaimer_str += ' CCSD_polarizability="' + (' {:f}'*9).format(*polarizability) + '"'
    for typ in type_list:
        disclaimer_str += ' ' + typ + '_polarizability="' + (' {:f}'*9).format(*all_polarizability[typ]) + '"'
    disclaimer_str += ' Comments="The polarizabilities were generated by ALPHA-ML v' + ALPHA_ML_version + ' (https://alphaml.epfl.ch).'
    disclaimer_str += ' They are subject to the terms of use (https://alphaml.epfl.ch).'
    disclaimer_str += ' Please read and cite the following:'
    disclaimer_str += ' David M. Wilkins, Andrea Grisafi, Yang Yang, Ka Un Lao, Robert A. DiStasio and Michele Ceriotti: Accurate molecular polarizabilities by coupled-cluster theory and machine learning (submitted).'
    disclaimer_str += ' per-atom polarizability for ' + select_type + ' is given."'

    # Insert the atom-centred polarizabilities
    xyz_file = str(len(frame.get_chemical_symbols())) + '\n'
    xyz_file += 'Lattice="' + str(frame.get_cell()[0][0]) + ' 0.0 0.0 0.0 ' + str(frame.get_cell()[1][1]) + ' 0.0 0.0 0.0 ' + str(frame.get_cell()[2][2]) + '" Properties=species:S:1:pos:R:3:axx:R:1:ayy:R:1:azz:R:1:axy:R:1:axz:R:1:ayz:R:1 ' + disclaimer_str + '\n'
    for i in xrange(len(positions)):
        xyz_file += symbols[i] + ('{:>17.8f}'*3).format(positions[i][0],positions[i][1],positions[i][2]) + '    ' + ('{:>11.6f}'*6).format(pols[i][0],pols[i][4],pols[i][8],pols[i][1],pols[i][2],pols[i][7]) + '\n'

    return xyz_file

def get_cif(frame):
    from ase.io.cif import write_cif
    my_cif = DummyFileObject()
    write_cif(my_cif,frame)
    return my_cif.read()


def get_json(frame):
    # {"mol":{"a":[{"l":H,"x":0,"y":0.5,"z":0},{"l":C,"x":1,"y":0.5,"z":0}],"b":[]}}
    pos_string = '{\"mol\":{\"a\":['
    for sym, pos in zip(frame.get_chemical_symbols(), frame.get_positions()):
        pos_string += '{' + '"l":{sym},"x":{p1:.3f},"y":{p2:.3f},"z":{p3:.3f}'.format(sym=sym, p1=pos[0], p2=pos[1],
                                                                                      p3=pos[2]) + '},'

    pos_string += '],\"b\":[]}}'
    cell = frame.get_cell().flatten()
    cell_string = ''
    for ss in cell:
#        cell_string += ' {:.4f} '.format(ss)
        cell_string += ' {:.4f} '.format(0.0)

    return pos_string, cell_string

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

def predict(kernel, alpha, csi):
    return np.dot(alpha, np.power(kernel, csi, out=kernel)).reshape((-1))

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)
