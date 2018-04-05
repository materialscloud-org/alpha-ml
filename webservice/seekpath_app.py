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

import os,sys,flask,datetime,logging, logging.handlers
import numpy as np
import cPickle as pck
import json
from celery import Celery
from structure_importers import get_structure
from ase import Atoms
from ase.io import write

sys.path.insert(0,'/local/git/glosim2/')
sys.path.insert(0,'/home/app/git/glosim2/')
sys.path.insert(0,'/Users/iMac/git/glosim2/')
sys.path.insert(0,'/Users/fedeparuzzo/git/glosim2/')
sys.path.insert(0,'/Users/albert/Git/glosim2/')

from libmatch.soap import get_Soaps

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



Shift_ML_version = '0.0.3'


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

is_debug = True
is_fast_dev = False
is_msk = False
# is_fast_dev = False
# is_debug =  False

# logger.info('######################################')
# logger.info(os.environ.get('DEBUG'))
# logger.info('{} {}'.format(is_debug,is_fast_dev))


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
    "GUMMUW04": ("GUMMUW04"),
    "EDEBUK": ("EDEBUK")
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

app.config['CELERYD_CONCURRENCY'] = 1 # number of worker in the task pool
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
            if outputformat=='shifts_intref':
                 slopes=json.dumps({1:1,6:1,7:1,8:1})
                 intercepts=json.dumps({1:30.8,6:169.5,7:0,8:0})
            elif outputformat=='shifts_extref':
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

            print outputformat
            print slopes
            print intercepts
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

        task = predict_chemical_shifts.apply_async(
                kwargs=dict(filecontent=filecontent, fileformat=fileformat,
                		     slopes=slopes, intercepts=intercepts,
                			call_source=call_source, request_info=request_info))


        urls = {'Location': url_for('taskstatus',task_id=task.id),
                'Redirect': url_for('longtask_completed',task_id=task.id)}

        return jsonify({}), 202, urls
    else:  # GET Request
        return flask.redirect(flask.url_for('input_structure'))


@app.route('/Chemical_Shielding_Predictions/')
@app.route('/Chemical_Shielding_Predictions/<task_id>')
def longtask_completed(task_id):
    from flask import render_template
    task = predict_chemical_shifts.AsyncResult(task_id)
    #print task.info['data']
    return render_template('visualizer.html', **task.info['data'])


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = predict_chemical_shifts.AsyncResult(task_id)
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
def predict_chemical_shifts(self, filecontent, fileformat, slopes, intercepts, call_source="", request_info=None):

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

    rm_spectra()
    ##############################################################################
    if is_fast_dev is False and is_msk is False:
        shielding = get_shielding_single_kernel(self, frame, slopes, intercepts)
    elif is_fast_dev is False and is_msk is True:
        shielding = get_shielding_multi_kernel(self, frame, slopes, intercepts)
    else:
        shielding = get_dummy_work(self, frame)
    ##############################################################################

    compute_time = time.time() - start_time

    out_json_data = get_json_for_visualizer(frame)

    raw_code_dict = copy.copy(out_json_data)

    raw_code_dict.update({'IsotropicChemicalShielding': shielding.tolist()})

    raw_code = json.dumps(raw_code_dict, indent=2)
    ## I manually escape it to then add <br> and pass it to a filter with
    ## |safe. I have to 'unicode' it otherwise it keeps escaping also the
    ## next replaces
    raw_code = str(jinja2.escape(raw_code)).replace('\n', '<br>').replace(
        ' ', '&nbsp;')
    magres_code = str(jinja2.escape(get_magres(frame,shielding))).replace(
        '\n', '<br>').replace(' ', '&nbsp;')

    xyz = str(jinja2.escape(get_xyz(frame, shielding))).replace(
        '\n', '<br>').replace(' ', '&nbsp;')

    tmp=xyz.split('<br>')
    for it in range(2,len(tmp)):
        tmp[it]=rreplace(tmp[it],'&nbsp;','&nbsp;<b>',10)
        tmp[it]=rreplace(tmp[it],'&nbsp;<b>','</b>&nbsp;',9)
        tmp[it]=rreplace(tmp[it],'</b>&nbsp;','&nbsp;',8)
        tmp[it]=rreplace(tmp[it],'0000','',1)
    xyz='<br>'.join([str(x) for x in tmp])
    with open('/Users/fedeparuzzo/git/nmr_test/webservice/tmp.txt','w') as f:
        f.write(xyz)
    jsmol_pos,jsmol_cell = get_json(frame)

    data = dict(
        xyz=xyz,
        jsmol_pos=jsmol_pos,
        jsmol_cell=jsmol_cell,
        jsondata=json.dumps(out_json_data),
        raw_code=raw_code,
        magres_code=magres_code,
        shielding=shielding.tolist(),
        inputstructure_cell_vectors=inputstructure_cell_vectors,
        inputstructure_atoms_scaled=inputstructure_atoms_scaled,
        inputstructure_atoms_cartesian=inputstructure_atoms_cartesian,
        compute_time=compute_time,
        Shift_ML_version=Shift_ML_version,
    )


    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42, 'data': data }

def get_dummy_work(celery_self,frame):
    from time import sleep
    shielding = np.ones((len(frame),))
    sps = list(set(frame.get_atomic_numbers()).intersection([1, 6, 7, 8]))
    train_sizes = {1: 30000, 6: 19532, 7: 12760, 8: 14770}

    numbers = frame.get_atomic_numbers()
    for ii,sp in enumerate(sps):
        celery_self.update_state(state='PROGRESS',
                                 meta={'current': ii, 'total': len(sps),
                                       'status': str(sp)})
        sleep(1)
    return shielding

def load_model_data(sp):
    '''
    Load the model data at the start of the app
    :return:
    '''
    from seekpath_app import dataPath

    with open(dataPath + 'model_sp_{}.pck'.format(z2symb[sp]), 'rb') as f:
        model = pck.load(f)
    return model

def get_shielding_single_kernel(celery_self,frame, slopes, intercepts):
    import json
    shielding = np.zeros((len(frame),))
    sps = list(set(frame.get_atomic_numbers()).intersection([1, 6, 7, 8]))

    celery_self.update_state(state='PROGRESS',
                             meta={'current': 0, 'total': len(sps),
                                   'status': ''})
    numbers = frame.get_atomic_numbers()

    if slopes!=None:
        slopes=json.loads(slopes)
        intercepts=json.loads(intercepts)

    for ii,sp in enumerate(sps):

        atoms = ase2qp(frame)
        model = load_model_data(sp)
        ll = list(sps)
        ll.remove(sp)
        soap_params = dict(
            centerweight=1., cutoff_transition_width=0.5, spkitMax=spkitMax,
            nocenters=ll, is_fast_average=False, chem_channels=False, dispbar=True
        )
        soap_params.update(model['soap_params'])

        descriptor = get_fingerprints([atoms], soap_params, nprocess=1, sp=sp)

        kernel_test = np.dot(model['soap_ref'], descriptor.T)

        y_pred = predict(kernel_test, model['model'].alpha, model['model'].csi)
        if slopes!=None:
            slope=slopes[str(sp)]
            intercept=intercepts[str(sp)]
            y_pred=shielding2shifts(y_pred,slope,intercept)

        shielding[frame.get_atomic_numbers() == sp] = y_pred
        #create_spectrum(sp, y_pred, is_debug)

        celery_self.update_state(state='PROGRESS',
                                 meta={'current': ii, 'total': len(sps),
                                       'status': str(sp)})

    np.round(shielding, decimals=4, out=shielding)
    return shielding

def load_model_data_msk(sp,cutoff):
    '''
    Load the model data at the start of the app
    :return:
    '''
    from seekpath_app import dataPath

    with open(dataPath + 'model_sp_{}_c_{}.pck'.format(z2symb[sp],cutoff), 'rb') as f:
        model = pck.load(f)
    return model

def get_shielding_multi_kernel(celery_self,frame, slopes, intercepts):
    import json

    shielding = np.zeros((len(frame),))
    sps = list(set(frame.get_atomic_numbers()).intersection([1, 6, 7, 8]))
    train_sizes = {1:30000,6:19532,7:12760,8:14770}
    celery_self.update_state(state='PROGRESS',
                             meta={'current': 0, 'total': len(sps),
                                   'status': ''})
    numbers = frame.get_atomic_numbers()

    if slopes!=None:
        slopes=json.loads(slopes)
        intercepts=json.loads(intercepts)

    for ii,sp in enumerate(sps):
        kernel_test = np.zeros((train_sizes[sp],len(numbers[numbers==sp])))
        kk = np.zeros((train_sizes[sp],len(numbers[numbers==sp])))
        model_msk = load_model_data_msk(sp, 'msk')
        msk_coefs = model_msk['msk_params']
        for cutoff in [2,3,4,5,6,7]:
            atoms = ase2qp(frame)
            model = load_model_data_msk(sp,cutoff)
            ll = list(sps)
            ll.remove(sp)
            soap_params = dict(
                centerweight=1., cutoff_transition_width=0.5, spkitMax=spkitMax,
                nocenters=ll, is_fast_average=False, chem_channels=False, dispbar=True
            )
            soap_params.update(model['soap_params'])

            descriptor = get_fingerprints([atoms], soap_params, nprocess=1, sp=sp)

            np.dot(model['soap_ref'], descriptor.T,out=kk)
            np.add(np.multiply(msk_coefs[cutoff],kk,out=kk),kernel_test,out=kernel_test)


        y_pred = predict(kernel_test, model_msk['model'].alpha, model_msk['model'].csi)
        if slopes!=None:
            slope=slopes[str(sp)]
            intercept=intercepts[str(sp)]
            y_pred=shielding2shifts(y_pred,slope,intercept)

        shielding[numbers == sp] = y_pred
        #create_spectrum(sp,y_pred,is_debug)

        celery_self.update_state(state='PROGRESS',
                                 meta={'current': ii, 'total': len(sps),
                                       'status': str(sp)})

    np.round(shielding, decimals=4, out=shielding)
    return shielding

def get_json_for_visualizer(frame):
    # TODO get a working jmol
    response = {}
    response['numbers'] = frame.get_atomic_numbers().tolist()
    response['positions'] = frame.get_positions().tolist()
    response['cell'] = frame.get_cell().tolist()
    response['pbc'] = frame.get_pbc().tolist()

    return response

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

def get_cif(frame):
    from ase.io.cif import write_cif
    my_cif = DummyFileObject()
    write_cif(my_cif,frame)
    return my_cif.read()

def get_magres(frame,shielding):
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
    write_magres(f_obj, new_frame)
    return f_obj.read()

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
        cell_string += ' {:.4f} '.format(ss)
    return pos_string, cell_string

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



def shielding2shifts(shielding,slope,intercept):
    shifts=float(intercept)-float(slope)*shielding
    return shifts

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def rm_spectra():
    import subprocess
    simbols={1:'H',6:'C',7:'N',8:'O'}
    if is_debug:
        fig_name='static/jsmol/data/uploaded/*SPECTRUM*'
        fig_name =os.path.abspath(fig_name)
    else:
        fig_name ='/home/app/code/webservice/static/jsmol/data/*SPECTRUM*'

    command='rm {}'.format(fig_name)
    subprocess.call(command, shell=True)

def create_spectrum(sp, shielding, is_debug):
    import matplotlib.pyplot as plt
    colors={1:'#1d4484',6:'#E42828',7:'#A3CB58',8:'#F47531'}
    increment={1:2,6:5,7:10,8:10}
    simbols={1:'H',6:'C',7:'N',8:'O'}
    labels={1:'$^1$H',6:'$^{13}$C',7:'$^{15}$N',8:'$^{17}$O'}


    #llim={1:-2,6:-5,7:-10,8:-50}
    #rlim={1:12,6:200,7:400,8:400}
    #if llim[sp]>min(shielding[numbers == sp]):
    #    llim[sp]=min(shielding[numbers == sp])-increment[sp]
    #if rlim[sp]<max(shielding[numbers == sp]):
    #    rlim[sp]=max(shielding[numbers == sp])+increment[sp]
    #t = np.arange(llim[sp], rlim[sp], 0.01)

    f = np.arange((min(shielding)-increment[sp]), (max(shielding)+increment[sp]), 0.01)
    i = np.zeros(len(f))
    idx=[np.isclose(f, it) for it in shielding]
    i[np.where(sum(idx)<>0)]=1

    plt.figure(figsize=(12,5),frameon=False)
    plt.tick_params(axis='y',which='both',left='off', labelleft='off')
    plt.plot(f,i,color=colors[sp])
    ax = plt.gca()
    ax.tick_params(direction='in', length=3, width=2, labelsize=16, pad=10)
    ax.yaxis.set_ticks_position('none')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(min(f), max(f));
    ax.set_xlabel("{} Chemical Shifts / ppm".format(labels[sp]),fontsize=22)

    if is_debug:
        fig_name = 'static/jsmol/data/uploaded/{}SPECTRUM_UPLOADED.png'.format(simbols[sp])
        fig_name = os.path.abspath(fig_name)
    else:
        fig_name = '/home/app/code/webservice/static/jsmol/data/{}SPECTRUM_UPLOADED.png'.format(simbols[sp])

    plt.savefig(fig_name, bbox_inches='tight')
    return 1


if __name__ == "__main__":
    # Don't use x-sendfile when testing it, because this is only good
    # if deployed with Apache
    # Use the local version of seekpath, not the installed one

    app.use_x_sendfile = False
    app.run(debug=True)
