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

import datetime
import flask
import os
import numpy as np
import cPickle as pck
from celery import Celery
import redis



from seekpath_web_module import (FlaskRedirectException, PROGRESS_BAR_TIMEOUT,)


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


import logging, logging.handlers

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


class ConfigurationError(Exception):
    pass


valid_examples = {
    "GUMMUW04": ("GUMMUW04"),
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
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
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



@celery.task(bind=True)
def process_frame(self,filecontent, fileformat, call_source="", request_info=None):
    from structure_importers import get_structure
    import io,time,copy, json, jinja2
    from ase.data import chemical_symbols
    from seekpath_web_module import (get_dummy_work,get_shielding_multi_kernel,
                is_debug,is_fast_dev,get_shielding_single_kernel,get_json_for_visualizer,
                                     get_xyz, Shift_ML_version)

    start_time = time.time()
    fileobject = io.StringIO(unicode(filecontent))
    # form_data = dict(request_info.form)
    form_data = request_info['form']


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
    ##############################################################################

    if is_debug is False and is_fast_dev is False:
        shielding = get_shielding_single_kernel(frame)
        # shielding = get_shielding_multi_kernel(frame)
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

    xyz = str(jinja2.escape(get_xyz(frame, shielding))).replace(
        '\n', '<br>').replace(' ', '&nbsp;')

    data = dict(
        xyz=xyz,
        jsondata=json.dumps(out_json_data),
        raw_code=raw_code,
        shielding=shielding.tolist(),
        inputstructure_cell_vectors=inputstructure_cell_vectors,
        inputstructure_atoms_scaled=inputstructure_atoms_scaled,
        inputstructure_atoms_cartesian=inputstructure_atoms_cartesian,
        compute_time=compute_time,
        Shift_ML_version=Shift_ML_version,
    )


    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42, 'data': data }



@app.route('/process_structure', methods=['GET', 'POST'])
def process_structure():
    from flask import request,jsonify,url_for,redirect
    from seekpath_web_module import is_debug,read_test_frame

    if request.method == 'POST':

        # Submit user defined structure
        if 'structurefile' in request.files:
            structurefile = flask.request.files['structurefile']
            fileformat = flask.request.form.get('fileformat', 'unknown')
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


        task = process_frame.apply_async(
                kwargs=dict(filecontent=filecontent, fileformat=fileformat,
                call_source=call_source, request_info=request_info))

        urls = {'Location': url_for('taskstatus',task_id=task.id),
                'Redirect': url_for('longtask_completed',task_id=task.id)}

        return jsonify({}), 202, urls
    else:  # GET Request
        return flask.redirect(flask.url_for('input_structure'))


@app.route('/Chemical_Shielding_Result/')
@app.route('/Chemical_Shielding_Result/<task_id>')
def longtask_completed(task_id):
    from flask import render_template
    task = process_frame.AsyncResult(task_id)
    print task.info['data']
    return render_template('visualizer.html', **task.info['data'])


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = process_frame.AsyncResult(task_id)
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




if __name__ == "__main__":
    # Don't use x-sendfile when testing it, because this is only good
    # if deployed with Apache
    # Use the local version of seekpath, not the installed one

    app.use_x_sendfile = False
    app.run(debug=True)
