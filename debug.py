import os
import json
import torch
import torchaudio
import base64
import io
import tempfile
import bottle
from bottle import run
from bottle import route
from bottle import post
from bottle import request
from bottle import static_file
from yakamoz import indices
from yakamoz import neutral
from yakamoz import inference
from yakamoz import ROOT


@post('/mask')
def mask():
    mood = json.loads(request.forms.get('mood'))
    audio = request.forms.get('audio')
    frameCount = int(float(request.forms.get('frameCount')))
    return json.dumps(list(map(lambda x: x.tolist(), inference(frameCount, audio, mood))))


@route('/maskIndices')
def maskIndices():
    return json.dumps(indices)


@route('/maskNeutral')
def maskNeutral():
    return json.dumps(neutral)


@route('/')
def index():
    return static_file('index.html', os.path.join(ROOT, 'static'))


@route('/<staticFile>')
def staticStuff(staticFile):
    return static_file(staticFile, os.path.join(ROOT, 'static'))


@route('/external/js/<staticFile>')
def externalJS(staticFile):
    return static_file(staticFile, os.path.join(ROOT, 'external/js/'))


@route('/external/fonts/<staticFile>')
def externalFonts(staticFile):
    return static_file(staticFile, os.path.join(ROOT, 'external/fonts/'))


if __name__ == '__main__':
    run(server='bjoern')
