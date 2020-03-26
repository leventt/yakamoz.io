import os
import random
import json
import torch
import torchaudio
import base64
import io
import tempfile
import numpy as np
import zfpy
import bottle
from bottle import run
from bottle import route
from bottle import post
from bottle import request
from bottle import static_file
from indices import indices
from neutral import neutral


app = bottle.default_app()

indices = (torch.Tensor(indices) - 1).tolist()
ROOT = os.path.expanduser('~/sandbox/yakamoz.io/')
if not os.path.exists(ROOT):
    ROOT = os.path.dirname(__file__)
tracedScriptPath = os.path.join(ROOT, 'yakamoz.pt')


def inference(frameCount, audio, mood):
    if not os.path.exists(tracedScriptPath):
        print('torch script not found')
        return
    else:
        tracedScript = torch.jit.load(tracedScriptPath)

    tokens = audio.split(';base64,')
    audio = tokens[1]
    audio = io.BytesIO(base64.b64decode(audio))
    with tempfile.NamedTemporaryFile(suffix='.' + tokens[0].split('/')[1]) as audioFile:
        audioFile.write(audio.read())
        waveform, sampleRate = torchaudio.load(audioFile.name)

    if sampleRate != 16000:
        waveform = torchaudio.transforms.Resample(sampleRate, 16000)(waveform)
        sampleRate = 16000

    MFCC = torchaudio.compliance.kaldi.mfcc(
        waveform,
        channel=0,
        remove_dc_offset=True,
        window_type='hanning',
        num_ceps=32,
        num_mel_bins=64,
        frame_length=256,
        frame_shift=32
    )
    MFCCLen = MFCC.size()[0]

    allMFCC = torch.Tensor([])
    for i in range(frameCount):
        audioIdxRoll = int(i * (MFCCLen / frameCount))
        allMFCC = torch.cat(
            (
                allMFCC,
                torch.cat(
                    (
                        torch.roll(
                            MFCC,
                            (audioIdxRoll * -1) + 32,
                            dims=0,
                        )[:32],
                        torch.roll(
                            MFCC,
                            (audioIdxRoll * -1),
                            dims=0,
                        )[:32],
                    ),
                    dim=0,
                ).view(1, 1, 64, 32)
            )
        )

    randomMoodRoll = random.randint(
        0, tracedScript.mood.size()[0] - frameCount
    )
    frames = tracedScript(
        allMFCC.view(-1, 1, 64, 32),
        torch.Tensor(mood).float().repeat(frameCount) *
        torch.roll(
            tracedScript.mood,
            (randomMoodRoll * -1),
            dims=0,
        )[:120, :].view(frameCount * 16)
    ).view(-1, 3) * 2.
    # blender has a different coordinate system than three.js
    frames = torch.cat(
        (
            torch.index_select(frames, 1, torch.LongTensor([0])),
            torch.index_select(frames, 1, torch.LongTensor([2])),
            torch.index_select(frames, 1, torch.LongTensor([1]))*-1
        ),
        dim=1
    ).detach().numpy().reshape(120, 8320, 3).flatten('C').tolist()

    return frames


@post('/mask')
def mask():
    mood = json.loads(request.forms.get('mood'))
    audio = request.forms.get('audio')
    frameCount = int(float(request.forms.get('frameCount')))
    inferred = inference(frameCount, audio, mood)

    return json.dumps(
        json.loads(
            json.dumps(inferred),
            parse_float=lambda x: round(float(x), 4)
        )
    )

    # TODO
    if frameCount != 120:
        raise  # TODO

    compressed = zfpy.compress_numpy(
        np.array(inferred, dtype=np.float32).reshape(120, 8320, 3),
        tolerance=0.0001,
        write_header=True,
    )

    return io.BytesIO(compressed)


@route('/maskIndices')
def maskIndices():
    return json.dumps(indices)


@route('/maskNeutral')
def maskNeutral():
    return json.dumps(neutral)


@route('/')
def index():
    return static_file('index.html', os.path.join(ROOT, 'static'))


@app.route('/zfpHelper.wasm')
def wasmStuff():
    return static_file('zfpHelper.wasm', os.path.join(ROOT, 'static'), 'application/wasm')


@route('/<staticFile>')
def staticStuff(staticFile):
    return static_file(staticFile, os.path.join(ROOT, 'static'))


if __name__ == '__main__':
    run(server='bjoern', host='127.0.0.1', port=2323)
