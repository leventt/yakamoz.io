import time
import os
import random
import math
import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
from pyrr import Quaternion
from pyrr import Vector3
from pyrr import vector
from neutral import neutral
from indices import indices
from importlib.machinery import SourceFileLoader
from main import ROOT
from main import tracedScriptPath
import torch


indices = (np.array(indices) - 1)
neutral = np.array(neutral)


surat = SourceFileLoader(
    'surat',
    os.path.join(ROOT, 'surat/surat.py')
).load_module()
surat.DEVICE = torch.device('cpu')


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class Camera():
    def __init__(self, ratio):
        self.fov = 18.5
        self.aspect = ratio
        self.near = .1
        self.far = 10000.
        self.theta = np.pi / 2.
        self.phi = np.pi / 2.
        self.radius = 8.
        self.upsign = 1.
        self.target = np.array([0., 0., 0.], np.float32)
        self.orbit(math.radians(0), math.radians(0))
        self.navigating = False
        self.init = True

        self.LMB = False
        self.MMB = False
        self.RMB = False
        self.MOD = False
        self.MOD1 = False
    
    def cameraPosition(self):
        height = math.cos(self.phi) * self.radius
        distance = math.sin(self.phi) * self.radius

        return np.array([
            distance * math.cos(self.theta),
            height,
            distance * math.sin(self.theta)
        ]) + self.target

    def orbit(self, theta, phi):
        self.phi += phi

        twoPi = np.pi * 2.
        while self.phi > twoPi:
            self.phi -= twoPi
        while self.phi < -twoPi:
            self.phi += twoPi

        if (self.phi < np.pi and self.phi > 0.0):
            self.upsign = 1.0
        elif (self.phi < -np.pi and self.phi > -2 * np.pi):
            self.upsign = 1.0
        else:
            self.upsign = -1.0

        self.theta += self.upsign * theta

    def pan(self, dx, dy):
        direction = normalize([self.target - self.cameraPosition()])[0]
        right = np.cross(direction, [0., self.upsign, 0.])
        up = np.cross(right, direction)

        self.target += right * dx
        self.target += up * dy

    def zoom(self, distance):
        if self.radius - distance > 0:
            self.radius -= distance

    def projectionMatrix(self):
        self.navigating = False

        return Matrix44.perspective_projection(
            self.fov,
            self.aspect,
            self.near,
            self.far
        )

    def viewatrix(self):
        self.navigating = False

        direction = normalize([self.target - self.cameraPosition()])[0]
        right = np.cross(direction, [0., self.upsign, 0.])
        up = np.cross(right, direction)
        eye = self.cameraPosition()

        return Matrix44.look_at(eye, self.target, up)

    def mouseDragEvent(self, dx, dy):
        self.navigating = True
        self.init = False

        if self.LMB and not self.MOD and not self.MOD1:
            self.orbit(
                dx * .02,
                -dy * .02
            )
        elif self.MMB or (self.LMB and self.MOD1):
            self.pan(
                -dx * self.radius * .001,
                dy * self.radius * .001
            )
        elif self.RMB or (self.LMB and self.MOD):
            if abs(dx) > abs(dy):
                self.zoom(-dx * self.radius * .01)
            else:
                self.zoom(dy * self.radius * .01)

    def mouseScrollEvent(self, delta):
        self.navigating = True

        self.init = False
        self.zoom(delta * (self.radius / 1000.))


class PreviewWindow(mglw.WindowConfig):
    gl_version = (4, 1)
    title = "yakamoz.io"
    window_size = (864, 486)
    aspect_ratio = 16 / 9
    resizable = False
    samples = 4

    resource_dir = os.path.normpath(os.path.dirname(__file__))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        validationData = surat.Data(
            validationAudioPath=os.path.join(
                ROOT,
                'data',
                'validation.wav'
            )
        )

        if not os.path.exists(tracedScriptPath):
            print('torch script not found')
        else:
            tracedScript = torch.jit.load(tracedScriptPath)
            tracedScript.eval()

        self.frame = 0
        self.frameCount = len(validationData)
        inputValues = torch.Tensor([])
        for _, inputValue, _ in validationData:
            inputValues = torch.cat(
                (
                    inputValues,
                    inputValue
                ),
                dim=1
            )
        inputValues = inputValues.view(self.frameCount, 1, 64, 32)

        randomMoodRoll = random.randint(
            0, tracedScript.mood.size()[0] - self.frameCount
        )
        frames = tracedScript(
            inputValues,
            torch.roll(
                tracedScript.mood,
                (randomMoodRoll * -1),
                dims=0,
            )[:self.frameCount, :].view(self.frameCount * 16)
        ).view(-1, 3) * 2.
        # blender has a different coordinate system
        self.frames = torch.cat(
            (
                torch.index_select(frames, 1, torch.LongTensor([0])),
                torch.index_select(frames, 1, torch.LongTensor([2])),
                torch.index_select(frames, 1, torch.LongTensor([1]))*-1
            ),
            dim=1
        ).detach().numpy().reshape(self.frameCount, 8320 * 3).astype(np.float32)

        self.prog = self.ctx.program(
            vertex_shader='''
            #version 410 core
            in vec3 position;
            uniform mat4 view;
            uniform mat4 projection;
            out vec3 vPosition;
            void main()
            {
                vPosition = position;
                gl_Position = projection * view * vec4(vPosition, 1.);
            }
            ''',
            geometry_shader='''
            #version 410 core
            layout(triangles) in;
            layout(triangle_strip, max_vertices = 3) out;
            in vec3 vPosition[];
            uniform mat4 view;
            out vec3 gNormal;
            out vec3 gPosition;
            void main()
            {
                vec3 flatNormal = cross(
                    vPosition[1] - vPosition[0],
                    vPosition[2] - vPosition[0]
                );
                gNormal = normalize(transpose(inverse(mat3(view))) * flatNormal);
                gPosition = vPosition[0];
                gl_Position = gl_in[0].gl_Position; EmitVertex();
                gPosition = vPosition[1];
                gl_Position = gl_in[1].gl_Position; EmitVertex();
                gPosition = vPosition[2];
                gl_Position = gl_in[2].gl_Position; EmitVertex();
                EndPrimitive();
            }
            ''',
            fragment_shader='''
            #version 410 core
            in vec3 gPosition;
            in vec3 gNormal;
            uniform mat4 view;
            uniform sampler2D matcap;
            out vec4 fragColor;
            void main()
            {
                vec3 r = reflect(normalize(view * vec4(gPosition, 1.)).xyz, gNormal);
                float m = 2. * sqrt(pow(r.x, 2.) + pow(r.y, 2.) + pow(r.z + 1., 2.));
                vec2 matcapUV = r.xy / m + .5;
                vec3 color = texture(matcap, matcapUV.xy).xyz;
                fragColor = vec4(color, 1.);
            }
            '''
        )

        self.camera = Camera(self.aspect_ratio)
        self.projection = self.prog['projection']
        self.view = self.prog['view']
        self.vbo = self.ctx.buffer(neutral.astype(np.float32).tobytes())
        indexBuffer = self.ctx.buffer(indices.astype(np.uint32).tobytes())
        self.vao = self.ctx.vertex_array(
            program=self.prog,
            content=[
                (self.vbo, '3f', 'position')
            ],
            index_buffer=indexBuffer,
        )
        self.matcap = self.load_texture_2d('matcap.png')

    def key_event(self, key, action, modifiers):
        if key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_PRESS:
            self.camera.MOD = True
        elif key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_RELEASE:
            self.camera.MOD = False

        if key == self.wnd.keys.X and action == self.wnd.keys.ACTION_PRESS:
            self.camera.MOD1 = True
        elif key == self.wnd.keys.X and action == self.wnd.keys.ACTION_RELEASE:
            self.camera.MOD1 = False
    
    def mouse_drag_event(self, x, y, dx, dy):
        self.camera.LMB = False
        self.camera.MMB = False
        self.camera.RMB = False

        if self.wnd.mouse_states.left:
            self.camera.LMB = True
        elif self.wnd.mouse_states.middle:
            self.camera.MMB = True
        elif self.wnd.mouse_states.right:
            self.camera.RMB = True

        self.camera.mouseDragEvent(dx, dy)

    def mouse_scroll_event(self, dx, dy):
        self.camera.mouseScrollEvent((dx ** 2. + dy ** 2.) ** .5)

    def render(self, time, frameTime):
        self.ctx.clear(.18, .18, .18)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.frame = int(time * 29.97)
        self.frame = self.frame % self.frameCount
        self.vbo.write(self.frames[self.frame].tobytes())

        if self.camera.navigating or self.camera.init:
            self.projection.write(
                (
                    self.camera.projectionMatrix()
                ).astype(np.float32).tobytes()
            )
            self.view.write(
                (
                    self.camera.viewatrix()
                ).astype(np.float32).tobytes()
            )

        self.matcap.use()
        self.vao.render(moderngl.TRIANGLES)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


if __name__ == '__main__':
    PreviewWindow.run()
