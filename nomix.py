# python/example1.py -- Python version of an example application that shows
# how to use the various widget classes. For a C++ implementation, see
# '../src/example1.cpp'.
#
# NanoGUI was developed by Wenzel Jakob <wenzel@inf.ethz.ch>.
# The widget drawing code is based on the NanoVG demo application
# by Mikko Mononen.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE.txt file.

import nanogui
import nanogui.gl
import math
import time
import gc
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import os
from random import shuffle
from PIL import Image

from scipy.fftpack import fft, fftfreq
import fftutils

from nanogui import Color, ColorPicker, Screen, Window, GroupLayout, BoxLayout, \
                    ToolButton, Label, Button, Widget, \
                    Popup, PopupButton, CheckBox, MessageDialog, VScrollPanel, \
                    ImagePanel, ImageView, ComboBox, ProgressBar, Slider, \
                    TextBox, ColorWheel, Graph, GridLayout, \
                    Alignment, Orientation, TabWidget, IntBox, GLShader, GLCanvas, \
                    AdvancedGridLayout

from nanogui import glfw, entypo

import pyaudio
from pydub import AudioSegment
from pydub.utils import make_chunks
import pyfftw


import argparse

pyfftw.interfaces.cache.enable()


# A simple counter, used for dynamic tab creation with TabWidget callback
counter = 1

SAMPLE_WIDTH = 2
CHANNELS = 2
FRAME_RATE = 44100


class AudioEngine:
    def __init__(self):
        self.player = None
        self.stream = None
        self.layers = []
        self.img_cache = []
        self.sound_cache = []

    def init_player(self):
        if self.player and self.stream:
            return

        self.player = pyaudio.PyAudio()

        self.stream = player.open(format=SAMPLE_WIDTH,
                                  channels=CHANNELS,
                                  rate=FRAME_RATE,
                                  output=True)

    def set_marker(self, marker):
        self.marker = marker

    def read_output(self, frames):
        print('[+] AudioEngine:: read_output')

    def get_visualized_output(self):
        print('[+] AudioEngine:: get_visualized_output')

    def on_new_layer(self, layer):
        print('[+] AudioEngine:: on_new_layer')


class LayersWindow(Window):
    def __init__(self, parent, engine, width=400):
        super(LayersWindow, self).__init__(parent, 'Layers')
        self.setLayout(GridLayout(orientation=Orientation.Vertical))

        SCROLL_BAR = 100
        self.layers_scroll = VScrollPanel(self)
        self.layers_scroll.setFixedSize((width, 600))
        self.layers = LayersList(self.layers_scroll, engine)
        self.redraw_spec_cb = None
        self.engine = engine

        right_align = Widget(self)
        right_align.setLayout(GridLayout())

        TOOLS_WIDTH = 130
        TOOLS_HEIGHT = 15

        spacer = Widget(right_align)
        spacer.setSize((width - TOOLS_WIDTH, TOOLS_HEIGHT))

        tools = Widget(right_align)
        tools.setLayout(BoxLayout(Orientation.Horizontal, spacing=6))
        tools.setFixedSize((TOOLS_WIDTH, TOOLS_HEIGHT))

        # ToolButton(tools, entypo.ICON_CONTROLLER_FAST_FORWARD)
        ToolButton(tools, entypo.ICON_COMPASS)
        tb = ToolButton(tools, entypo.ICON_ADD_TO_LIST)

        def cb():
            layer = self.layers.add_layer("New Layer")
            if self.redraw_spec_cb:
                self.redraw_spec_cb()

        tb.setCallback(cb)
        ToolButton(tools, entypo.ICON_TRASH)

        self.setPosition((960, TOOLS_HEIGHT))
        self.setSize((width, 800))
        self.setLayout(GridLayout(Orientation.Vertical, resolution=2))


def hex_to_rgb(h):
    ret = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    # print(h, ret)
    return ret


class LayersList(Widget):
    def __init__(self, parent, engine):
        super(LayersList, self).__init__(parent)
        self.setLayout(GroupLayout())
        self.layers = []
        self.shouldPerformLayout = False
        self.engine = engine
        self.colorchoices = []
        self.colorchoices.append('f5a430')
        self.colorchoices.append('f56332')
        self.colorchoices.append('f54fc9')
        self.colorchoices.append('40e43d')
        self.colorchoices.append('40dcff')
        self.colorchoices.append('49a4ff')
        self.colorchoices.append('fc7f7f')

        for i in range(0):
            self.add_layer('Layer ' + str(i+1))

        self.setFixedSize((400, 0))

    def add_layer(self, name, file_path=None):
        valid = [("mp3", ""), ("wav", "")]
        if not file_path:
            file_path = nanogui.file_dialog(valid, False)
        # result = '/Users/amiramitai/Projects/nomix/st.mp3'
        if not os.path.isfile(file_path):
            RuntimeError("Selected file isn't in place", result)

        song = AudioSegment.from_file(file_path)
        shuffle(self.colorchoices)
        h = self.colorchoices.pop()
        print(h)
        rgb = hex_to_rgb(h)
        color = Color(rgb[0], rgb[1], rgb[2], 1.0)
        sl = SoundLayer(self, name, song, color)
        self.layers.append(sl)
        self.shouldPerformLayout = True
        self.engine.on_new_layer(sl)

    def draw(self, ctx):
        if self.shouldPerformLayout:
            print('[+] performing layout!')
            self.performLayout(ctx)
            self.shouldPerformLayout = False
        super(LayersList, self).draw(ctx)


class SoundLayer(Widget):
    def __init__(self, parent, name, sound, color):
        super(SoundLayer, self).__init__(parent)
        print('[+] SoundLayer::__init__')
        self.setLayout(GridLayout(resolution=4))
        self.sound = sound
        self.cp = Button(self, "")
        self.color = color
        self.cp.setBackgroundColor(color)
        label = Label(self, name + ":", "sans-bold")
        label.setFixedSize((70, 20))
        self.cp.setFixedSize((20, 40))
        self.solomute = Widget(self)
        self.solomute.setLayout(GridLayout(resolution=3))

        self.issolo = False
        self.ismute = False
        slider = Slider(self.solomute)
        slider.setFixedSize((180, 20))

        def mute_cb():
            print("mute")

        mute = Button(self.solomute, 'M')
        mute.setCallback(mute_cb)

        def solo_cb():
            print("solo")

        solo = Button(self.solomute, 'S')
        solo.setCallback(solo_cb)
        spacer = Widget(self)
        spacer.setWidth(20)

        self.setFixedSize((400, 40))

    def get_spect_image(self, fft_size, zoom=1, offset=0):
        print("[+] sound.sample_width", self.sound.sample_width)
        print("[+] sound.channels", self.sound.channels)
        print("[+] sound.frame_rate", self.sound.frame_rate)
        print("[+] sound.frame_count", int(self.sound.frame_count()))

        for chunks in make_chunks(self.sound, int(self.sound.frame_count())):
            samps = chunks.get_array_of_samples()
            left = np.array(samps[::2])  # left ch

        fbins = fftutils.time_to_freq(left, fft_size=fft_size)
        self.fbins = fbins
        ret = fbins[:, :fft_size]
        frames_num = ret.shape[0]
        ret = np.abs(ret)
        ret = 20 * np.log10(ret)          # scale to db
        ret = np.clip(ret, -40, 200)    # clip values
        ret = ret + 40  # to pix
        ret = np.concatenate(ret.T)  # join frames and tilt
        ret = np.uint8(ret).reshape(fft_size, frames_num)  # reshape as bins/time
        ret = ret[::-1, :]  # flip vertically for PIL
        return ret


class AudioWindow(Window):
    def __init__(self, parent):
        super(AudioWindow, self).__init__(parent, "Audio View")
        self.setPosition((15, 15))
        self.setLayout(GroupLayout())
        self.canvas = AudioCanvas(self)


class AudioCanvas(GLCanvas):
    def __init__(self, parent):
        super(AudioCanvas, self).__init__(parent)
        self.rotation = [0.25, 0.5, 0.33]
        self.shader = GLShader()
        self.color = (1.0, 1.0, 1.0)
        self.marker = 0
        self.frames = 10000
        self.left_click_down = False
        self.shader.init(
            # An identifying name
            "a_simple_shader",

            # Vertex shader
            """#version 330
            uniform sampler2D texture;
            uniform mat4 modelViewProj;
            in vec2 position;
            in vec2 in_uvs;
            out vec2 uv;
            void main() {
                uv = in_uvs;
                gl_Position = modelViewProj * vec4(position, 0, 1.0);
            }""",

            # Fragment shader
            """#version 330
            uniform sampler2D image;
            uniform vec3 in_color;
            uniform float marker;
            in vec2 uv;
            out vec4 color;
            void main() {
                vec4 current = texture(image, uv);
                color = vec4(current.x * in_color.x,
                             current.y * in_color.y,
                             current.z * in_color.z, 1.0);
                if (marker > 0) {
                    if (abs(uv.x - marker) <= 0.001) {
                        color = vec4(1.0 - color.x, 1.0 - color.y, 1.0 - color.z, 1.0);
                    }
                }
            }"""
        )

        # 1##################2
        # ####################
        # 3##################4
        # Draw a cube

        indices2 = np.array(
            [[0, 2],
             [1, 3],
             [2, 0]],
            dtype=np.int32)

        positions2 = np.array(
            [[-1, -1, 1,  1],
             [-1,  1, 1, -1]],
            dtype=np.float32)

        uvs = np.array(
            [[0, 0, 1, 1],
             [1, 0, 0, 1]],
            dtype=np.float32)

        self.shader.bind()
        self.shader.uploadIndices(indices2)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        self.videotexid = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.videotexid)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        self.shader.uploadAttrib("position", positions2)
        self.shader.uploadAttrib("in_uvs", uvs)

        self.dial = 0
        self.setSize((900, 250))

    def mouseButtonEvent(self, p, button, down, modifiers):
        print("[+] AudioCanvas::mouseButtonEvent", p, button, down, modifiers)
        if (button == 0):
            self.left_click_down = down
            if down:
                self.marker = ((p[0]-15) / self.width()) * self.frames
        return super(AudioCanvas, self).mouseButtonEvent(p, button, down, modifiers)

    def mouseMotionEvent(self, p, rel, button, modifiers):
        print("[+] AudioCanvas::mouseMotionEvent", p, rel, button, modifiers)
        if self.left_click_down:
            self.marker = ((p[0]-15) / self.width()) * self.frames
        return super(AudioCanvas, self).mouseMotionEvent(p, rel, button, modifiers)

    def draw_spect(self, layers):
        print("[+] AudioCanvas::draw_spect", layers)
        for layer in layers:
            self.color = (layer.color.r, layer.color.g, layer.color.b)
            textureData = layer.get_spect_image(fft_size=2048)
            # import pdb; pdb.set_trace()
            width = 800
            im = Image.fromarray(textureData).resize((width, 128)).convert('RGB')
            textureData = np.array(im)
            height = 128
            # import pdb; pdb.set_trace()
            self.shader.bind()
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.videotexid)
            glu.gluBuild2DMipmaps(gl.GL_TEXTURE_2D, gl.GL_RGB, width, height, gl.GL_RGB,
                                  gl.GL_UNSIGNED_BYTE, textureData)
            self.drawGL()

    def drawContents(self):
        # print("[+] AudioCanvas::drawContents")
        super(AudioCanvas, self).drawContents()

    def drawGL(self):
        # print("[+] AudioCanvas::drawGL")
        self.shader.bind()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.videotexid)

        current_time = time.time()
        mvp = np.identity(4)

        # fac = (math.sin(current_time) + 1) / 2.0
        fac = 1
        mvp[0:3, 0:3] *= 0.75 * fac + 0.25

        self.shader.setUniform("modelViewProj", mvp)
        self.shader.setUniform("in_color", self.color)
        self.shader.setUniform("marker", self.marker/self.frames)

        nanogui.gl.Enable(nanogui.gl.DEPTH_TEST)
        self.shader.setUniform("image", 0)
        self.shader.drawIndexed(nanogui.gl.TRIANGLES, 0, 12)
        nanogui.gl.Disable(nanogui.gl.DEPTH_TEST)
        super(AudioCanvas, self).drawGL()

    def scrollEvent(self, event, event1):
        print('[+] AudioCanvas::scrollEvent', event, event1)


class PlaybackWindow(Window):
    def __init__(self, parent):
        super(PlaybackWindow, self).__init__(parent, "Playback")
        self.setPosition((15, 330))
        self.setLayout(GroupLayout())
        self.setFixedSize((400, 400))

        Label(self, "Location", "sans-bold")
        panel = Widget(self)
        panel.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Middle, 0, 0))
        self.frametb = TextBox(panel)
        self.frametb.setFixedSize((100, 25))
        self.frametb.setValue("0")
        self.frametb.setFontSize(14)
        self.frametb.setAlignment(TextBox.Alignment.Right)

        label = Label(panel, " ", "sans-bold")
        label.setFixedSize((15, 15))
        label = Label(panel, "/", "sans-bold")
        label.setFixedSize((20, 15))

        self.total_framestb = TextBox(panel)
        self.total_framestb.setFixedSize((100, 25))
        self.total_framestb.setValue("1000")
        self.total_framestb.setFontSize(14)
        self.total_framestb.setAlignment(TextBox.Alignment.Right)

        Label(self, "Controls", "sans-bold")
        panel = Widget(self)
        panel.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Minimum, 0, 0))
        ToolButton(panel, entypo.ICON_CONTROLLER_FAST_BACKWARD)
        ToolButton(panel, entypo.ICON_CONTROLLER_STOP)
        # ToolButton(panel, entypo.ICON_CONTROLLER_PAUS)
        ToolButton(panel, entypo.ICON_CONTROLLER_PLAY)
        ToolButton(panel, entypo.ICON_CONTROLLER_FAST_FORWARD)


class NomixApp(Screen):
    def __init__(self):
        super(NomixApp, self).__init__((1440, 900), "Nomix")

        self.engine = AudioEngine()

        self.audio_window = AudioWindow(self)
        self.layers_window = LayersWindow(self, self.engine)
        self.layers_window.redraw_spec_cb = self.redraw_spect

        self.pbwindow = PlaybackWindow(self)

        window = Window(self, "Misc. widgets")
        window.setPosition((675, 330))
        window.setLayout(GroupLayout())

        tabWidget = TabWidget(window)
        layer = tabWidget.createTab("Color Wheel")
        layer.setLayout(GroupLayout())

        Label(layer, "Color wheel widget", "sans-bold")
        ColorWheel(layer)

        layer = tabWidget.createTab("Function Graph")
        layer.setLayout(GroupLayout())
        Label(layer, "Function graph widget", "sans-bold")

        graph = Graph(layer, "Some function")
        graph.setHeader("E = 2.35e-3")
        graph.setFooter("Iteration 89")
        values = [0.5 * (0.5 * math.sin(i / 10.0) +
                         0.5 * math.cos(i / 23.0) + 1)
                  for i in range(100)]
        graph.setValues(values)
        tabWidget.setActiveTab(0)

        # Dummy tab used to represent the last tab button.
        tabWidget.createTab("+")

        def tab_cb(index):
            if index == (tabWidget.tabCount()-1):
                global counter
                # When the "+" tab has been clicked, simply add a new tab.
                tabName = "Dynamic {0}".format(counter)
                layerDyn = tabWidget.createTab(index, tabName)
                layerDyn.setLayout(GroupLayout())
                Label(layerDyn, "Function graph widget", "sans-bold")
                graphDyn = Graph(layerDyn, "Dynamic function")

                graphDyn.setHeader("E = 2.35e-3")
                graphDyn.setFooter("Iteration {0}".format(index*counter))
                valuesDyn = [0.5 * abs((0.5 * math.sin(i / 10.0 + counter)) +
                                       (0.5 * math.cos(i / 23.0 + 1 + counter)))
                             for i in range(100)]
                graphDyn.setValues(valuesDyn)
                counter += 1
                # We must invoke perform layout from the screen instance to keep everything
                # in order. This is essential when creating tabs dynamically.
                self.performLayout()
                # Ensure that the newly added header is visible on screen
                tabWidget.ensureTabVisible(index)

        tabWidget.setCallback(tab_cb)
        tabWidget.setActiveTab(0)

        window = Window(self, "Grid of small widgets")
        window.setPosition((425, 330))
        layout = GridLayout(Orientation.Horizontal, 2,
                            Alignment.Middle, 15, 5)
        layout.setColAlignment(
            [Alignment.Maximum, Alignment.Fill])
        layout.setSpacing(0, 10)
        window.setLayout(layout)

        Label(window, "Floating point :", "sans-bold")
        floatBox = TextBox(window)
        floatBox.setEditable(True)
        floatBox.setFixedSize((100, 20))
        floatBox.setValue("50")
        # floatBox.setUnits("GiB")
        floatBox.setDefaultValue("0.0")
        floatBox.setFontSize(16)
        floatBox.setFormat("[-]?[0-9]*\\.?[0-9]+")

        Label(window, "Positive integer :", "sans-bold")
        intBox = IntBox(window)
        intBox.setEditable(True)
        intBox.setFixedSize((100, 20))
        intBox.setValue(50)
        intBox.setUnits("Mhz")
        intBox.setDefaultValue("0")
        intBox.setFontSize(16)
        intBox.setFormat("[1-9][0-9]*")
        intBox.setSpinnable(True)
        intBox.setMinValue(1)
        intBox.setValueIncrement(2)

        Label(window, "Checkbox :", "sans-bold")

        cb = CheckBox(window, "Check me")
        cb.setFontSize(16)
        cb.setChecked(True)

        Label(window, "Combo box :", "sans-bold")
        cobo = ComboBox(window, ["Item 1", "Item 2", "Item 3"])
        cobo.setFontSize(16)
        cobo.setFixedSize((100, 20))

        Label(window, "Color picker :", "sans-bold")
        cp = ColorPicker(window, Color(255, 120, 0, 255))
        cp.setFixedSize((100, 20))

        def cp_final_cb(color):
            print(
                "ColorPicker Final Callback: [{0}, {1}, {2}, {3}]".format(color.r,
                                                                          color.g,
                                                                          color.b,
                                                                          color.w)
            )

        cp.setFinalCallback(cp_final_cb)

        # setup a fast callback for the color picker widget on a new window
        # for demonstrative purposes
        window = Window(self, "Color Picker Fast Callback")
        layout = GridLayout(Orientation.Horizontal, 2,
                            Alignment.Middle, 15, 5)
        layout.setColAlignment(
            [Alignment.Maximum, Alignment.Fill])
        layout.setSpacing(0, 10)
        window.setLayout(layout)
        window.setPosition((425, 515))
        window.setFixedSize((235, 300))
        Label(window, "Combined: ")
        b = Button(window, "ColorWheel", entypo.ICON_500PX)
        Label(window, "Red: ")
        redIntBox = IntBox(window)
        redIntBox.setEditable(False)
        Label(window, "Green: ")
        greenIntBox = IntBox(window)
        greenIntBox.setEditable(False)
        Label(window, "Blue: ")
        blueIntBox = IntBox(window)
        blueIntBox.setEditable(False)
        Label(window, "Alpha: ")
        alphaIntBox = IntBox(window)

        def cp_fast_cb(color):
            b.setBackgroundColor(color)
            b.setTextColor(color.contrastingColor())
            red = int(color.r * 255.0)
            redIntBox.setValue(red)
            green = int(color.g * 255.0)
            greenIntBox.setValue(green)
            blue = int(color.b * 255.0)
            blueIntBox.setValue(blue)
            alpha = int(color.w * 255.0)
            alphaIntBox.setValue(alpha)

        cp.setCallback(cp_fast_cb)

        self.performLayout()

    def add_layer(self, file_path):
        name = os.path.basename(file_path)
        self.layers_window.layers.add_layer(name, file_path=file_path)

    def redraw_spect(self):
        self.audio_window.canvas.draw_spect(self.layers_window.layers.layers)

    def draw(self, ctx):
        # self.progress.setValue(math.fmod(time.time() / 10, 1))
        super(NomixApp, self).draw(ctx)

    def drawContents(self):
        super(NomixApp, self).drawContents()

    def keyboardEvent(self, key, scancode, action, modifiers):
        if super(NomixApp, self).keyboardEvent(key, scancode,
                                               action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.setVisible(False)
            return True
        return False


if __name__ == "__main__":
    # app = NSApplication.sharedApplication()
    # delegate = MyApplicationAppDelegate.alloc().init()
    # app.setDelegate_(delegate)
    parser = argparse.ArgumentParser()
    parser.add_argument('audiofiles', nargs='*')

    args = parser.parse_args()
    nanogui.init()
    nomix = NomixApp()
    for audiofile in args.audiofiles:
        nomix.add_layer(audiofile)
    if args.audiofiles:
        nomix.redraw_spect()
    nomix.drawAll()
    nomix.setVisible(True)
    nanogui.mainloop()
    del nomix
    gc.collect()
    nanogui.shutdown()
