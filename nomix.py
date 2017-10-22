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
import _thread


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
import threading

import argparse

# pyfftw.interfaces.cache.enable()


def nomix_set_status(status):
    if not _nomix_set_status:
        return
    _nomix_set_status(status)

_nomix_set_status = None

# A simple counter, used for dynamic tab creation with TabWidget callback
counter = 1

SAMPLE_WIDTH = 2
CHANNELS = 2
FRAME_RATE = 44100
CHUNK_SIZE = 1024
FFT_SIZE = 2048
FFW_AMOUNT = 50000


def open_song_from_file(file_path):
    song = AudioSegment.from_file(file_path)
    song.set_frame_rate(FRAME_RATE)
    song.set_sample_width(SAMPLE_WIDTH)
    song.set_channels(CHANNELS)
    return song


class AudioEngine:
    def __init__(self):
        self.layer_list = None
        self.pbwindow = None
        self.audwindow = None
        self.img_cache = []
        self.sound_cache = []
        self.total_frames = 1

        # PLAYER
        self.data_ptr = None
        self.player = None
        self.stream = None
        self.is_playing = False
        self.start_time = time.time()
        self.start_frame = 0

    def init_stream(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.player:
            self.player.terminate()
            self.player = None
        self.data_ptr = self.start_frame * CHANNELS
        self.player = pyaudio.PyAudio()
        self.start_time = time.time()
        # print('[+] start_time = ', self.start_time)
        self.stream = self.player.open(format=self.player.get_format_from_width(SAMPLE_WIDTH),
                                       channels=CHANNELS,
                                       rate=FRAME_RATE,
                                       output=True,
                                       stream_callback=self.stream_callback)

    def set_gamma(self, gamma):
        self.audwindow.set_gamma(gamma)

    def stream_callback(self, sin_data, frame_count, time_info, status):
        # print('[+] NomixPlayer::stream_callback', time_info)
        if self.track is None:
            return (None, pyaudio.paComplete)

        if not self.is_playing:
            return (None, pyaudio.paComplete)
           
        dptr = self.data_ptr
        data = self.track[dptr:dptr + (frame_count * CHANNELS)]
        if len(data) < frame_count * CHANNELS:
            # print('[+] end of stream detected!')
            self.jump_to_frame(0)
            self.is_playing = False
            return (data, pyaudio.paComplete)
        self.data_ptr += len(data)
        time_passed = time.time() - self.start_time
        self.time_update_callback(self.start_frame + int(time_passed * FRAME_RATE))
        return (data, pyaudio.paContinue)
        

    def read_output(self, frames):
        print('[+] AudioEngine:: read_output')

    def get_visualized_output(self):
        print('[+] AudioEngine:: get_visualized_output')

    def _update_total_frames(self):
        print('[+] AudioEngine::_update_total_frames')
        if self.pbwindow is None:
            # print('[+] self.pbwindow is None')
            return
        if self.layer_list is None:
            # print('[+] self.layer_list is None')
            return
        framenum = self.layer_list.get_number_of_frames()
        self.total_frames = framenum
        self.pbwindow.total_framestb.setValue(str(framenum))

    def jump_to_frame(self, frame):
        # print('[+] AudioEngine::jump_to_frame', frame)
        if self.is_playing:
            self.data_ptr = frame * CHANNELS
            self.start_frame = frame
            self.start_time = time.time()
        else:
            self.start_frame = frame
        self.audwindow.set_cursor(frame/self.total_frames)
        self.time_update_callback(frame)

    def set_layer_list(self, layer_list):
        self.layer_list = layer_list
        self._update_total_frames()

    def set_playback_window(self, pbwindow):
        self.pbwindow = pbwindow
        self._update_total_frames()

    def set_audio_window(self, audwindow):
        self.audwindow = audwindow

        def callback(cursor):
            self.jump_to_frame(int(self.total_frames * cursor))

        self.audwindow.set_on_cursor_change_callback(callback)

    def time_update_callback(self, frames):
        # print('[+] AudioEngine:: time_update_callback', frames)
        self.pbwindow.frametb.setValue(str(frames))
        self.audwindow.set_cursor(frames/self.total_frames)

    def on_new_layer(self, layer):
        print('[+] AudioEngine:: on_new_layer', layer)
        self._update_total_frames()
        
    def pause(self):
        self.is_playing = False

    def get_total_frames(self):
        return self.total_frames

    def get_cursor(self):
        return int(self.audwindow.canvas.cursor * self.total_frames)

    def play(self, start_frame=None):
            
        self.last_play_time = time.time()
        if start_frame is None:
            self.start_frame = self.get_cursor()
        else:
            self.start_frame = start_frame
        nomix_set_status('[+] Getting mixdown.. please wait..')
        
        def _init_with_mixdown():
            self.track = self.layer_list.get_mixdown()
            nomix_set_status('[+] Initializing stream..')
            self.is_playing = True
            self.init_stream()
            nomix_set_status('[+] Done.')
        
        args = ()
        _thread.start_new_thread(_init_with_mixdown, args)


class StatusWindow(Window):
    def __init__(self, parent):
        global nomix_set_status
        super(StatusWindow, self).__init__(parent, 'Status')
        # self.setLayout(GridLayout(orientation=Orientation.Vertical))
        self.setSize((1410, 90))
        self.setPosition((15, 750))
        self.label = Label(self, '', 'sans-bold')
        self.label.setFixedSize((1400, 50))
        self.label.setPosition((15, 50))
        nomix_set_status = self.set_status

    def set_status(self, status):
        self.label.setCaption(status)


class LayersWindow(Window):
    def __init__(self, parent, width=400):
        super(LayersWindow, self).__init__(parent, 'Layers')
        self.setLayout(GridLayout(orientation=Orientation.Vertical))

        SCROLL_BAR = 100
        self.layers_scroll = VScrollPanel(self)
        self.layers_scroll.setFixedSize((width, 600))
        self.layers = LayersList(self.layers_scroll)
        self.redraw_spec_cb = None

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
            layer = self.layers.add_layer('New Layer')
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
    def __init__(self, parent):
        super(LayersList, self).__init__(parent)
        self.setLayout(GroupLayout())
        self.layers = []
        self.shouldPerformLayout = False
        self.engine = None
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

        self.cache = None

    def get_mixdown(self):
        if self.cache is not None:
            return self.cache

        self.cache = self.flatten()
        return self.cache

    def set_engine(self, engine):
        self.engine = engine
        
    def get_focused(self):
        for layer in self.layers:
            if layer.is_focused():
                return LayersList

        return None

    def blur_all(self):
        for layer in self.layers:
            layer.set_focus(False)

    def add_layer(self, name, file_path=None):
        valid = [('mp3', ''), ('wav', '')]
        if not file_path:
            file_path = nanogui.file_dialog(valid, False)
        # result = '/Users/amiramitai/Projects/nomix/st.mp3'
        if not os.path.isfile(file_path):
            RuntimeError('Selected file isnt in place', result)

        song = open_song_from_file(file_path)
        shuffle(self.colorchoices)
        h = self.colorchoices.pop()
        # print(h)
        rgb = hex_to_rgb(h)
        color = Color(rgb[0], rgb[1], rgb[2], 1.0)
        sl = SoundLayer(self, name, song, color)
        self.layers.append(sl)
        self.shouldPerformLayout = True
        if self.engine:
            self.engine.on_new_layer(sl)

    def draw(self, ctx):
        if self.shouldPerformLayout:
            print('[+] performing layout!')
            self.performLayout(ctx)
            self.shouldPerformLayout = False
        super(LayersList, self).draw(ctx)

    def get_number_of_frames(self):
        frames = 0
        for layer in self.layers:
            l, r = layer.get_channels()
            frames = max(frames, len(l))
        return frames

    def flatten(self, fft_size=FFT_SIZE):
        max_layer = 0
        for layer in self.layers:
            l, r = layer.get_bins(fft_size)
            max_layer = max(max_layer, l.shape[0])

        out_l = np.zeros((max_layer, fft_size*2), dtype=np.complex128)
        out_r = np.zeros((max_layer, fft_size*2), dtype=np.complex128)
        total_layers = 0
        for layer in self.layers:
            total_layers += 1
            l, r = layer.get_bins(fft_size)
            if l.shape[0] < max_layer:
                s = max_layer - l.shape[0]
                to_pad = np.zeros((s, fft_size*2), dtype=np.complex128)
                l = np.vstack((l, to_pad))
                r = np.vstack((r, to_pad))
            out_l += l
            out_r += r
        out_l = fftutils.freq_to_time(out_l / total_layers, fft_size=FFT_SIZE)
        # import pdb; pdb.set_trace()
        out_r = fftutils.freq_to_time(out_r / total_layers, fft_size=FFT_SIZE)
        return np.vstack((out_l, out_r)).T.flatten()
        # return out_l
        # return self.layers[0].get_channels()

    def mouseButtonEvent(self, p, button, down, modifiers):
        ret = super(LayersList, self).mouseButtonEvent(p, button, down, modifiers)
        if ret:
            return ret
        self.blur_all()
        return False
        

class SoundLayer(Widget):
    def __init__(self, parent, name, sound, color):
        super(SoundLayer, self).__init__(parent)
        print('[+] SoundLayer::__init__')
        self.setLayout(GridLayout(resolution=4))
        self.sound = sound
        self.cp = Button(self, '')
        self.color = color
        self.cp.setBackgroundColor(color)
        label = Label(self, name + ':', 'sans-bold')
        label.setFixedSize((70, 20))
        self.cp.setFixedSize((20, 40))
        self.solomute = Widget(self)
        self.solomute.setLayout(GridLayout(resolution=3))
        self._parent = parent
        self._focused = False

        self.issolo = False
        self.ismute = False
        self.selected_color = Color(0x3e, 0x3e, 0x3f, 0xff)
        slider = Slider(self.solomute)
        slider.setFixedSize((180, 20))

        def mute_cb():
            print('mute')

        mute = Button(self.solomute, 'M')
        mute.setCallback(mute_cb)

        def solo_cb():
            print('solo')

        solo = Button(self.solomute, 'S')
        solo.setCallback(solo_cb)
        spacer = Widget(self)
        spacer.setWidth(20)

        self.setFixedSize((400, 40))

    def get_spect_image(self, fft_size=FFT_SIZE, zoom=1, offset=0):
        print('[+] sound.sample_width', self.sound.sample_width)
        print('[+] sound.channels', self.sound.channels)
        print('[+] sound.frame_rate', self.sound.frame_rate)
        print('[+] sound.frame_count', int(self.sound.frame_count()))

        fbins, rbins = self.get_bins(fft_size)
        
        ret = fbins[:, :fft_size]
        frames_num = ret.shape[0]
        ret = np.abs(ret)
        ret = 20 * np.log10(ret)          # scale to db
        ret = np.clip(ret, -40, 200)    # clip values
        ret = ret + 40  # to pix
        ret = (ret / 240.0) * 255.0
        ret = np.concatenate(ret.T)  # join frames and tilt
        ret = np.uint8(ret).reshape(fft_size, frames_num)  # reshape as bins/time
        ret = ret[::-1, :]  # flip vertically for PIL
        return ret

    def get_channels(self):
        for chunks in make_chunks(self.sound, int(self.sound.frame_count())):
            samps = chunks.get_array_of_samples()
            left = np.array(samps[::2])  # left ch
            right = np.array(samps[1::2])
        return left, right

    def get_bins(self, fft_size=FFT_SIZE):
        left, right = self.get_channels()
        self.l_bins = fftutils.time_to_freq(left, fft_size=fft_size)
        self.r_bins = fftutils.time_to_freq(right, fft_size=fft_size)
        return self.l_bins, self.r_bins

    def draw(self, ctx):
        if self.is_focused():
            ctx.BeginPath()
            ctx.Rect(self.position()[0], self.position()[1], self.size()[0], self.size()[1])
            ctx.FillColor(self.selected_color)
            ctx.Fill()
            bg = ctx.LinearGradient(self.position()[0],
                                    self.position()[1],
                                    self.size()[0],
                                    self.size()[1],
                                    self.selected_color,
                                    self.selected_color)
            ctx.FillPaint(bg)
            ctx.Fill()
        return super(SoundLayer, self).draw(ctx)

    def is_focused(self):
        return self._focused

    def set_focus(self, val):
        if val:
            self._parent.blur_all()
        self._focused = val

    def mouseButtonEvent(self, p, button, down, modifiers):
        ret = super(SoundLayer, self).mouseButtonEvent(p, button, down, modifiers)
        if ret:
            return ret
        if (button == 0):
            if down:
                self.set_focus(True)
            return True
        return False


class AudioWindow(Window):
    def __init__(self, parent):
        super(AudioWindow, self).__init__(parent, 'Audio View')
        self.setPosition((15, 15))
        self.setLayout(GroupLayout())
        self.canvas = AudioCanvas(self)

    def set_cursor(self, cursor):
        self.canvas.cursor = cursor

    def set_on_cursor_change_callback(self, cb):
        self.canvas.on_cursor_change_cb = cb

    def set_gamma(self, gamma):
        self.canvas.gamma = gamma


class AudioCanvas(GLCanvas):
    def __init__(self, parent):
        super(AudioCanvas, self).__init__(parent)
        self.rotation = [0.25, 0.5, 0.33]
        self.shader = GLShader()
        self.color = (1.0, 1.0, 1.0)
        self.cursor = 0
        self.gamma = 2.0
        self.frames = 10000
        self.left_click_down = False
        self.on_cursor_change_cb = None
        self.shader.init(
            # An identifying name
            'a_simple_shader',

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
            uniform float cursor;
            uniform float gamma;
            in vec2 uv;
            out vec4 color;
            void main() {
                float inv_gamma = 1.0 / gamma;
                vec4 current = texture(image, uv);
                color = vec4(pow(current.x * in_color.x, inv_gamma),
                             pow(current.y * in_color.y, inv_gamma),
                             pow(current.z * in_color.z, inv_gamma), 1.0);
                if (abs(uv.x - cursor) <= 0.001) {
                    color = vec4(1.0 - color.x, 1.0 - color.y, 1.0 - color.z, 1.0);
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

        self.shader.uploadAttrib('position', positions2)
        self.shader.uploadAttrib('in_uvs', uvs)

        self.dial = 0
        self.setSize((900, 250))
        self.cursor = 0.0

    def mouseButtonEvent(self, p, button, down, modifiers):
        # print('[+] AudioCanvas::mouseButtonEvent', p, button, down, modifiers)
        if (button == 0):
            self.left_click_down = down
            if down:
                self.on_cursor_change((p[0]-15) / self.width())
        return super(AudioCanvas, self).mouseButtonEvent(p, button, down, modifiers)

    def mouseMotionEvent(self, p, rel, button, modifiers):
        # print('[+] AudioCanvas::mouseMotionEvent', p, rel, button, modifiers)
        if self.left_click_down:
            self.on_cursor_change((p[0]-15) / self.width())
        return super(AudioCanvas, self).mouseMotionEvent(p, rel, button, modifiers)

    def on_cursor_change(self, cursor):
        if self.on_cursor_change_cb:
            self.on_cursor_change_cb(cursor)

    def draw_spect(self, layers):
        print('[+] AudioCanvas::draw_spect', layers)
        for layer in layers:
            self.color = (layer.color.r, layer.color.g, layer.color.b)
            textureData = layer.get_spect_image(fft_size=FFT_SIZE)
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
        # print('[+] AudioCanvas::drawContents')
        super(AudioCanvas, self).drawContents()

    def drawGL(self):
        # print('[+] AudioCanvas::drawGL')
        self.shader.bind()

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.videotexid)

        current_time = time.time()
        mvp = np.identity(4)

        # fac = (math.sin(current_time) + 1) / 2.0
        fac = 1
        mvp[0:3, 0:3] *= 0.75 * fac + 0.25

        self.shader.setUniform('modelViewProj', mvp)
        self.shader.setUniform('in_color', self.color)
        self.shader.setUniform('cursor', self.cursor)
        self.shader.setUniform('gamma', self.gamma)

        nanogui.gl.Enable(nanogui.gl.DEPTH_TEST)
        self.shader.setUniform('image', 0)
        self.shader.drawIndexed(nanogui.gl.TRIANGLES, 0, 12)
        nanogui.gl.Disable(nanogui.gl.DEPTH_TEST)
        super(AudioCanvas, self).drawGL()

    def scrollEvent(self, event, event1):
        print('[+] AudioCanvas::scrollEvent', event, event1)


class PlaybackWindow(Window):
    def __init__(self, parent, engine):
        super(PlaybackWindow, self).__init__(parent, 'Playback')
        self.setPosition((15, 330))
        self.setLayout(GroupLayout())
        self.setFixedSize((400, 400))
        self.engine = engine

        Label(self, 'Location', 'sans-bold')
        panel = Widget(self)
        panel.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Middle, 0, 0))
        self.frametb = TextBox(panel)
        self.frametb.setFixedSize((100, 25))
        self.frametb.setValue('0')
        self.frametb.setFontSize(14)
        self.frametb.setAlignment(TextBox.Alignment.Right)
        self.frametb.setEditable(True)

        label = Label(panel, ' ', 'sans-bold')
        label.setFixedSize((15, 15))
        label = Label(panel, '/', 'sans-bold')
        label.setFixedSize((20, 15))

        self.total_framestb = TextBox(panel)
        self.total_framestb.setFixedSize((100, 25))
        self.total_framestb.setValue('1000')
        self.total_framestb.setFontSize(14)
        self.total_framestb.setAlignment(TextBox.Alignment.Right)
        self.total_framestb.setEditable(True)

        Label(self, 'Controls', 'sans-bold')
        panel = Widget(self)
        panel.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Minimum, 0, 0))
        self.fbw_button = ToolButton(panel, entypo.ICON_CONTROLLER_FAST_BACKWARD)
        self.fbw_button.setFlags(Button.Flags.NormalButton)
        self.fbw_button.setCallback(lambda: self._fbw_cb())

        self.stop_button = ToolButton(panel, entypo.ICON_CONTROLLER_STOP)
        self.stop_button.setFlags(Button.Flags.NormalButton)
        self.stop_button.setCallback(lambda: self._stop_cb())
        
        self.play_button = ToolButton(panel, entypo.ICON_CONTROLLER_PLAY)
        self.play_button.setFlags(Button.Flags.NormalButton)
        self.play_button.setCallback(lambda: self._play_cb())
        
        self.ffw_button = ToolButton(panel, entypo.ICON_CONTROLLER_FAST_FORWARD)
        self.ffw_button.setFlags(Button.Flags.NormalButton)
        self.ffw_button.setCallback(lambda: self._ffw_cb())

        # Label(self, 'View Params', 'sans-bold')
        Label(self, 'Gamma', 'sans-bold')
        # panel = Widget(self)
        sub_panel = Widget(self)
        sub_panel.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Minimum, 0, 0))
        self.gslider = Slider(sub_panel)
        self.gslider.setFixedSize((180, 20))
        self.gslider.setValue(0.5)
        self.gtb = TextBox(sub_panel)
        self.gtb.setFixedSize((100, 25))
        self.gtb.setValue('2.0')

        def cb(value):
            # print (value)
            self.gtb.setValue('{:.2f}'.format(value * 6.0))
            self.engine.set_gamma(value * 6.0)

        self.gslider.setCallback(cb)
        
   
    def _fbw_cb(self):
        self.engine.jump_to_frame(max(self.engine.get_cursor() - FFW_AMOUNT, 0))

    def _ffw_cb(self):
        self.engine.jump_to_frame(min(self.engine.get_cursor() + FFW_AMOUNT, 
                                      self.engine.get_total_frames()))

    def _play_cb(self):
        print('Play/Pause')
        if self.engine.is_playing:
            self.engine.pause()
            self.play_button.setIcon(entypo.ICON_CONTROLLER_PLAY)
        else:
            self.engine.play()
            self.play_button.setIcon(entypo.ICON_CONTROLLER_PAUS)

    def _stop_cb(self):
        print('Stop')
        self.engine.pause()
        self.engine.jump_to_frame(0)
        self.play_button.setIcon(entypo.ICON_CONTROLLER_PLAY)



class NomixApp(Screen):
    def __init__(self):
        super(NomixApp, self).__init__((1440, 900), 'Nomix')

        self.status = StatusWindow(self)
        self.audio_window = AudioWindow(self)
        self.layers_window = LayersWindow(self)
        self.engine = AudioEngine()
        self.layers_window.layers.set_engine(self.engine)
        self.layers_window.redraw_spec_cb = self.redraw_spect
        
        self.pbwindow = PlaybackWindow(self, self.engine)
        
        self.engine.set_layer_list(self.layers_window.layers)
        self.engine.set_playback_window(self.pbwindow)
        self.engine.set_audio_window(self.audio_window)

        window = Window(self, 'Misc. widgets')
        window.setPosition((675, 330))
        window.setLayout(GroupLayout())

        tabWidget = TabWidget(window)
        layer = tabWidget.createTab('Color Wheel')
        layer.setLayout(GroupLayout())

        Label(layer, 'Color wheel widget', 'sans-bold')
        ColorWheel(layer)

        layer = tabWidget.createTab('Function Graph')
        layer.setLayout(GroupLayout())
        Label(layer, 'Function graph widget', 'sans-bold')

        graph = Graph(layer, 'Some function')
        graph.setHeader('E = 2.35e-3')
        graph.setFooter('Iteration 89')
        values = [0.5 * (0.5 * math.sin(i / 10.0) +
                         0.5 * math.cos(i / 23.0) + 1)
                  for i in range(100)]
        graph.setValues(values)
        tabWidget.setActiveTab(0)

        # Dummy tab used to represent the last tab button.
        tabWidget.createTab('+')

        def tab_cb(index):
            if index == (tabWidget.tabCount()-1):
                global counter
                # When the '+' tab has been clicked, simply add a new tab.
                tabName = 'Dynamic {0}'.format(counter)
                layerDyn = tabWidget.createTab(index, tabName)
                layerDyn.setLayout(GroupLayout())
                Label(layerDyn, 'Function graph widget', 'sans-bold')
                graphDyn = Graph(layerDyn, 'Dynamic function')

                graphDyn.setHeader('E = 2.35e-3')
                graphDyn.setFooter('Iteration {0}'.format(index*counter))
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

        window = Window(self, 'Grid of small widgets')
        window.setPosition((425, 330))
        layout = GridLayout(Orientation.Horizontal, 2,
                            Alignment.Middle, 15, 5)
        layout.setColAlignment(
            [Alignment.Maximum, Alignment.Fill])
        layout.setSpacing(0, 10)
        window.setLayout(layout)

        Label(window, 'Floating point :', 'sans-bold')
        floatBox = TextBox(window)
        floatBox.setEditable(True)
        floatBox.setFixedSize((100, 20))
        floatBox.setValue('50')
        # floatBox.setUnits('GiB')
        floatBox.setDefaultValue('0.0')
        floatBox.setFontSize(16)
        floatBox.setFormat('[-]?[0-9]*\\.?[0-9]+')

        Label(window, 'Positive integer :', 'sans-bold')
        intBox = IntBox(window)
        intBox.setEditable(True)
        intBox.setFixedSize((100, 20))
        intBox.setValue(50)
        intBox.setUnits('Mhz')
        intBox.setDefaultValue('0')
        intBox.setFontSize(16)
        intBox.setFormat('[1-9][0-9]*')
        intBox.setSpinnable(True)
        intBox.setMinValue(1)
        intBox.setValueIncrement(2)

        Label(window, 'Checkbox :', 'sans-bold')

        cb = CheckBox(window, 'Check me')
        cb.setFontSize(16)
        cb.setChecked(True)

        Label(window, 'Combo box :', 'sans-bold')
        cobo = ComboBox(window, ['Item 1', 'Item 2', 'Item 3'])
        cobo.setFontSize(16)
        cobo.setFixedSize((100, 20))

        Label(window, 'Color picker :', 'sans-bold')
        cp = ColorPicker(window, Color(255, 120, 0, 255))
        cp.setFixedSize((100, 20))

        def cp_final_cb(color):
            print(
                'ColorPicker Final Callback: [{0}, {1}, {2}, {3}]'.format(color.r,
                                                                          color.g,
                                                                          color.b,
                                                                          color.w)
            )

        cp.setFinalCallback(cp_final_cb)

        # setup a fast callback for the color picker widget on a new window
        # for demonstrative purposes
        # window = Window(self, 'Color Picker Fast Callback')
        # layout = GridLayout(Orientation.Horizontal, 2,
        #                     Alignment.Middle, 15, 5)
        # layout.setColAlignment(
        #     [Alignment.Maximum, Alignment.Fill])
        # layout.setSpacing(0, 10)
        # window.setLayout(layout)
        # window.setPosition((425, 515))
        # window.setFixedSize((235, 300))
        # Label(window, 'Combined: ')
        # b = Button(window, 'ColorWheel', entypo.ICON_500PX)
        # Label(window, 'Red: ')
        # redIntBox = IntBox(window)
        # redIntBox.setEditable(False)
        # Label(window, 'Green: ')
        # greenIntBox = IntBox(window)
        # greenIntBox.setEditable(False)
        # Label(window, 'Blue: ')
        # blueIntBox = IntBox(window)
        # blueIntBox.setEditable(False)
        # Label(window, 'Alpha: ')
        # alphaIntBox = IntBox(window)

        # def cp_fast_cb(color):
        #     b.setBackgroundColor(color)
        #     b.setTextColor(color.contrastingColor())
        #     red = int(color.r * 255.0)
        #     redIntBox.setValue(red)
        #     green = int(color.g * 255.0)
        #     greenIntBox.setValue(green)
        #     blue = int(color.b * 255.0)
        #     blueIntBox.setValue(blue)
        #     alpha = int(color.w * 255.0)
        #     alphaIntBox.setValue(alpha)

        # cp.setCallback(cp_fast_cb)

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
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.pbwindow._play_cb()
            return True
        return False


if __name__ == '__main__':
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
    # nomix.engine.play()
    nomix_set_status('[+] Initialized')
    nomix.drawAll()
    nomix.setVisible(True)
    nanogui.mainloop()
    del nomix
    gc.collect()
    nanogui.shutdown()
