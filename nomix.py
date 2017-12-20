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
import soundfile
from bregman.suite import *


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

from actions import do_action
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
FFT_SIZE = 1024
FFW_AMOUNT = 50000
OVERLAP = 0.5

Minor = 0
Major = 1
Suspended = 2
Dominant = 3
Dimished5th = 4
Augmented5th = 5


def get_normalized_path(file_path):
    comps = file_path.split('.')
    filebase = '.'.join(comps[:-1])
    return '{}.norm.wav'.format(filebase)


def open_song_from_file(file_path):
    # if has norm cache
    norm_file_path = get_normalized_path(file_path)
    if not os.path.isfile(norm_file_path):
        # if no cache
        print('[+] normalizing')
        tmp = AudioSegment.from_file(file_path)
        # does it need normalization?
        if tmp.sample_width == SAMPLE_WIDTH:
            return tmp
        downsampled = tmp.set_sample_width(SAMPLE_WIDTH)
        downsampled.export(norm_file_path, format='wav')
    
    return AudioSegment.from_file(norm_file_path)


def calculateChordScore(chroma, chordProfile, biasToUse, N):
    _sum = 0.0
    delta = 0

    for i in range(12):
        _sum += (1 - chordProfile[i]) * (chroma[i] * chroma[i])

    delta = math.sqrt(_sum) / ((12 - N) * biasToUse)
    
    return delta


def minimumIndex(array, arrayLength):
    minValue = 100000
    minIndex = 0

    for i in range(len(array)):
        if array[i] < minValue:
            minValue = array[i]
            minIndex = i

    return minIndex


def makeChordProfiles():
    chordProfiles = np.zeros((108, 12), dtype=np.uint8)
    v1 = 1
    v2 = 1
    v3 = 1

    # set profiles matrix to all zeros
    for j in range(108):
        for t in range(12):
            chordProfiles[j, t] = 0
    
    # reset j to zero to begin creating profiles
    j = 0
    
    # major chords
    for i in range(12):
        root = i % 12
        third = (i + 4) % 12
        fifth = (i + 7) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        
        j += 1

    # minor chords
    for i in range(12):
        root = i % 12
        third = (i + 3) % 12
        fifth = (i + 7) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        
        j += 1		

    # diminished chords
    for i in range(12):
        root = i % 12
        third = (i + 3) % 12
        fifth = (i + 6) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        
        j += 1		
    
    # augmented chords
    for i in range(12):
        root = i % 12
        third = (i + 4) % 12
        fifth = (i + 8) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        
        j += 1
    
    # sus2 chords
    for i in range(12):
        root = i % 12
        third = (i + 2) % 12
        fifth = (i + 7) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        
        j += 1
    
    # sus4 chords
    for i in range(12):
        root = i % 12
        third = (i + 5) % 12
        fifth = (i + 7) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        
        j += 1	
    
    # major 7th chords
    for i in range(12):
        root = i % 12
        third = (i + 4) % 12
        fifth = (i + 7) % 12
        seventh = (i + 11) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        chordProfiles[j, seventh] = v3
        
        j += 1
    
    # minor 7th chords
    for i in range(12):
        root = i % 12
        third = (i + 3) % 12
        fifth = (i + 7) % 12
        seventh = (i + 10) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        chordProfiles[j, seventh] = v3
        
        j += 1
    
    # dominant 7th chords
    for i in range(12):
        root = i % 12
        third = (i + 4) % 12
        fifth = (i + 7) % 12
        seventh = (i + 10) % 12
        
        chordProfiles[j, root] = v1
        chordProfiles[j, third] = v2
        chordProfiles[j, fifth] = v3
        chordProfiles[j, seventh] = v3
        
        j += 1

    return chordProfiles


def classify_chromagram(chromagram, chordProfiles):
    chord = np.zeros(108, dtype=np.uint8)
    bias = 1.06
    # remove some of the 5th note energy from chromagram
    for i in range(12):
        fifth = (i+7) % 12
        chromagram[fifth] = chromagram[fifth] - (0.1 * chromagram[i])
        
        if (chromagram[fifth] < 0):
            chromagram[fifth] = 0
    
    # major chords
    for i in range(12):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], bias, 3)
    
    # minor chords
    for i in range(12):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], bias, 3)

    # diminished 5th chords
    for i in range(24, 36):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], bias, 3);

    # augmented 5th chords
    for i in range(36, 48):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], bias, 3);

    # sus2 chords
    for i in range(48, 60):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], 1, 3)
    
    # sus4 chords
    for i in range(60, 72):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], 1, 3)
    
    # major 7th chords
    for i in range(72, 84):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], 1, 4)
    
    # minor 7th chords
    for i in range(84, 96):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], bias, 4)

    # dominant 7th chords
    for i in range(96, 108):
        chord[i] = calculateChordScore(chromagram, chordProfiles[i], bias, 4)

    chordindex = minimumIndex(chord, 108)
    
    # major
    if (chordindex < 12):
        rootNote = chordindex
        quality = Major
        intervals = 0
    
    # minor
    if ((chordindex >= 12) and (chordindex < 24)):
        rootNote = chordindex - 12
        quality = Minor
        intervals = 0
    
    # diminished 5th
    if ((chordindex >= 24) and (chordindex < 36)):
        rootNote = chordindex - 24
        quality = Dimished5th
        intervals = 0
    
    # augmented 5th
    if ((chordindex >= 36) and (chordindex < 48)):
        rootNote = chordindex - 36
        quality = Augmented5th
        intervals = 0
    
    # sus2
    if ((chordindex >= 48) and (chordindex < 60)):
        rootNote = chordindex - 48
        quality = Suspended
        intervals = 2
    
    # sus4
    if ((chordindex >= 60) and (chordindex < 72)):
        rootNote = chordindex - 60
        quality = Suspended
        intervals = 4
    
    # major 7th
    if ((chordindex >= 72) and (chordindex < 84)):
        rootNote = chordindex - 72
        quality = Major
        intervals = 7
    
    # minor 7th
    if ((chordindex >= 84) and (chordindex < 96)):
        rootNote = chordindex - 84
        quality = Minor
        intervals = 7
    
    # dominant 7th
    if ((chordindex >= 96) and (chordindex < 108)):
        rootNote = chordindex - 96
        quality = Dominant
        intervals = 7

    return rootNote, quality, intervals

    
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
        nomix_set_status('[+] Initializing chromogram...')
        F = Chromagram("st.norm.wav", nfft=16384, wfft=8192, nhop=2205)
        nomix_set_status('[+] Making chord profiles...')
        chord_profiles = makeChordProfiles()
        nomix_set_status('[+] classifying chords...')
        self.notes = [classify_chromagram(c, chord_profiles) for c in F.X.T]
        nomix_set_status('[+] Initialzing audio...')
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
        cursor = (self.start_frame + (time_passed * FRAME_RATE)) / self.total_frames
        note_cursor = int(cursor * len(self.notes))
        cur_note = self.notes[note_cursor][0]
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        print('[+] NOTE:', note_cursor, NOTES[cur_note])
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
            valid = [('mp3', ''), ('wav', '')]
            file_path = nanogui.file_dialog(valid, False)
            song = open_song_from_file(file_path)
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
        self.setFixedSize((400, 0))
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

    def add_layer(self, name, song):
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
        out_l = fftutils.windowed_ifft(out_l / total_layers, fft_size=FFT_SIZE)
        # import pdb; pdb.set_trace()
        out_r = fftutils.windowed_ifft(out_r / total_layers, fft_size=FFT_SIZE)
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

    def get_spect_image(self, fft_size=FFT_SIZE, zoom=1, offset=0, dtype=np.uint8):
        print('[+] sound.sample_width', self.sound.sample_width)
        print('[+] sound.channels', self.sound.channels)
        print('[+] sound.frame_rate', self.sound.frame_rate)
        print('[+] sound.frame_count', int(self.sound.frame_count()))

        fbins, rbins = self.get_bins(fft_size)
        
        ret = fbins[:, :fft_size]
        frames_num = ret.shape[0]
        # import pdb; pdb.set_trace()
        ret = np.abs(ret)
        ret = 20 * np.log10(ret)          # scale to db
        ret = np.clip(ret, -40, 200)    # clip values
        ret = ret + 40  # to pix
        ret = (ret / 240.0) * np.iinfo(dtype).max
        ret = np.concatenate(ret.T)  # join frames and tilt
        ret = dtype(ret).reshape(fft_size, frames_num)  # reshape as bins/time
        ret = ret[::-1, :]  # flip vertically for PIL
        return ret

    def get_channels(self):
        if self.sound.channels == 1:
            for chunks in make_chunks(self.sound, int(self.sound.frame_count())):
                left = np.array(chunks.get_array_of_samples())
                return left, left

        for chunks in make_chunks(self.sound, int(self.sound.frame_count())):
            samps = chunks.get_array_of_samples()
            left = np.array(samps[::2])  # left ch
            right = np.array(samps[1::2])
        return left, right

    def get_bins(self, fft_size=FFT_SIZE):
        left, right = self.get_channels()
        self.l_bins = fftutils.windowed_fft(left, fft_size=fft_size, overlap_fac=OVERLAP)
        self.r_bins = fftutils.windowed_fft(right, fft_size=fft_size, overlap_fac=OVERLAP)
        return self.l_bins, self.r_bins

    def draw(self, ctx):
        if self.is_focused():
            ctx.BeginPath()
            ctx.Rect(self.position()[0], 
                     self.position()[1], 
                     self.size()[0] - 30, 
                     self.size()[1])
            ctx.FillColor(self.selected_color)
            ctx.Fill()
            bg = ctx.LinearGradient(self.position()[0],
                                    self.position()[1],
                                    self.size()[0] - 30,
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
    
    def set_zoom(self, zoom):
        self.canvas.zoom = zoom


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
            const float max_freq = 22050.0;
            uniform sampler2D image;
            uniform sampler2D scaletex;
            uniform vec3 in_color;
            uniform float cursor;
            uniform float gamma;
            //uniform float zoom;
            in vec2 uv;
            out vec4 color;
            float log10(float x) {
                return log(x) / log(10);
            }
            float mel(float freq) {
                return 2595.0 * log10(1.0 + freq/700.0);
            }
            float new_v(float v) {
                float freq = max_freq * v;
                return mel(freq) / mel(max_freq);
            }
            void main() {
                float inv_gamma = 1.0 / gamma;
                float scale = texture(scaletex, uv).r;
                float power = pow(texture(image, vec2(uv.x, new_v(uv.y))).r, inv_gamma);
                color = vec4(in_color.xyz * power, 1.0);
                if (abs(uv.x - cursor) <= 0.001) {
                    color = vec4(1.0 - color.x, 1.0 - color.y, 1.0 - color.z, scale);
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
        
        self.scaletextid = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.scaletextid)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        hz_freqs = (np.fft.fftfreq(FFT_SIZE) * FRAME_RATE)[:int(FFT_SIZE/2)]

        def freq_to_mel(x):
            return 2595 * np.log10(1 + x / 700.0)

        mel_freqs = freq_to_mel(hz_freqs)
        mel_scale = mel_freqs / max(mel_freqs)
        scale = mel_scale
        textureData = (scale * np.iinfo(np.uint16).max).astype(np.uint16)[::-1]

        width = 1
        height = textureData.shape[0]
        gl.glTexImage2D(gl.GL_TEXTURE_2D,
                        0,
                        gl.GL_R8,
                        width, height, 0,
                        gl.GL_RED,
                        gl.GL_UNSIGNED_SHORT,
                        textureData)

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
            textureData = layer.get_spect_image(fft_size=FFT_SIZE, dtype=np.uint16)

            height = textureData.shape[0]
            width = textureData.shape[1]
            self.shader.bind()
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.videotexid)

            gl.glTexImage2D(gl.GL_TEXTURE_2D,
                            0,
                            gl.GL_R8,
                            width, height, 0,
                            gl.GL_RED,
                            gl.GL_UNSIGNED_SHORT,
                            textureData)
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

        fac = (math.sin(current_time) + 1) / 2.0
        # fac = 1
        # mvp[0:3, 0:3] *= 0.75 * fac + 0.25
        self.shader.setUniform('modelViewProj', mvp)
        self.shader.setUniform('in_color', self.color)
        self.shader.setUniform('cursor', self.cursor)
        self.shader.setUniform('gamma', self.gamma)

        nanogui.gl.Enable(nanogui.gl.DEPTH_TEST)
        self.shader.setUniform('image', 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.scaletextid)
        self.shader.setUniform('scaletex', 1)
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
        self.gslider.setValue((1.0 / 6.0) * 2.0)
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
        self.engine = AudioEngine()
        self.pbwindow = PlaybackWindow(self, self.engine)
        self.layers_window = LayersWindow(self)
        self.layers_window.layers.set_engine(self.engine)
        self.layers_window.redraw_spec_cb = self.redraw_spect
        
        self.engine.set_layer_list(self.layers_window.layers)
        self.engine.set_audio_window(self.audio_window)
        self.engine.set_playback_window(self.pbwindow)

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
        self.performLayout()

    def add_layer(self, name, song):
        self.layers_window.layers.add_layer(name, song)

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


def print_info(song):
    print('frame_rate', song.frame_rate)
    print('frames', int(song.frame_count()))
    print('sample_width', int(song.sample_width))
    print('channels', int(song.channels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audiofiles', nargs='*')
    parser.add_argument('--norm', action='store_true', default=False, help='normalizes input audio')
    parser.add_argument('--info', '-i', action='store_true', default=False, help='prints info')
    parser.add_argument('--nogui', action='store_true', default=False, help='commandline mode')
    parser.add_argument('--play', action='store_true', default=False, help='plays on startup')
    parser.add_argument('--action', '-a')

    args = parser.parse_args()

    songs = []
    for audiofile in args.audiofiles:
        # nomix.add_layer(audiofile)
        songs.append(open_song_from_file(audiofile))
    
    # if args.norm:
    #     for song in songs:
    #         normalize(song)

    if args.info:
        for song in songs:
            print_info(song)
        return

    if args.nogui:
        return

    nanogui.init()
    nomix = NomixApp()
    i = 0
    if songs:
        for song in songs:
            nomix.add_layer(str(i), song)
            i += 1
        nomix.redraw_spect()
    nomix_set_status('[+] Initialized')
    print('[+] starting in gui mode')
    nomix.drawAll()
    nomix.setVisible(True)
    if args.play:
        nomix.engine.play(0)
    nanogui.mainloop()
    del nomix
    gc.collect()
    nanogui.shutdown()


if __name__ == '__main__':
    main()
