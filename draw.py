# Kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.vector import Vector
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.utils import get_color_from_hex
Window.clearcolor = (1, 1, 1, 1)
Window.top = 260
Window.left = 600
from kivy.metrics import dp
import time
import math
import load_tensor
import separate
import conv_toy
import lstm_toy
import generator
import draw_parser
import ink_parser as ip
import numpy as np
import baseline_symmetry
import stars
class MainLayout(BoxLayout):
    pass

class Toolbar(BoxLayout):
    pass

class Canvas(Widget):
    def __init__(self):
        super(Canvas, self).__init__()
        self.strokes = []
        self.points = []

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.mode == 0:
                # Begin new stroke
                self.strokes.append([touch.x,touch.y])
                self.timer = time.time()
                with self.canvas:
                    Color(0.6,0.6,0.6)
                    touch.ud['line'] = Line(points=(touch.x, touch.y), width=dp(2))
            else:
                print("%s,%s"% (touch.x,touch.y))

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            if self.mode == 0:
                # Add new point
                elapsed = time.time() - self.timer

                last_x = self.strokes[-1][-2]
                last_y = self.strokes[-1][-1]

                delta = math.sqrt(math.pow(touch.x - last_x,2) + math.pow(touch.y - last_y,2))
                if delta > 40:
                    self.timer = time.time()
                    self.strokes[-1] += [touch.x, touch.y]
                    touch.ud['line'].points += [touch.x, touch.y]

    """Get the stroke list"""
    def get_strokes(self):
        return self.strokes

    """Set the stroke list"""
    def set_strokes(self, strokes, colors = []):
        self.canvas.clear()
        self.strokes = strokes
        with self.canvas:
            for stroke in strokes:
                Color(0.6,0.6,0.6)
                Line(points=stroke, width=dp(2))

    """Set the colors of the current strokes"""
    def set_colors(self, colors):
        self.canvas.clear()
        with self.canvas:
            for (stroke,color) in zip(self.strokes,colors):
                Color(*color)
                Line(points=stroke, width=dp(2))

class DrawApp(App):
    def build(self):
        root = FloatLayout()
        parent = MainLayout()
        root.add_widget(parent)
        self.painter = Canvas()
        self.index = 0
        self.painter.mode = 0

        # Main toolbar
        toolbar = Toolbar()

        get_button = Button(text='Print', size = (80,60), size_hint=(None, 1))
        get_button.bind(on_release=self.print_strokes)

        set_button = Button(text='Set', size = (80,60), size_hint=(None, 1))
        set_button.bind(on_release=self.set_strokes)

        clear_button = Button(text='Clear', size = (80,60), size_hint=(None, 1))
        clear_button.bind(on_release=self.clear_strokes)

        self.filename_input = TextInput(text='dataset/reflected-eval.tfrecords', size_hint=(1, 1))
        self.index_input = TextInput(text='0', size = (60,60), size_hint=(None, 1))

        # Model toolbar
        model_toolbar = Toolbar()

        sep_button = Button(text='Separate', size = (160,60), size_hint=(None, 1))
        sep_button.bind(on_release=self.sep_strokes)

        plot_button = Button(text='Plot', size = (160,60), size_hint=(None, 1))
        plot_button.bind(on_release=self.plot)

        gen_button = Button(text='Generate', size = (160,60), size_hint=(None, 1))
        gen_button.bind(on_release=self.generate)

        plt_button = Button(text='Plot', size = (160,60), size_hint=(None, 1))
        plt_button.bind(on_release=self.plot_inks)

        sym_button = Button(text='Sym', size = (160,60), size_hint=(None, 1))
        sym_button.bind(on_release=self.sym)

        base_button = Button(text='Baseline', size = (160,60), size_hint=(None, 1))
        base_button.bind(on_release=self.baseline)

        print_button = Button(text='Print', size = (160,60), size_hint=(None, 1))
        print_button.bind(on_release=self.print_strokes)

        star_button = Button(text='Star', size = (160,60), size_hint=(None, 1))
        star_button.bind(on_release=self.get_stars)

        # Children
        parent.add_widget(toolbar)
        parent.add_widget(model_toolbar)
        parent.add_widget(self.painter)

        toolbar.add_widget(get_button)
        toolbar.add_widget(self.filename_input)
        toolbar.add_widget(self.index_input)
        toolbar.add_widget(set_button)
        toolbar.add_widget(clear_button)

        model_toolbar.add_widget(sep_button)
        model_toolbar.add_widget(print_button)
        model_toolbar.add_widget(plot_button)
        model_toolbar.add_widget(gen_button)
        model_toolbar.add_widget(sym_button)
        model_toolbar.add_widget(plt_button)
        model_toolbar.add_widget(base_button)
        model_toolbar.add_widget(star_button)

        return root

    """Print the current strokes to the command line"""
    def print_strokes(self, obj):
        print(self.painter.get_strokes())

    """Clear all strokes"""
    def clear_strokes(self, obj):
        self.painter.set_strokes([])

    """Set strokes"""
    def set_strokes(self, obj):
        try:
            index = int(self.index_input.text)
        except ValueError:
            index = 0

        scale = min(self.painter.width,self.painter.height)
        scale = (scale - 200,scale - 200)
        center = (self.painter.width/2, self.painter.height/2)

        tensor = load_tensor.load(self.filename_input.text, center, scale, index)
        #tensor = [item for sublist in tensor for item in sublist]

        self.painter.set_strokes(tensor)

    """Run current strokes through shape model and set colors accordingly"""
    def sep_strokes(self, obj):
        colors = load_tensor.classify(self.painter.get_strokes())
        self.painter.set_colors(colors)

    def plot(self, obj):
        conv_toy.plot_conv(self.painter.get_strokes())

    def generate(self, obj):
        strokes, class_index = generator.generate()
        strokes = np.array(ip.ink_rep_to_array_rep(strokes))

        scale = min(self.painter.width,self.painter.height)
        scale = (scale - 200,scale - 200)
        center = (self.painter.width/2, self.painter.height/2)
        strokes = draw_parser.scale_and_center(strokes, scale, center)

        self.painter.set_strokes(strokes)

    def sym(self,obj):
        # Fast way to iterate though a dateset, used for square analysis
        self.index += 1
        print(self.index)
        self.painter.mode = 1

        scale = min(self.painter.width,self.painter.height)
        scale = (scale - 200,scale - 200)
        center = (self.painter.width/2, self.painter.height/2)

        tensor = load_tensor.load('dataset/square.tfrecords', center, scale, self.index)
        #tensor = [item for sublist in tensor for item in sublist]

        self.painter.set_strokes(tensor)

    def baseline(self, obj):
        baseline_symmetry.analyse(self.painter.get_strokes())

    def plot_inks(self, obj):
        conv_toy.plot_inks(self.painter.get_strokes())

    def get_stars(self, obj):
        """
        n = 0
        for i in range(0,20):
            strokes = stars.get_stars(int(i))
            c = load_tensor.classify(strokes)[0]
            if (c == (1,1,0)):
                n += 1
        print(n)
        """
        strokes = stars.get_stars(int(self.index_input.text))
        self.painter.set_strokes(strokes)




if __name__ == '__main__':
    DrawApp().run()
