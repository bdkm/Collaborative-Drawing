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
import utility.load_tensor as load_tensor
import utility.separate as separate

class MainLayout(BoxLayout):
    pass

class Toolbar(BoxLayout):
    pass

class Canvas(Widget):
    def __init__(self):
        super(Canvas, self).__init__()
        self.strokes = []

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Begin new stroke
            self.strokes.append([touch.x,touch.y])
            self.timer = time.time()
            with self.canvas:
                Color(0.6,0.6,0.6)
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=dp(2))

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            # Add new point
            elapsed = time.time() - self.timer
            if elapsed > 0.05:
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

        # Main toolbar
        toolbar = Toolbar()

        get_button = Button(text='Print', size = (80,60), size_hint=(None, 1))
        get_button.bind(on_release=self.print_strokes)

        set_button = Button(text='Set', size = (80,60), size_hint=(None, 1))
        set_button.bind(on_release=self.set_strokes)

        clear_button = Button(text='Clear', size = (80,60), size_hint=(None, 1))
        clear_button.bind(on_release=self.clear_strokes)

        self.filename_input = TextInput(text='dataset/circle.tfrecords', size_hint=(1, 1))
        self.index_input = TextInput(text='0', size = (60,60), size_hint=(None, 1))

        # Model toolbar
        model_toolbar = Toolbar()

        sep_button = Button(text='Separate', size = (160,60), size_hint=(None, 1))
        sep_button.bind(on_release=self.sep_strokes)

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
        """
        try:
            index = int(self.index_input.text)
        except ValueError:
            index = 0

        scale = min(self.painter.width-200,self.painter.height-200)
        print(scale)
        scale = (scale,scale)
        print(scale)
        center = (self.painter.width/2, self.painter.height/2)

        tensor = load_tensor.load("dataset/camera.tfrecords", center, scale, index)
        self.painter.set_strokes(tensor)
        colors = separate.separate("dataset/splits/camera-split-%05d.tfrecords" % index)
        self.painter.set_colors(colors)
        """
        colors = load_tensor.classify(self.painter.get_strokes())
        self.painter.set_colors(colors)

if __name__ == '__main__':
    DrawApp().run()
