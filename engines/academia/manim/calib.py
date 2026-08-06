from manim import *
class Calib(Scene):
    def construct(self):
        # reference: horizontal line -3..3 (6 units), vertical line -3..3 (6 units)
        self.add(Line([-3,0,0],[3,0,0],color=RED, stroke_width=8))
        self.add(Line([0,-3,0],[0,3,0],color=GREEN, stroke_width=8))
        self.add(Dot([0,0,0],color=WHITE))
        self.wait(0.1)
