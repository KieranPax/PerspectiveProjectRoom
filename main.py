import math
import time
import pyglet
from pyglet.gl import *
from gl_lib import *


class Camera:
    def __init__(self, initial_pos: Vector):
        self.pos = initial_pos
        self.rot = 0, 0

    def move(self, frwd, rght):
        a = math.radians(self.rot[0])
        self.pos += Vector(
            rght * math.cos(a) - frwd * math.sin(a),
            0,
            rght * math.sin(a) + frwd * math.cos(a),
        )
        self.pos.x = max(-4.95, min(self.pos.x, 4.95))
        self.pos.z = max(-4.95, min(self.pos.z, 4.95))

    def pan(self, hori, vert):
        self.rot = hori + self.rot[0], max(-90, min(vert + self.rot[1], 90))

    def to_matrix(self):
        return Matrix().translate(self.pos).rotate_y(self.rot[0]).rotate_x(self.rot[1])


class App:
    width = 1200
    height = 800
    target_fps = 60

    def __init__(self):
        self._starttime = 0
        self.time = 0
        self.delta_time = 0
        self.frame_count = 0
        self.walkspeed = 5
        self.fps_tracker = []

        self.cam = Camera(Vector(0, -2, -3))

        self.room = Model()
        self.object = Model()
        self.show_object = True

        self.obj_render_image = RenderImage((self.width, self.height))
        self.room_render_image = RenderImage((self.width, self.height))

        self.proj_mat = Matrix().perspective(60, self.width / self.height, 0.001, 100)

    def start(self):
        glEnable(GL_ALPHA_TEST)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.room.load_shader('project')
        self.room.load_shader('room_base')
        self.room.load_model('room.o')
        self.room.load_texture('walls_base.img')

        self.object.load_shader('object')
        self.object.load_model('example.torus.o')
        # self.object.model_mat.scale(0.5)
        self.object.model_mat.translate(Vector(0, 1.5, 0))

        self.obj_render_image.initiate()
        self.room_render_image.dim = self.room.texture.dim
        self.room_render_image.initiate(self.room.texture)

    def frame(self, pressed):
        tt = self.time
        self.time = time.perf_counter() - self._starttime
        self.delta_time = self.time - tt
        self.fps_tracker.append(1 / self.delta_time)

        glClearColor(0.2, 0.7, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = self.cam.to_matrix()
        self.room.draw(self.proj_mat * view)
        if self.show_object:
            self.object.draw(self.proj_mat * view)

        if self.frame_count % 120 == 119:
            print(f'fps : {sum(self.fps_tracker) / len(self.fps_tracker)}')
            self.fps_tracker.clear()
        self.frame_count += 1

        if pressed[pyglet.window.key.W]:
            self.cam.move(self.walkspeed * self.delta_time, 0)
        if pressed[pyglet.window.key.S]:
            self.cam.move(-self.walkspeed * self.delta_time, 0)

        if pressed[pyglet.window.key.A]:
            self.cam.move(0, self.walkspeed * self.delta_time)
        if pressed[pyglet.window.key.D]:
            self.cam.move(0, -self.walkspeed * self.delta_time)

    def apply_perspective(self):
        if self.cam.pos.length() < 3:
            return

        print('Applying Perspective')
        start = time.perf_counter()

        self._apply_perspective()
        self.show_object = False

        end = time.perf_counter()
        print(f'Finish : {end - start:0.5f}s')

    def _apply_perspective(self):

        self.obj_render_image.start_render()

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view = self.cam.to_matrix()
        self.object.draw(self.proj_mat * view)

        self.obj_render_image.finish_render()

        glViewport(0, 0, *self.room_render_image.dim)

        self.room_render_image.start_render()

        glClear(GL_DEPTH_BUFFER_BIT)
        view = self.cam.to_matrix()
        self.room.load_shader('project', False)
        self.room.draw(self.proj_mat * view, self.obj_render_image.tex)
        self.room.load_shader('room_base', False)

        self.room_render_image.finish_render()

        glViewport(0, 0, self.width, self.height)


def main():
    app = App()
    # create window
    window = pyglet.window.Window(app.width, app.height, 'Perpective')
    window.set_exclusive_mouse(True)

    # store store of key values
    p = pyglet.window.key.KeyStateHandler()

    # bind 'app' to window functions
    app.start()

    @window.event
    def on_draw():
        window.push_handlers(p)
        app.frame(p)

    @window.event
    def on_mouse_press(x, y, button, mod):
        if button == 1:
            if app.show_object:
                app.apply_perspective()
        if button == 4:
            app.show_object = True

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        s = 0.05
        app.cam.pan(dx * s, -dy * s)

    # set fps
    pyglet.clock.schedule_interval(lambda a: a, 1 / app.target_fps)

    pyglet.app.run()


if __name__ == '__main__':
    main()
