import numpy as np
import cv2 as cv

import vispy as vp
from vispy import app, gloo, scene, plot
import vispy.visuals as visuals


import time
import sys
import math

import graph_util

class MultiGraph():
    def __init__(self, graphs):
        """
            Graphs should be a list of LiveGraphBase

            MultiGraph is required to run LiveGraphs even if you only have one
        """
        x_size = 0
        y_size = 0
        self.graphs = graphs

        self.pause = False

        self._timer = app.Timer('auto', connect=self.on_timer, start=False)


    def start(self, paused=False):
        self.pause = paused

        for g in self.graphs[::-1]:
            g.start()
        if self.pause:
            self.do_paused()
        self._timer.start()
        app.run()
        for g in self.graphs:
            g.end()

    def on_timer(self, event):
        if not self.pause:
            for g in self.graphs:
                g.on_timer(self._timer.interval)

    def do_paused(self):
        for g in self.graphs:
            g.update_data_(time.time())

class LiveGraphBase():
    def __init__(self, callback, v_size, title):
        """
            Callback:
                (current_data : np.ndarray, step:int, desired_end_time:float)
                    -> next_step_data : np.ndarray
        """
        self.v_size = v_size

        self.callback = callback
        self.data = None
        self.it = 0
        self.start_time = time.time()

        self.draw_time = .01

        self.pause = False
        self.speed_div = 0

        self.recording=False

        self.canvas = scene.SceneCanvas(title, keys="interactive",
                size=self.v_size, show=False)

        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_resize)

        self.physical_size = self.canvas.physical_size
        self.view = self.canvas.central_widget.add_view()

    def autozoom(self):
        self.view.camera = scene.PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()

    def make_3d_camera(self, fov=60):
        self.camera = scene.cameras.ArcballCamera(parent=self.view.scene, fov=fov)
        self.view.camera = self.camera

    # Override these
    def on_start_(self):
        pass

    def on_resize_(self, event):
        pass

    def update_data_(self, end_time):
        self.data = self.callback(self.data, self.it, end_time)
        self.visual.set_data(self.data)

    def on_draw_(self, event):
        gloo.clear()
        self.visual.draw()

    def start(self):
        self.canvas.show()
        self.on_start_()

    def end(self):
        if self.recording:
            self.end_record()

    def on_resize(self, event):
        print ('resize:', event)
        self.physical_size = event.size
        print (self.physical_size)

    def on_timer(self, event):
        interval = 0.0166
        if not self.pause:
            runtime = interval - self.draw_time
            if self.speed_div > 0:
                runtime /= (1+self.speed_div)
            self.start_time = time.time()
            self.update_data_(self.start_time + runtime)
            self.record_frame()
            self.it += 1

    def speed_up(self, dx):
        self.speed_div+=dx
        if self.speed_div < 0:
            self.speed_div = 0
        print ('speed', 1/(1+self.speed_div))

    def number_press(self, num):
        pass

    def on_key_press(self, event):
        if event.text==' ':
            self.pause = not self.pause
        elif event.text=='[':
            self.speed_up(1)
        elif event.text==']':
            self.speed_up(-1)
        elif event.text >= '0' and event.text <= '9':
            n = int(event.text)
            if event.text=='0':
                n=10
            self.number_press(n)
        elif event.text=='r':
            self.toggle_record()
        else:
            print (event.type, event.text)

    def on_draw(self, event):
        print('draw')
        draw_start = time.time()
        self.on_draw_(event)
        self.draw_time = time.time() - draw_start
        # print ('redraw')

    def get_visual(self):
        return self.visual

    def on_mouse_release(self, event):
        pass

    def toggle_record(self):
        if not self.recording:
            import datetime
            tstr=  datetime.datetime.now().strftime("%m-%d %H_%M_%S")
            name= self.__class__.__name__ +  ' '+ tstr
            if self.canvas.title:
                name += ' '+self.canvas.title
            name += '.avi'
            self.record(name, 1)
        else:
            self.end_record()

    def record(self, name, updates_per_frame):
        import threading
        if name.endswith('.gif'):
            self.record_fmt = 'gif'
            import imageio
        elif name.endswith('.avi'):
            self.record_fmt = 'avi'
            import cv2
        else:
            print("Export format not supported:", name.split('.')[-1])
            assert(False)
            return
        self.recording=True
        self.curUpdate = 0
        self.updatesPerFrame = updates_per_frame

        self.writeQueue = []
        self.shouldStop=False
        self.thread = threading.Thread(target=lambda : self.record_write_thread(name))
        self.thread.start()

    def record_write_thread(self, name):
        print ('Recording', name)
        import os
        frame=0
        name = 'recordings/'+name
        if not os.path.exists('recordings/'):
            os.path.mkdir('recordings/')
        lastTime = time.time()

        if self.record_fmt=='gif':
            import imageio
            writer = imageio.get_writer(name, fps=fps)
            fps = 30
        elif self.record_fmt=='avi':
            import cv2
            fps = 30
            size = np.array(self.physical_size)
            if sys.platform =='darwin':
                size*=2
                fmt = 'MJPG'
            else:
                # I don't actually know what this should be for windows
                fmt = 'DIVX'
            writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*fmt), fps, size)

        while not self.shouldStop:
            while self.writeQueue:
                im = self.writeQueue.pop(0)

                if self.record_fmt=='gif':
                    writer.append_data(im)
                elif self.record_fmt=='avi':
                    writer.write(im)

                frame += 1
                if frame%fps==0:
                    s = f'{self.record_fmt} t= {frame//fps}'
                    t = time.time()
                    s += f' x{t-lastTime:.1f}'
                    lastTime=t
                    if self.writeQueue:
                        s+= f'\tbehind {len(self.writeQueue)}'
                    print(s)

            time.sleep(.01)

        if self.record_fmt=='gif':
            writer.close()
        elif self.record_fmt=='avi':
            writer.release()

        print (f"saved {name}")


    def record_frame(self):
        if self.recording:
            if self.curUpdate>=self.updatesPerFrame:
                self.curUpdate=0
                im = self.canvas.render(alpha=False)
                self.writeQueue.append(im)
            else:
                self.curUpdate+=1

    def end_record(self):
        if self.recording:
            self.shouldStop=True
            self.recording=False
            self.thread.join()
            print('Finished recording')

class Display2D(LiveGraphBase):
    def __init__(self, callback, data, cmap='hsv', title=''):
        """
        """
        print ('constructor')
        if isinstance(data, np.ndarray):
            self.src = (data.shape[0], data.shape[1])
            # self.shape = data.shape
        else:
            self.src = data[:2]
        v_size = (800,  800)

        self.cmap = cmap

        self.scratch = None
        super().__init__(callback, v_size, title)

        self.visual = scene.visuals.Image(self.to_color(data), interpolate='nearest', parent=self.view.scene)

        self.autozoom()

    def update_data_(self, end_time):
        d = self.callback(self.data, self.it, end_time)
        if d is not None:
            self.data = d
            rgb = self.to_color(self.data)
            self.visual.set_data(rgb)
            self.visual.update()

    def to_color(self, data):
        if self.cmap == 'rgb':
            return data
        if self.cmap == 'hsv':
            return graph_util.hsv_to_rgb(data)
        elif self.cmap[:4]=='hsv_':
            mode = self.cmap[4:]
            return graph_util.complex_to_rgb(data, mode=mode)

class Voxel3D(LiveGraphBase):
    def __init__(self, callback, data, cmap='hsv', title=''):
        """
        """
        print ('constructor')
        if isinstance(data, np.ndarray):
            self.src = (data.shape[0], data.shape[1])
            # self.shape = data.shape
        else:
            self.src = data[:2]
        v_size = (800,800)

        self.cmap = cmap
        self.scratch = None

        super().__init__(callback, v_size, title)

        self.visual = scene.visuals.Volume(self.to_color(data), clim=(0,1), parent=self.view.scene, interpolation='nearest')
        if self.cmap == 'abs':
            self.visual.cmap = graph_util.GLSL_Oranges()
        elif self.cmap == 'hsv':
            self.visual.cmap = graph_util.GLSL_HSVColor()
        elif self.cmap == 'greens':
            self.visual.cmap = graph_util.GLSL_DualColor()

        self.make_3d_camera()

    def update_data_(self, end_time):
        d = self.callback(self.data, self.it, end_time)
        if d is not None:
            self.data = d
            rgb = self.to_color(self.data)
            self.visual.set_data(rgb)
            self.visual.update()

    def to_color(self, data):
        data = self.to_color_(data)
        #Highlight corners for visibility
        cur = 0
        for x in [0, -1]:
            for y in [0, -1]:
                for z in [0, -1]:
                    data[x,y,z] = .5
        data[0,0,0]=.75
        return data

    def to_color_(self, data):
        if self.cmap == 'rgb':
            return data
        if self.cmap == 'hsv':

            abs = np.abs(data)
            ang = np.angle(data) / math.tau + .5
            ang = np.clip(ang, 0, .99)

            aLevels = 4
            abs = (abs*aLevels).astype(int)
            abs[abs>=aLevels] = aLevels
            c = (abs + ang) / (aLevels+1)
            return c

        elif self.cmap == 'abs':
            return np.abs(data)

        elif self.cmap[:4]=='hsv_':
            mode = self.cmap[4:]
            return graph_util.complex_to_rgb(data, mode=mode)


class Line1D(LiveGraphBase):
    def __init__(self, callback, data, title='', **kwargs):
        v_size = (1000, 500)
        LiveGraphBase.__init__(self, callback, v_size, title)

        self.N = data.shape[-1]
        self.L = data.shape[0] if data.ndim > 1 else 1

        self.data = data

        defaults = {'color':'w', 'marker_size':0, 'connect':'strip'}
        for k, d in defaults.items():
            if k not in kwargs:
                kwargs[k] = d

        self.x = np.linspace(0, self.physical_size[0], num=self.N)
        self.visual = scene.visuals.LinePlot(self.stack(0), parent=self.view.scene, **kwargs)

    def get_data(self, i):
        if self.data.ndim==1:
            return self.data
        else:
            return self.data[i]

    def y_to_p(self, y):
        return self.physical_size[1] * (.25-y)

    def stack(self, i):
        return np.stack((self.x, self.y_to_p(self.get_data(i))), axis=-1)

    def get_col(self, i):
        return [(1,1,1), (1,0,1), (0,1,1), (1,1,0), (1,0,0), (0,1,0), (0,1,1)][i]

    def on_start_(self):
        pass
    def on_resize_(self, event):
        pass
    def update_data_(self, end_time):
        self.data = self.callback(self.data, self.it, end_time)

    def on_draw(self, event):
        # print ('redraw')
        gloo.clear()

        for l in range(self.L):
            self.visual.set_data(self.stack(l), color=self.get_col(l))
            self.visual.draw()


class ComplexLine(LiveGraphBase):
    def __init__(self, callback, data, title='',  **kwargs):
        v_size = (1000, 500)
        LiveGraphBase.__init__(self, callback, v_size, title)

        self.N = data.shape[-1]
        self.L = data.shape[0] if data.ndim > 1 else 1

        self.data = data

        defaults = {'color':'w', 'marker_size':0, 'connect':'strip'}
        for k, d in defaults.items():
            if k not in kwargs:
                kwargs[k] = d

        self.x = np.linspace(0, self.physical_size[0], num=self.N)
        self.x = np.tile(self.x, self.L)
        self.hsv = np.ones((self.N*self.L, 3), float)
        self.hsv[:, 2] = .7

        self.visual = scene.visuals.LinePlot(self.stack(np.abs(self.data)), parent=self.view.scene, **kwargs)

    def y_to_p(self, y):
        tm = 10
        return  (self.physical_size[1]-tm-10) * (1-y) + tm

    def get_col(self, abs, ang):
        return graph_util.abs_ang_to_rgb(abs, ang, self.hsv, 'line0cap')

    def stack(self, data):
        data = data.reshape(-1)
        return np.stack((self.x, self.y_to_p(data)), axis=-1)

    def on_start_(self):
        pass
    def on_resize_(self, event):
        pass

    def update_data_(self, end):
        self.data = self.callback(self.data, self.it, end)
        if self.data.shape[1]>1:
            self.data[:, 0] = 0
            self.data[:, -1] = 0
        data = self.data.reshape(-1)
        abs, ang = np.abs(data), np.angle(data)

        col = self.get_col(abs, ang)
        self.visual.set_data(self.stack(abs), color=col)


class ComplexLineIn3D(LiveGraphBase):
    def __init__(self, callback, data, title='',  **kwargs):
        v_size = (1000, 500)
        LiveGraphBase.__init__(self, callback, v_size, title)

        self.N = data.shape[-1]
        self.L = data.shape[0] if data.ndim > 1 else 1

        self.data = data

        defaults = {'color':'w', 'marker_size':0, 'connect':'strip'}
        for k, d in defaults.items():
            if k not in kwargs:
                kwargs[k] = d

        self.x = np.linspace(0, self.physical_size[0], num=self.N)
        self.x = np.tile(self.x, self.L)

        self.hsv = np.ones((self.N*self.L, 3), float)
        self.hsv[:, 2] = .7

        self.visual = scene.visuals.LinePlot(self.stack(self.data.real), parent=self.view.scene, **kwargs)
        self.make_3d_camera(fov=10)

    def y_to_p(self, y):
        return self.physical_size[1] * (1-y) - 10

    def get_col(self, data):
        return graph_util.complex_to_rgb(data, self.hsv, 'max')

    def stack(self, data):
        data = data.reshape(-1)
        real = self.y_to_p(data.real)
        imag = self.y_to_p(data.imag)
        return np.stack((self.x, imag, -real), axis=-1)

    def update_data_(self, end):
        self.data = self.callback(self.data, self.it, end)
        if self.data.shape[1]>1:
            self.data[:, 0] = 0
            self.data[:, -1] = 0

        data = self.data.reshape(-1)

        col = self.get_col(data)
        self.visual.set_data(self.stack(data), color=col)

if __name__ == '__main__':
    win = Display2D(lambda x,y,z: x, (100,100), 1)

    if sys.flags.interactive != 1:
        app.run()
