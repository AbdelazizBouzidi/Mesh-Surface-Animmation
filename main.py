import numpy as np
from numpy.core.arrayprint import dtype_is_implied
from pyqtgraph.Qt import QtCore, QtGui, ver
import pyqtgraph.opengl as gl
import sys
from opensimplex import OpenSimplex
import wave
np.set_printoptions(threshold=sys.maxsize)


class Terrain(object):
    def __init__(self, song):
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setGeometry(0, 200, 1280, 720)
        self.window.show()
        self.window.setWindowTitle('Terrain')
        self.window.setCameraPosition(
            distance=28, elevation=2, azimuth=0)
        self.nstep = 1
        self.ypoints1 = [i for i in range(-25, 26, self.nstep)]
        self.xpoints1 = [i for i in range(-52, -1, self.nstep)]
        self.ypoints2 = [i for i in range(-25, 26, self.nstep)]
        self.xpoints2 = [i for i in range(2, 53, self.nstep)]
        self.nfaces = len(self.ypoints1)
        self.perlin = OpenSimplex()
        self.offset = 0
        self.beat, self.posbeat = self.BeatMaker(song)
        self.beatindex = 0
        self.color_choice = [{"R": (117/255), "G": (221/255), "B": (221/255)}, {"R": (25/255), "G": (123/255), "B": (189/255)}, {
            "R": (11/255), "G": (19/255), "B": (43/255)}, {"R": (144/255), "G": (41/255), "B": (35/255)}, {"R": (144/255), "G": (41/255), "B": (35/255)}]

        def mesh_init(X, Y, Ampli):
            verts = np.array([
                [
                    x, y, Ampli *
                    self.perlin.noise2d(x=x / 5, y=y / 5 + self.offset)]
                for x in X for y in Y
            ], dtype=np.float32)

            faces = []
            colors = []
            for m in range(self.nfaces - 1):
                yoff = m * self.nfaces
                for n in range(self.nfaces - 1):
                    faces.append([n + yoff, yoff + n + self.nfaces,
                                  yoff + n + self.nfaces + 1])
                    faces.append(
                        [n + yoff, yoff + n + 1, yoff + n + self.nfaces + 1])
                    colors.append(
                        [int(117/255), int(221/255), int(221/255), 0.78])
                    colors.append(
                        [int(117/255), int(221/255), int(221/255), 0.8])

            return np.array(verts), np.array(faces), np.array(colors)
        verts1, faces1, colors1 = mesh_init(self.ypoints1, self.xpoints1, 0.2)
        verts2, faces2, colors2 = mesh_init(self.ypoints2, self.xpoints2, 0.2)
        self.m1 = gl.GLMeshItem(
            vertexes=verts1,
            faces=faces1, faceColors=colors1,
            smooth=False, drawEdges=False,
        )
        self.m1.setGLOptions('additive')
        self.window.addItem(self.m1)
        self.m2 = gl.GLMeshItem(
            vertexes=verts2,
            faces=faces2, faceColors=colors2,
            smooth=False, drawEdges=False,
        )
        self.m2.setGLOptions('additive')
        self.window.addItem(self.m2)

    def start(self):
        """
        get the graphics window open and setup
        """

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(10)
        self.start()
        self.update()

    def update(self):
        """
        update the mesh and shift the noise each time
        """
        def mesh_init(X, Y, Ampli):
            verts = np.array([
                [
                    x, y, Ampli * self.perlin.noise2d(x=(x / 5)+self.offset, y=(y / 5))]
                for x in X for y in Y
            ], dtype=np.float32)

            faces = []
            colors = []
            a = int(self.posbeat[self.beatindex]*4)
            if a > 4:
                a = 4

            color = self.color_choice[a]
            # print(color["R"])
            for m in range(self.nfaces - 1):
                yoff = m * self.nfaces
                for n in range(self.nfaces - 1):
                    faces.append([n + yoff, yoff + n + self.nfaces,
                                  yoff + n + self.nfaces + 1])
                    faces.append(
                        [n + yoff, yoff + n + 1, yoff + n + self.nfaces + 1])
                    colors.append([color["R"], color["G"], color["B"], 0.795])
                    colors.append([color["R"], color["G"], color["B"], 0.8])

            return np.array(verts), np.array(faces), np.array(colors)
        if self.beatindex < len(self.beat):
            verts1, faces1, colors1 = mesh_init(
                self.ypoints1, self.xpoints1, 3*self.beat[self.beatindex])
            verts2, faces2, colors2 = mesh_init(
                self.ypoints2, self.xpoints2, 3*self.beat[self.beatindex])
            self.m1.setMeshData(
                vertexes=verts1, faces=faces1, faceColors=colors1
            )
            self.m2.setMeshData(
                vertexes=verts2, faces=faces2, faceColors=colors2
            )
            self.offset -= 0.18
            self.window.grabFrameBuffer().save("frames10/"+str(self.beatindex)+'.png')
            self.beatindex += 1
        else:
            print("done")

    def BeatMaker(self, song):
        spf = wave.open(song, "r")
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, "int16")
        signal = signal.astype(float)
        fs = spf.getframerate()
        print(fs)
        print(len(signal))
        signal = np.array([signal[t]
                           for t in range(0, 3639500, 2900)])
        print(len(signal))
        return signal/np.max(signal), np.absolute(2*signal/np.max(signal))


if __name__ == '__main__':
    t = Terrain("Human (Frank Castle) [High quality].wav")
    t.animation()
