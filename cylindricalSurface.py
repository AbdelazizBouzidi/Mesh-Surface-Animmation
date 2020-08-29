from math import exp
import time
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
        self.window.setGeometry(0, 500, 1280, 720)
        self.window.show()
        self.window.setWindowTitle('Terrain')
        self.window.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0),
                                      distance=105, elevation=0, azimuth=90)
        self.nstep = 1
        self.ypoints1 = [i for i in range(-200, 201, self.nstep)]
        self.xpoints1 = [i for i in range(-10, 11, self.nstep)]
        self.ypoints2 = [i for i in range(-200, 201, self.nstep)]
        self.xpoints2 = [i for i in range(-10, 11, self.nstep)]
        self.nfaces1 = len(self.ypoints1)
        self.nfaces2 = len(self.xpoints1)
        self.perlin = OpenSimplex()
        self.offset = 0
        self.beat, self.posbeat = self.BeatMaker(song)
        self.beatindex = 0
        self.t = 0
        self.color_choice = [{"R": (115 / 255), "G": (115 / 255), "B": (115 / 255)}, {"R": (64 / 255), "G": (64 / 255), "B": (64/255)}, {
            "R": (38/255), "G": (38/255), "B": (38/255)}, {"R": (38/255), "G": (38/255), "B": (38/255)}, {
            "R": (11/255), "G": (19/255), "B": (43/255)}]

        def mesh_init(X, Y, z, Ampli):
            verts = np.array([
                [
                    x, y, z + Ampli *
                    self.perlin.noise2d(x=x / 5, y=y / 5 + self.offset)]
                for x in X for y in Y
            ], dtype=np.float32)
            # for j in range(0, len(verts)):
            #     if abs(verts[j, 1]) > 20:
            #         verts[j, 2] = 100
            faces = []
            colors = []
            for m in range(self.nfaces2 - 1):
                yoff = m * self.nfaces2
                for n in range(self.nfaces1 - 1):
                    faces.append([n + yoff, yoff + n + self.nfaces1,
                                  yoff + n + self.nfaces1 + 1])
                    faces.append(
                        [n + yoff, yoff + n + 1, yoff + n + self.nfaces1 + 1])
                    colors.append(
                        [int(117/255), int(221/255), int(221/255), 1])
                    colors.append(
                        [int(117/255), int(221/255), int(221/255), 1])

            return np.array(verts), np.array(faces), np.array(colors)
        verts1, faces1, colors1 = mesh_init(
            self.ypoints1, self.xpoints1, 5, 0.2)
        verts2, faces2, colors2 = mesh_init(
            self.ypoints2, self.xpoints2, -5, 0.2)
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
        # timer = QtCore.QTimer()
        # timer.timeout.connect(self.update)
        # timer.start(10)
        self.start()

    def update(self):
        """
        update the mesh and shift the noise each time
        """
        def mesh_init(X, Y, z, Ampli, signe):

            verts = np.array([
                [
                    x+(Ampli/1.5)*self.perlin.noise2d(x=y/10, y=y/10 - self.offset + self.offset), y,  (z+signe*np.sqrt(10**2-x**2) - signe*Ampli*self.perlin.noise2d(x=x/5, y=y/5 + self.offset))] if abs(x) < 10 else [x, y, 0]
                for y in Y for x in X
            ], dtype=np.float32)
            faces = []
            colors = []
            a = int(self.posbeat[self.beatindex]*4)
            if a > 4:
                a = 4

            color = self.color_choice[a]
            # print(color["R"])
            for m in range(self.nfaces1 - 1):
                yoff = m * self.nfaces2
                for n in range(self.nfaces2 - 1):
                    faces.append([n + yoff, yoff + n + self.nfaces2,
                                  yoff + n+self.nfaces2 + 1])
                    faces.append(
                        [n + yoff, yoff + n + 1, yoff + self.nfaces2 + n + 1])
                    colors.append([color["R"], color["G"], color["B"], 1])
                    colors.append([color["R"], color["G"], color["B"], 0.995])

            return np.array(verts), np.array(faces), np.array(colors)
        stime = time.time()
        if self.beatindex < len(self.beat):
            verts1, faces1, colors1 = mesh_init(
                self.xpoints1, self.ypoints1, 0, 10*self.beat[self.beatindex], 1)
            verts2, faces2, colors2 = mesh_init(
                self.xpoints2, self.ypoints2, 0, 10*self.beat[self.beatindex], -1)
            self.m1.setMeshData(
                vertexes=verts1, faces=faces1, faceColors=colors1
            )
            self.m2.setMeshData(
                vertexes=verts2, faces=faces2, faceColors=colors2
            )

            self.window.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0),
                                          distance=200, elevation=4*(self.perlin.noise2d(x=self.beat[self.beatindex+1], y=self.beat[self.beatindex])), azimuth=90+4*(self.perlin.noise2d(x=self.beat[self.beatindex+1], y=self.beat[self.beatindex])))
            self.offset -= 0.2
            self.window.grabFrameBuffer().save("frames/"+str(self.beatindex)+'.png')
            self.beatindex += 1

        else:
            print("done")
        self.t += (time.time() - stime)/30
        print('{:.0f} FPS'.format(self.t))

    def BeatMaker(self, song):
        spf = wave.open(song, "r")
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, "int16")
        signal = signal.astype(float)
        fs = spf.getframerate()
        print(fs)

        signal = np.array([signal[t]
                           for t in range(0, len(signal), int((2*fs)/30))])
        signalabs = np.absolute(signal)

        return (signal-np.mean(signal))/(np.max(signal)-np.min(signal)), 5*np.absolute(signalabs-np.mean(signalabs)/(np.max(signalabs)-np.min(signalabs)))/max(np.absolute(signalabs-np.mean(signalabs)/(np.max(signalabs)-np.min(signalabs))))


if __name__ == '__main__':
    t = Terrain("Ludovico Einaudi - Eros.wav")
    t.animation()
