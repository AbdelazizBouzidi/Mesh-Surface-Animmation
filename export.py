from moviepy.editor import *
import numpy as np

img1 = ["frames10/"+str(i)+".png" for i in range(0, 1255)]
img2 = ["frames/"+str(i)+".png" for i in range(0, 5548)]
img = np.concatenate((img1, img2))

your_song_mp2 = "Human (Frank Castle)152.mp3"

clips = []
for j in range(0, 3406):
    clips.append(ImageClip(img[j]).set_duration(0.03289))
    if j % 10 == 0:
        print((j/6803)*100)
concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile(
    "final.mp4", fps=30, audio=your_song_mp2)
