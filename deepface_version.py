import operator

from PIL import Image, ImageGrab
import time
import mss
from deepface import DeepFace
from deepface.extendedmodels import Emotion
import matplotlib.pyplot as plt
import numpy as np

start = time.time()
old_emotion = {}


def analyze_screen():
    local_start_timer = time.time()
    with mss.mss() as sct:
        sct.compression_level = 1
        img_path = sct.shot(output = 'tmp/temp.png')
    print('After Grab and Load', time.time() - local_start_timer)
    response = DeepFace.analyze(img_path, ['emotion'])
    print('After Process', time.time() - local_start_timer)
    return response['emotion']


def stats_update():
    try:
        emotions = analyze_screen()
        fig, ax = plt.subplots()
        group_data = list(emotions.values())
        group_names = list(emotions.keys())
        group_mean = np.mean(group_data)
        ax.barh(group_names, group_data)
        labels = ax.get_xticklabels()
        plt.show()
    except:
        time.sleep(1.0)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width() / 2, height),
                    xytext = (0, 3),  # 3 points vertical offset
                    textcoords = "offset points",
                    ha = 'center', va = 'bottom')


def draw_plot():  # natural, happy, sad, surprise, fear, disgust, angry
    global old_emotion
    try:
        emotions = analyze_screen()
    except:
        return
    labels = list(emotions.keys())

    group_data = list(emotions.values())
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, group_data, width, label = 'Current')
    if len(old_emotion) != 0:
        rects2 = ax.bar(x + width / 2, old_emotion, width, label = 'Old')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Emotion')
    ax.set_title('Emotion by deepface')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)
    if len(old_emotion) != 0:
        autolabel(rects2, ax)
    fig.tight_layout()
    plt.show()
    old_emotion = group_data


time.sleep(3)
while True:
    draw_plot()
    time.sleep(1)
