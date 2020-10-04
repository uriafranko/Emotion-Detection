

from lib.FaceCropper import FaceCropper
import time
import mss
import boto3
import matplotlib.pyplot as plt
import numpy as np

aws_access_key_id = ""  # aws access key id
aws_secret_access_key = ""  # aws secret access key
aws_region_name = ""  # aws region name

client = boto3.client('rekognition',
                      aws_access_key_id = aws_access_key_id,
                      aws_secret_access_key = aws_secret_access_key,
                      region_name = aws_region_name)

old_emotion = {}


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width() / 2, height),
                    xytext = (0, 3),  # 3 points vertical offset
                    textcoords = "offset points",
                    ha = 'center', va = 'bottom')


def format_response(response):
    labels = ['HAPPY', 'CALM', 'ANGRY', 'SAD', 'FEAR', 'CONFUSED', 'SURPRISED', 'DISGUST']
    new_response = {}
    for label in labels:
        for em in response:
            if em['Type'] in label:
                new_response[label] = em['Confidence']
    return new_response


def get_face_data(face):
    with open(face, 'rb') as img:
        img_data = img.read()
    response = client.detect_faces(
        Image = {
            'Bytes': img_data,
        },
        Attributes = [
            'ALL',
        ]
    )
    return response['FaceDetails'][0]['Emotions']


def analyze_screen():
    global client
    with mss.mss() as sct:
        sct.compression_level = 1
        img_path = sct.shot(output = 'tmp/mon-3.png')
    detector = FaceCropper()
    img_path = detector.generate(img_path, False)
    response = None
    for face in img_path:
        try:
            response = get_face_data(face)
            break
        except Exception as e:
            print(e)
            continue
    if response is None:
        return
    emotions = format_response(response)
    return emotions


def draw_plot():  # natural, happy, sad, surprise, fear, disgust, angry
    start = time.time()
    global old_emotion
    try:
        emotions = analyze_screen()
    except Exception as e:
        print('Error', e)
        return
    if emotions is None:
        return
    labels = emotions.keys()
    group_data = [round(num, 1) for num in emotions.values()]
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
    print('All Process', time.time() - start)  # Debug
    plt.show()
    old_emotion = group_data


time.sleep(3)
while True:
    draw_plot()
    time.sleep(1)
