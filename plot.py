import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import vocabulary
import numpy as np


def plot_and_save(text, weight, mel, step, path):
    # 한글 폰트 깨짐 대응
    font_name = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\NanumGothic.ttf").get_name()
    rc("font", family=font_name)

    w, h = plt.figaspect(0.5)  # 가로로 2배
    fig = plt.figure(figsize=(w, h))

    ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.4])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])

    ax1.imshow(weight, aspect="auto")
    ax1.set_xlabel("Encoder")
    ax1.set_ylabel("Decoder")
    ax1.set_xticks(np.arange(len(text)))
    ax1.set_xticklabels([vocabulary.IDX2CHAR[c] for c in text])
    ax1.set_title(vocabulary.token2str(text))

    ax2.imshow(np.flipud(mel))
    ax2.set_xlabel("step = " + str(step))

    plt.savefig(path)
    plt.close()
