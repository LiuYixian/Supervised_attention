file = 'attention_output.txt'
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os
def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)
#
#
def simple_count(file_name):
    lines = 0
    for _ in open(file_name):
        lines += 1
    return lines
# print(iter_count(file)/4)
num = -1
# print(simple_count(file))
# print(iter_count(file))
# wf = open('src_output.txt', 'w', encoding='utf8')
# ss=set()
# with open(file, encoding='utf') as f:
#     for line in tqdm(f):
#     # while True:
#         num += 1
#         line = f.readline()
#         if num % 4 in [0]:
#             if line not in ss:
#                 ss.add(line)
#             else:
#                 print(num)
#                 print(line)
#     print(num)
#
num =-1
# ss=set()
# for line in tqdm(open(file)):
#     num += 1
#     if num % 4 in [0]:
#         if line not in ss:
#             ss.add(line)
#         else:
#             print(num)
#             print(line)
#         ls = line
# print(ls)
# print(num)

normalize = True
def draw(cm, title, color=plt.cm.Blues, text=None, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(text[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", normalize=False, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if normalize:
        data = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
        data = data * 100.
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels())

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # return im, cbar
    return im, None

import json
import numpy as np
for line in tqdm(open(file)):
    num += 1
    tag = num % 4
    if tag == 0:
        src = json.loads(line)
    if tag == 1:
        tgt = json.loads(line)
    if tag == 2:
        one_head_attention = np.array(json.loads(line))
    if tag == 3:
        mean_head_attention = np.array(json.loads(line))
        i == num // 4
        font = {'size': 2}
        # plt.figure(dpi=300)
        matplotlib.rc('font', **font)
        one_head_attention_matrix = one_head_attention
        mean_attention_matrix = mean_head_attention
        X_idx = tgt[:40]
        Y_idx = src[:60]
        # plt.cla()
        plt.figure(dpi=600)
        plt.title(str(i) + str(type) + 'one_head.png')
        im2, cbar = heatmap(one_head_attention_matrix[:40, :60], X_idx, Y_idx, cmap=plt.cm.Reds)
        plt.show()
        plt.close('all')
        plt.figure(dpi=600)
        plt.title(str(i) + str(type) + 'mean_head.png')
        im2, cbar = heatmap(mean_attention_matrix[:40, :60], X_idx, Y_idx, cmap=plt.cm.Reds)
        plt.show()
        plt.savefig('save_test')
        plt.close('all')

# print(ls)
# print(num)





# para = 'python generate.py cnndm/processed --user-dir prophetnet --task translation_prophetnet ' \
#        '--gen-subset test --beam 5 --num-workers 0 --min-len 45 --max-len-b 110 ' \
#        '--no-repeat-ngram-size 3 --lenpen 1.2 --sampling --nbest 5 --sampling-topk 100'
# model = 'models/ex_1_weight_0.1_85_20_SMA_SCE/checkpoint6.pt'
# batch_size = '32'
#
# para += ' --path {}'.format(model)
# para += ' --batch-size {}'.format(batch_size)
# # --path models/ex_1_weight_0.1_85_20_SMA_SCE/checkpoint6.pt
# # --batch-size 32
# def iter_count(file_name):
#     from itertools import (takewhile, repeat)
#     buffer = 1024 * 1024
#     with open(file_name) as f:
#         buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
#         return sum(buf.count('\n') for buf in buf_gen)
#
# while True:
#     if os.path.exists('/'.join(model.split('/')[:-1]) + '/' + 'attention_output.txt'):
#         line_count = iter_count('/'.join(model.split('/')[:-1]) + '/' + 'attention_output.txt')
#         sample_count = line_count / 4
#         if sample_count >= 11490:
#             break
#         else:
#             os.system(para)