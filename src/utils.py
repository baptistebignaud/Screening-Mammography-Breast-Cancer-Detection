### util functions

import time
import torch
import numpy
import scipy.stats as st
from args import args

lastDisplay = time.time()
def display(string, end = '\n', force = False):
    global lastDisplay
    if time.time() - lastDisplay > 0.1 or force:
        lastDisplay = time.time()
        print(string, end=end)

def timeToStr(time):
    hours = int(time) // 3600
    minutes = (int(time) % 3600) // 60
    seconds = int(time) % 60
    return "{:d}h{:02d}m{:02d}s".format(hours, minutes, seconds)

def confInterval(scores):
    if scores.shape[0] == 1:
        low, up = -1., -1.
    elif scores.shape[0] < 30:
        low, up = st.t.interval(0.95, df = scores.shape[0] - 1, loc = scores.mean(), scale = st.sem(scores.numpy()))
    else:
        low, up = st.norm.interval(0.95, loc = scores.mean(), scale = st.sem(scores.numpy()))
    return low, up

def createCSV(trainSet, validationSet, testSet):
    if args.csv != "":
        f = open(args.csv, "w")
        text = "epochs, "
        for datasetType in [trainSet, validationSet, testSet]:
            for dataset in datasetType:
                text += dataset["name"] + " loss, " + dataset["name"] + " accuracy, "
        f.write(text + "\n")
        f.close()

def updateCSV(stats, epoch = -1):
    if args.csv != "":
        f = open(args.csv, "a")
        text = ""
        if epoch >= 0:
            text += "\n" + str(epoch) + ", "
        for i in range(stats.shape[0]):
            text += str(stats[i,0].item()) + ", " + str(stats[i,1].item()) + ", "
        f.write(text)
        f.close()
def pfbeta(labels, predictions, beta):
    """
    Probabilistic F-beta score
    """
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / (y_true_count+10e-8)
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0
print(" utils,", end="")
