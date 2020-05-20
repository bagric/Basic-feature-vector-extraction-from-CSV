import EVM, scipy, sys, glob, os, fnmatch, random, sklearn
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix

os.chdir("data")

def season_switch(i):
    switcher={
        1:1418, #from march(spring)
        2:3626, #from jun(summer)
        3:6554, #from october(fall)
        4:7298, #from november(winter)
        }
    return switcher.get(i,1418)

def calc_feature_vector(inputs):
    fft_form = np.abs(scipy.fftpack.fft(inputs))
    peaks, _ = find_peaks(fft_form)
    fft_peaks = inputs.iloc[peaks]
    #numpy array for EVM
    feat_vector = np.array([[inputs.min(), inputs.max(), inputs.std(), inputs.mean(), inputs.mad(), 
    inputs.skew(), inputs.var(), inputs.kurt(), inputs.autocorr(),
    fft_peaks.autocorr(), scipy.stats.entropy(inputs)]])
    return feat_vector

def unknown_class_correction(indexes, probabilities, clo_list):
    indexes = [i[0] for i in indexes]
    i = 0
    while i < len(indexes):
        if probabilities[i] < 0.7:
            indexes[i] = -99999
        else:
            for missing_elem in clo_list:
                if indexes[i] >= missing_elem:
                    indexes[i] += 1
        i+=1
    return indexes

def generate_true_classes(noepc):
    true_classes = np.empty(16, dtype=object)
    i=0
    while i < 16:
        true_classes[i] = np.full(noepc, i)
        i+=1
    true_classes = np.hstack(true_classes)
    return true_classes

def calculate_accuracy(indexes, true_classes):
    accuracy = sklearn.metrics.accuracy_score(np.array(indexes), true_classes)*100
    return accuracy

def calculate_conf_matrix(indexes, true_classes):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -99999]
    conf_m = str(confusion_matrix(true_classes, indexes, labels))
    return conf_m

def main(argv):
    # Time frame of rows to be used in months, default: 3, now in weeks
    # num_rows = 2016 if len(argv)<1 else int(argv[0])*(24*7*4)
    season = 1 if len(argv)<1 else int(argv[0])
    num_rows = 2016 if len(argv)<2 else int(argv[1])*(24*7)
    dir_list = list(filter(os.path.isdir, glob.glob("*")))
    num_dir = len(dir_list)
    # Training data in percantage(%), default: 70
    t_limit = round((num_dir/100)*70) if len(argv)<3 else round((num_dir/100)*int(argv[2]))
    # Cover threshold for EVM
    cover_threshold = 1 if len(argv)<4 else float(argv[3].replace(",", "."))
    # Class numbers to be left out from training data, values from 0 to 15, default: none
    clo_list = [] if len(argv)<5 else [int(l) for l in argv[4:]]
    clo_list = sorted(clo_list)
    #print(clo_list)
    
    diff_season = season_switch(season) #5*(24*7*4)<- itt kaptam egyszer 100%-ot #745: original(from february)

    random.shuffle(dir_list)
    #init EVM
    mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.euclidean)
    training_classes = np.empty(16, dtype=object)
    test_classes = np.empty(16, dtype=object)

    u = 0
    for subdir in dir_list:
        i = 0
        for file in sorted(glob.glob(os.path.join(subdir, "*.csv"))):
            with open(file, newline='') as csvfile:
                spamreader = pd.read_csv(csvfile, names=['Usage'], usecols=[1],  skiprows=diff_season, nrows=num_rows, sep=',')
                if u<t_limit and i not in clo_list:
                    if training_classes[i] is None:
                        training_classes[i] = calc_feature_vector(spamreader.Usage[:num_rows])
                    else:
                        training_classes[i] = np.append(training_classes[i], calc_feature_vector(spamreader.Usage[:num_rows]), axis=0)
                else:
                    if u>=t_limit:
                        if test_classes[i] is None:
                            test_classes[i] = calc_feature_vector(spamreader.Usage[:num_rows])
                        else:
                            test_classes[i] = np.append(test_classes[i], calc_feature_vector(spamreader.Usage[:num_rows]), axis=0)
            i += 1
        u += 1
    #removing empty elements
    training_classes = np.array([i for i in training_classes if i is not None])
    test_classes = np.array([i for i in test_classes if i is not None])
    #actual training
    mevm.train(training_classes)
    #number of elements per class
    noepc = len(test_classes[0])
    #testing
    test_classes = test_classes.reshape(-1, test_classes.shape[-1])

    probabilities, indexes = mevm.max_probabilities(test_classes)
    indexes = unknown_class_correction(indexes, probabilities, clo_list)
    true_classes = generate_true_classes(noepc)
    print(calculate_accuracy(indexes, true_classes))
    #print(calculate_conf_matrix(indexes, true_classes))


if __name__ == "__main__":
   main(sys.argv[1:])