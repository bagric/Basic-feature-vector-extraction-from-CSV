import EVM, scipy, sys, glob, os, fnmatch, random, sklearn
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from random import randrange

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

#def unknown_class_correction(indexes, probabilities, probability_threshold, clo_list):
def unknown_class_correction(indexes, probabilities, clo_list):
    indexes = [i[0] for i in indexes]
    i = 0
    while i < len(indexes):
        #if probabilities[i] < probability_threshold:
        if probabilities[i] < 0.7:
            indexes[i] = -99999
        else:
            for missing_elem in clo_list:
                if indexes[i] >= missing_elem:
                    indexes[i] += 1
        i+=1
    return indexes

def generate_true_classes(noepc, num_classes):
    true_classes = np.empty(num_classes, dtype=object)
    for i in range(0, num_classes):
        true_classes[i] = np.full(noepc, i)
    true_classes = np.hstack(true_classes)
    return true_classes

def calculate_accuracy(indexes, true_classes):
    accuracy = sklearn.metrics.accuracy_score(np.array(indexes), true_classes)*100
    return accuracy

def calculate_conf_matrix(indexes, true_classes):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -99999]
    conf_m = str(confusion_matrix(true_classes, indexes, labels))
    return conf_m

def read_data(dir_list, diff_season, num_rows, num_classes):
    data_classes = np.empty(num_classes, dtype=object)
    u = 0
    for subdir in dir_list:
        i = 0
        for file in sorted(glob.glob(os.path.join(subdir, "*.csv"))):
            with open(file, newline='') as csvfile:
                spamreader = pd.read_csv(csvfile, names=['Usage'], usecols=[1],  skiprows=diff_season, nrows=num_rows, sep=',')
                if data_classes[i] is None:
                    data_classes[i] = calc_feature_vector(spamreader.Usage[:num_rows])
                else:
                    data_classes[i] = np.append(data_classes[i], calc_feature_vector(spamreader.Usage[:num_rows]), axis=0)
            i += 1
        u += 1
    return data_classes

def shuffle_array(data_classes, num_classes):
    for i in range(0, num_classes):
        np.random.shuffle(data_classes[i])
    return data_classes

def split_into_training_test_sets(data_classes, num_classes, t_limit):
    # Creating training and test arrays
    training_classes = np.empty(num_classes, dtype=object)
    test_classes = np.empty(num_classes, dtype=object)
    for i in range(0, num_classes):
        training_classes[i], test_classes[i] = data_classes[i][:t_limit,:], data_classes[i][t_limit:,:]
    return training_classes, test_classes

def main_for_caller(season, num_weeks, tperc, cover_threshold, clo_list):
    # Number of classes
    num_classes = 16
    # Select season
    diff_season = season_switch(season)
    # Directory list where the data are
    dir_list = list(filter(os.path.isdir, glob.glob("*")))
    # Time frame of rows to be used in months, default: 3, now in weeks
    # num_rows = 2016 if len(argv)<1 else int(argv[0])*(24*7*4)
    num_rows = num_weeks*(24*7)
    if num_weeks==1:
        diff_season=diff_season+(randrange(8)*24*7)

    # Reading the files only once
    #data_classes_ft = read_data(dir_list, season_switch(2), num_rows, num_classes) # diff season for training
    data_classes = read_data(dir_list, diff_season, num_rows, num_classes)

    #number of elements per class
    noepc = len(data_classes[0])
    # Training data in percantage(%), default: 70
    t_limit = round((noepc/100)*tperc)
    # Class numbers to be left out from training data, values from 0 to 15, default: none
    clo_list = sorted(clo_list)
    # probability_threshold
    #probability_threshold = 0.6
    # Starting experiment loop
    #for i in range(0, 3):
    for i in range(0, 100):
        # Shuffling data every iteration
        #data_classes_ft = shuffle_array(data_classes_ft, num_classes)
        data_classes = shuffle_array(data_classes, num_classes)

        # Splitting data to training set and test set
        training_classes, test_classes = split_into_training_test_sets(data_classes, num_classes, t_limit)
        #training_classes, _ = split_into_training_test_sets(data_classes_ft, num_classes, t_limit)
        #_, test_classes = split_into_training_test_sets(data_classes, num_classes, t_limit)

        # Removing classes from training
        training_classes = np.delete(training_classes, clo_list, axis=0)
        # Removing empty elements
        training_classes = np.array([i for i in training_classes if i is not None])
        test_classes = np.array([i for i in test_classes if i is not None])
        
        # Init EVM
        #mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.euclidean)
        ##mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.seuclidean)
        #mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.sqeuclidean)
        #mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.cosine)
        #mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.jensenshannon)
        mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.chebyshev)

        # Actual training
        mevm.train(training_classes)
        #number of elements per class in test
        noepct = len(test_classes[0])
        # Generating true class array
        true_classes = generate_true_classes(noepct, num_classes)
        # Reshaping test_classes for mevm testing
        test_classes = test_classes.reshape(-1, test_classes.shape[-1])
        # Actual testing
        probabilities, indexes = mevm.max_probabilities(test_classes)
        # Correction
        #indexes = unknown_class_correction(indexes, probabilities, probability_threshold, clo_list)
        #probability_threshold = probability_threshold + 0.1
        indexes = unknown_class_correction(indexes, probabilities, clo_list)
        with open("evm_res_multi_classes_missing.txt", "a") as myfile:
            myfile.write(str(calculate_accuracy(indexes, true_classes)))
            myfile.write('\n')
        #print(calculate_accuracy(indexes, true_classes))
        #print(calculate_conf_matrix(indexes, true_classes))

def main(argv):
    # Number of classes
    num_classes = 16
    # Select season
    season = 1 if len(argv)<1 else int(argv[0])
    diff_season = season_switch(season)
    # Directory list where the data are
    dir_list = list(filter(os.path.isdir, glob.glob("*")))
    # Time frame of rows to be used in months, default: 3, now in weeks
    # num_rows = 2016 if len(argv)<1 else int(argv[0])*(24*7*4)
    num_rows = 2016 if len(argv)<2 else int(argv[1])*(24*7)
    # Reading the files only once
    data_classes = read_data(dir_list, diff_season, num_rows, num_classes)
    #number of elements per class
    noepc = len(data_classes[0])
    # Training data in percantage(%), default: 70
    t_limit = round((noepc/100)*70) if len(argv)<3 else round((noepc/100)*int(argv[2]))
    # Cover threshold for EVM
    cover_threshold = 1 if len(argv)<4 else float(argv[3].replace(",", "."))
    # Class numbers to be left out from training data, values from 0 to 15, default: none
    clo_list = [] if len(argv)<5 else [int(l) for l in argv[4:]]
    clo_list = sorted(clo_list)
    # Creating/opening file to write results into
    f = open("evm_res_multi_classes_missing.txt", "w")
    # Starting experiment loop
    for i in range(0, 10):
        # Shuffling data every iteration
        data_classes = shuffle_array(data_classes, num_classes)
        # Splitting data to training set and test set
        training_classes, test_classes = split_into_training_test_sets(data_classes, num_classes, t_limit)
        # Removing classes from training
        training_classes = np.delete(training_classes, clo_list, axis=0)
        # Removing empty elements
        training_classes = np.array([i for i in training_classes if i is not None])
        test_classes = np.array([i for i in test_classes if i is not None])
        # Init EVM
        mevm = EVM.MultipleEVM(tailsize=0, cover_threshold = cover_threshold, distance_function=scipy.spatial.distance.euclidean)
        # Actual training
        mevm.train(training_classes)
        #number of elements per class in test
        noepct = len(test_classes[0])
        # Generating true class array
        true_classes = generate_true_classes(noepct, num_classes)
        # Reshaping test_classes for mevm testing
        test_classes = test_classes.reshape(-1, test_classes.shape[-1])
        # Actual testing
        probabilities, indexes = mevm.max_probabilities(test_classes)
        # Correction
        indexes = unknown_class_correction(indexes, probabilities, clo_list)
        f.write(str(calculate_accuracy(indexes, true_classes)))
        f.write('\n')
        #print(calculate_accuracy(indexes, true_classes))
        #print(calculate_conf_matrix(indexes, true_classes))
    f.close()


if __name__ == "__main__":
   main(sys.argv[1:])