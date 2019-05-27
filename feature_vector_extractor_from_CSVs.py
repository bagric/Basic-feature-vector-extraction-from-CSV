import pandas as pd
import numpy as np
import scipy
from scipy.signal import find_peaks
import sys, glob, os, fnmatch

f1 = open('training.txt','w')
f2 = open('test.txt','w')

os.chdir("data")


def calc_feature_vector(inputs):
    fft_form = np.abs(scipy.fftpack.fft(inputs))
    #print(fft_form)
    peaks, _ = find_peaks(fft_form)
    fft_peaks = inputs.iloc[peaks]
    feat_vector = [inputs.min(), inputs.max(), inputs.std(), inputs.mean(), inputs.mad(), 
    inputs.skew(), inputs.var(), inputs.kurt(), inputs.autocorr(),
    fft_peaks.autocorr(), scipy.stats.entropy(inputs)]
    return feat_vector

def main(argv):
    # Time frame of rows to be used in months, default: 3
    num_rows = 2016 if len(argv)<1 else int(argv[0])*(24*7*4)
    dir_list = list(filter(os.path.isdir, glob.glob("*")))
    num_dir = len(dir_list)
    # Training data in percantage(%), default: 70
    t_limit = round((num_dir/100)*70) if len(argv)<2 else round((num_dir/100)*int(argv[1]))
    # Class numbers to be left out from training data, values from 0 to 15, default: none
    clo_list = [] if len(argv)<3 else [int(l) for l in argv[2:]]
    #print(clo_list)

    u = 0
    for subdir in dir_list:
        i = 0
        for file in sorted(glob.glob(os.path.join(subdir, "*.csv"))):
            if u<t_limit and i not in clo_list:
                f1.write(str(i)+' ')
            else:
                f2.write(str(i)+' ')
            with open(file, newline='') as csvfile:
                #print(file)
                spamreader = pd.read_csv(csvfile, names=['Usage'], usecols=[1],  skiprows=745, nrows=num_rows, sep=',')
                if u<t_limit and i not in clo_list:
                    training = calc_feature_vector(spamreader.Usage[:num_rows])
                    for j in range(0,len(training)):
                        f1.write(str(j+1)+':'+str(training[j])+' ')
                    f1.write('\n')
                else:
                    test = calc_feature_vector(spamreader.Usage[:num_rows])
                    for j in range(0,len(test)):
                        f2.write(str(j+1)+':'+str(test[j])+' ')
                    f2.write('\n')
                
            i += 1
        u += 1


if __name__ == "__main__":
   main(sys.argv[1:])
