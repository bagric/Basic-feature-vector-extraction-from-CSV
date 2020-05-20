import pandas as pd
import numpy as np
import scipy
from scipy.signal import find_peaks
import sys, glob, os, fnmatch
import random

f1 = open('training.txt','w')
f2 = open('test.txt','w')
#f3 = open('autocorr.txt','w')

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
    #print(fft_form)
    peaks, _ = find_peaks(fft_form)
    fft_peaks = inputs.iloc[peaks]
    feat_vector = [inputs.min(), inputs.max(), inputs.std(), inputs.mean(), inputs.mad(), 
    inputs.skew(), inputs.var(), inputs.kurt(), inputs.autocorr(),
    fft_peaks.autocorr(), scipy.stats.entropy(inputs)]
    return feat_vector

def main(argv):
    # Time frame of rows to be used in months, default: 3, now in weeks
    # num_rows = 2016 if len(argv)<1 else int(argv[0])*(24*7*4)
    season = 1 if len(argv)<1 else int(argv[0])
    num_rows = 2016 if len(argv)<2 else int(argv[1])*(24*7)
    dir_list = list(filter(os.path.isdir, glob.glob("*")))
    num_dir = len(dir_list)
    # Training data in percantage(%), default: 70
    t_limit = round((num_dir/100)*70) if len(argv)<3 else round((num_dir/100)*int(argv[2]))
    # Class numbers to be left out from training data, values from 0 to 15, default: none
    clo_list = [] if len(argv)<3 else [int(l) for l in argv[3:]]
    #print(clo_list)
    
    diff_season = season_switch(season) #5*(24*7*4)<- itt kaptam egyszer 100%-ot #745: original(from february)

    random.shuffle(dir_list)

    u = 0
    for subdir in dir_list:
        i = 0
        for file in sorted(glob.glob(os.path.join(subdir, "*.csv"))):
            if u<t_limit and i not in clo_list:
                f1.write(str(i)+' ')

                #f3.write(str(i)+' ')

            else:
                if u>=t_limit:
                    f2.write(str(i)+' ')
            with open(file, newline='') as csvfile:
                #print(file)
                spamreader = pd.read_csv(csvfile, names=['Usage'], usecols=[1],  skiprows=diff_season, nrows=num_rows, sep=',')
                if u<t_limit and i not in clo_list:
                    training = calc_feature_vector(spamreader.Usage[:num_rows])

                    #f3.write(str(training[len(training)-3])+', ')
                    #f3.write('\n')

                    for j in range(0,len(training)):
                        f1.write(str(j+1)+':'+str(training[j])+' ')
                    f1.write('\n')
                else:
                    if u>=t_limit:
                        test = calc_feature_vector(spamreader.Usage[:num_rows])
                        for j in range(0,len(test)):
                            f2.write(str(j+1)+':'+str(test[j])+' ')
                        f2.write('\n')
                
            i += 1
        u += 1


if __name__ == "__main__":
   main(sys.argv[1:])