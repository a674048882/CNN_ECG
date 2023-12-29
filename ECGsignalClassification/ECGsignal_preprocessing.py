import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# wavelet denoise preprocess using mallat algorithm
def denoise(data):
    # wavelet decomposition
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # denoise using soft threshold
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # get the denoised signal by inverse wavelet transform
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# load the ecg data and the corresponding labels, then denoise the data using wavelet transform
def get_data_set(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # load the ecg data record
    print("loading the ecg data of No." + number)
    record = wfdb.rdrecord('mit-bih-ecg-data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # get the positions of R-wave and the corresponding labels
    annotation = wfdb.rdann('mit-bih-ecg-data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # remove the unstable data at the beginning and the end
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # the data with specific labels (N/A/V/L/R) required in this record are selected, and the others are discarded
    # X_data: data points of length 300 around the R-wave
    # Y_data: convert N/A/V/L/R to 0/1/2/3/4 in order
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# load dataset and preprocess
def load_data(ratio, random_seed):
    # Total 45 patients data 97826
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        get_data_set(n, dataSet, lableSet)

    # reshape the data and split the dataset
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(dataSet, lableSet, test_size=ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test


plt.rc('font', family='Times New Roman')


# confusion matrix
def plot_heat_map(y_test, y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # plot
    plt.figure(figsize=(8, 8))
    plt.tick_params(labelsize=12)
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Reds')
    plt.ylim(0, 5)
    plt.title('Confusion Matrix', font={'family': 'Times New Roman', 'size': 18})
    plt.xlabel('Predicted labels', font={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('True labels', font={'family': 'Times New Roman', 'size': 16})
    plt.savefig('Confusion_matrix.png')
    plt.show()


# draw accuracy and loss figure of the model.
def plot_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 10))
    plt.tick_params(labelsize=20)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy', font={'family': 'Times New Roman', 'size': 24})
    plt.ylabel('Accuracy(*100%)', font={'family': 'Times New Roman', 'size': 22})
    plt.xlabel('Epochs(1)', font={'family': 'Times New Roman', 'size': 22})
    plt.rcParams.update({'font.size': 16})
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig('Accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.tick_params(labelsize=20)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss', font={'family': 'Times New Roman', 'size': 24})
    plt.ylabel('Loss(*100%)', font={'family': 'Times New Roman', 'size': 22})
    plt.xlabel('Epochs(1)', font={'family': 'Times New Roman', 'size': 22})
    plt.rcParams.update({'font.size': 16})
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('Loss.png')
    plt.show()


