import numpy as np # linear algebra
import os # reading data
import cv2 # reading images
from sklearn.preprocessing import MinMaxScaler
import pickle as cpickle # store data for fast processing


trainDataDir = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\train"
testDataDir = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\test"
validateDataDir = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\val"
categories = ["NORMAL", "PNEUMONIA"]
training_data = {}
testing_data = {}
validate_data = {}
i = 0


def Get_Kaze_features(image, vector_size=32):
    try:
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return dsc
    except cv2.error as e:
        print('Error: ' + e)
        return None

def Get_Hog_Features(image):
    try:
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        dsc = hog.compute(image, winStride, padding, locations)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        return dsc
    except cv2.error as e:
        print('Error: ' + e)
        return None



def extract_features(image_path, extractFeaturesUsing = ''):
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image_array, (100, 100))
    if extractFeaturesUsing == '':
        return image.flatten()
    elif extractFeaturesUsing == 'KAZE':
        return Get_Kaze_features(image)
    elif extractFeaturesUsing == 'HOG':
        return Get_Hog_Features(image)
    return None


def LoadData(dictionaryToStoreData, dataDir, pickleName, extractFeaturesUsing = ''):
    # load training data
    for category in categories:
        path = os.path.join(dataDir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            name = img.split('/')[-1].lower()
            try:
                imageLocation = os.path.join(path, img)
                features = extract_features(imageLocation, extractFeaturesUsing)
                dictionaryToStoreData[imageLocation] = [features, class_num]
            except:
                print("An exception occurred while extracting features from image " + name)
    # saving all our feature vectors in pickled file
    with open(pickleName + '.pickle', 'wb') as fp:
        cpickle.dump(dictionaryToStoreData, fp)


def ShowValidationImagesAndResults(model):
    global i
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(100, 100))
    columns = 8
    rows = 2
    i = 1
    for file, value in validate_data.items():
        img = cv2.imread(file)
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
        category = file.split('\\')[8].lower()
        predicted_value = model.predict(value[0].reshape(1, -1))
        color = 'green'
        if predicted_value != value[1]:
            color = 'red'
        ax.set_title(category, color=color)
        i = i + 1
    plt.show()


if __name__ == "__main__":
    #LoadData(training_data, trainDataDir, 'trainingDataUsingHog', 'HOG')
    #LoadData(testing_data, testDataDir, 'testingDataUsingHog', 'HOG')
    #LoadData(validate_data, validateDataDir, 'validateDataUsingKaze', 'KAZE')

    training_data = cpickle.load(open('trainingDataUsingKaze.pickle', 'rb'))
    testing_data = cpickle.load(open('testingDataUsingKaze.pickle', 'rb'))
    validate_data = cpickle.load(open('validateDataUsingKaze.pickle', 'rb'))
    x_Train = list(td[0] for td in training_data.values())
    y_Train = list(td[1] for td in training_data.values())
    x_Test = list(td[0] for td in testing_data.values())
    y_Test = list(td[1] for td in testing_data.values())
    x_Validate = list(td[0] for td in validate_data.values())
    y_Validate = list(td[1] for td in validate_data.values())

    # apply scaling
    scaler = MinMaxScaler()
    x_Train = scaler.fit_transform(x_Train)
    x_Test = scaler.transform(x_Test)

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(x_Train, y_Train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
           .format(logreg.score(x_Train, y_Train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
           .format(logreg.score(x_Test, y_Test)))
    print('Accuracy of Logistic regression classifier on validate set: {:.2f}'
           .format(logreg.score(x_Validate, y_Validate)))

    ShowValidationImagesAndResults(logreg)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(x_Train, y_Train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(x_Train, y_Train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(x_Test, y_Test)))
    print('Accuracy of K-NN classifier on validate set: {:.2f}'
          .format(knn.score(x_Validate, y_Validate)))

    ShowValidationImagesAndResults(knn)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_Train, y_Train)
    print('Accuracy of LDA classifier on training set: {:.2f}'
          .format(lda.score(x_Train, y_Train)))
    print('Accuracy of LDA classifier on test set: {:.2f}'
          .format(lda.score(x_Test, y_Test)))
    print('Accuracy of LDA classifier on validate set: {:.2f}'
          .format(lda.score(x_Validate, y_Validate)))

    ShowValidationImagesAndResults(lda)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(x_Train, y_Train)
    print('Accuracy of GNB classifier on training set: {:.2f}'
          .format(gnb.score(x_Train, y_Train)))
    print('Accuracy of GNB classifier on test set: {:.2f}'
          .format(gnb.score(x_Test, y_Test)))
    print('Accuracy of GNB classifier on validate set: {:.2f}'
          .format(gnb.score(x_Validate, y_Validate)))
    ShowValidationImagesAndResults(gnb)

    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(x_Train, y_Train)
    print('Accuracy of SVM classifier on training set: {:.2f}'
          .format(svm.score(x_Train, y_Train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
          .format(svm.score(x_Test, y_Test)))
    print('Accuracy of SVM classifier on validate set: {:.2f}'
          .format(svm.score(x_Validate, y_Validate)))
    ShowValidationImagesAndResults(svm)



