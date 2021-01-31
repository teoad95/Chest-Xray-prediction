import numpy as np # linear algebra
import os # reading data
import cv2 # reading images
from sklearn.preprocessing import MinMaxScaler
import pickle as cpickle # store data for fast processing


trainDataDir = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\train"
testDataDir = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\test"
categories = ["NORMAL", "PNEUMONIA"]
training_data = {}
testing_data = {}
i = 0

def extract_features(image_path, vector_size=32):
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image_array, (100, 100))
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
    except cv2.error as e:
        print('Error: ' + e)
        return None
    return dsc


def LoadData(dictionaryToStoreData, dataDir, pickleName):
    # load training data
    for category in categories:
        path = os.path.join(dataDir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            name = img.split('/')[-1].lower()
            try:
                features = extract_features(os.path.join(path, img), 32)
                dictionaryToStoreData[name] = [features, class_num]
            except:
                print("An exception occurred while extracting features from image " + name)
    # saving all our feature vectors in pickled file
    with open(pickleName + '.pickle', 'wb') as fp:
        cpickle.dump(dictionaryToStoreData, fp)


if __name__ == "__main__":
    # LoadData(training_data, trainDataDir, 'trainingData')
    # LoadData(testing_data, testDataDir, 'testingData')
    training_data = cpickle.load(open('trainingData.pickle', 'rb'))
    testing_data = cpickle.load(open('testingData.pickle', 'rb'))
    x_Train = list(td[0] for td in training_data.values())
    y_Train = list(td[1] for td in training_data.values())
    x_Test = list(td[0] for td in testing_data.values())
    y_Test = list(td[1] for td in testing_data.values())
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

