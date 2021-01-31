import cv2
import numpy as np

imageWithPneumonia1 = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\train\\PNEUMONIA\\person114_virus_217.jpeg"
imageWithPneumonia2 = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\train\\PNEUMONIA\\person115_virus_218.jpeg"
imageWithoutPneumonia1 = "C:\\Users\\thodo\\Downloads\\archive\\chest_xray\\chest_xray\\train\\NORMAL\\IM-0201-0001.jpeg"


def showImage(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (1000, 500))
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # draw the detected key points
    sift_image = cv2.drawKeypoints(gray, keypoints, img)
    # show the image
    cv2.imshow('sift_image', sift_image)
    # save the image
    cv2.imwrite("kaze.jpg", sift_image)
    # create SIFT feature extractor
    kaze = cv2.KAZE_create()
    # detect features from the image
    keypoints, descriptors = kaze.detectAndCompute(img, None)
    # draw the detected key points
    kaze_image = cv2.drawKeypoints(gray, keypoints, img)
    # show the image
    cv2.imshow('Kaze_image', kaze_image)
    # save the image
    cv2.imwrite("kaze.jpg", kaze_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CompareImagesWithSift(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(descriptors1, descriptors2)
    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
    # -- Show detected matches
    cv2.imshow('Matches with SIFT', img_matches)

def CompareImagesWithKAZE(img1, img2):
    kaze = cv2.KAZE_create()
    keypoints1, descriptors1 = kaze.detectAndCompute(img1, None)
    keypoints2, descriptors2 = kaze.detectAndCompute(img2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(descriptors1, descriptors2)
    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
    # -- Show detected matches
    cv2.imshow('Matches with Kaze', img_matches)

def CompareImagesWithKAZEUsingFlannMatcher(img1, img2):
    kaze = cv2.KAZE_create()
    keypoints1, descriptors1 = kaze.detectAndCompute(img1, None)
    keypoints2, descriptors2 = kaze.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # -- Show detected matches
    cv2.imshow('Matches with Kaze using flann matcher', img_matches)
    print('*******************************')
    print('# Keypoints 1:                        \t', len(keypoints1))
    print('# Keypoints 2:                        \t', len(keypoints2))
    print('# Matches:                            \t', len(good_matches))


def CompareImagesWithSIFTUsingFlannMatcher(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # -- Show detected matches
    cv2.imshow('Matches with Sift using flann matcher', img_matches)
    print('*******************************')
    print('# Keypoints 1:                        \t', len(keypoints1))
    print('# Keypoints 2:                        \t', len(keypoints2))
    print('# Matches:                            \t', len(good_matches))

if __name__ == "__main__":
    img1 = cv2.imread(imageWithPneumonia1)
    img1 = cv2.resize(img1, (500, 250))
    img2 = cv2.imread(imageWithPneumonia2)
    img2 = cv2.resize(img2, (500, 250))
    CompareImagesWithKAZE(img1, img2)
    CompareImagesWithSift(img1, img2)
    CompareImagesWithKAZEUsingFlannMatcher(img1, img2)
    CompareImagesWithSIFTUsingFlannMatcher(img1, img2)
    cv2.waitKey()

