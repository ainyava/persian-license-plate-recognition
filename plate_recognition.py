import cv2
import numpy as np

IMAGE_WIDTH = 70
IMAGE_HEIGHT = 70
SHOW_STEPS = False

# load trained data from images and classifications
npa_classifications = np.loadtxt("classifications.txt", np.float32)
npa_flattenedImages = np.loadtxt("flattened_images.txt", np.float32)
# create KNN object
k_nearest = cv2.ml.KNearest_create()
# train KNN object with training data
k_nearest.train(npa_flattenedImages, cv2.ml.ROW_SAMPLE, npa_classifications)

# load plate image
img = cv2.imread('plate.jpg')
# apply a blur to make edges thick
img = cv2.GaussianBlur(img, (5, 5), 0)
# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a threshhold to make image black & white
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# find contours on the black & white image
contours, hir = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# validating characters
# Note: Since in Farsi characters we have characters with more than one characters (like chars with dots)
# so we need to validate that they are characters or not 
valid_contours = []
for c in contours:
    [_, _, w, h] = cv2.boundingRect(c)
    if w * h >= 400:
        valid_contours.append(c)

# cv2.drawContours(img, valid_contours, -1, (0, 255, 0), 2)

# sort characters by their bouding size
valid_contours = sorted(valid_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# loop through the valid contours (characters)
for c in valid_contours:
    # store boundings
    [x, y, w, h] = cv2.boundingRect(c)
    # check for dots or etc
    # Method: check anything in Y axis if their boundings are in X axis
    # Explain: cause dots in Arabic/Farsi characters are above and below the character itself
    for c2 in contours:
        [x2, y2, w2, h2] = cv2.boundingRect(c2)
        if x2 >= x and x2 <= x + w:
            if y2 < y:
                h += abs(y - y2)
                y = y2
            if y2 > y + h:
                h += abs(y2 - (y + h))

    # draw rectangle around the characters
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # crop the character from grayscale image
    char_image = gray[y:y + h, x:x + w]
    # resize character to training size so we could compare those
    char_image = cv2.resize(char_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # reshape resized character to numpy array
    npa = np.float32(char_image.reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT)))

    # store results of compare between character and training data and find nearest possible example
    retval, npaResults, neigh_resp, dists = k_nearest.findNearest(npa, k=1)

    # convert the result found in classifications to normal character and print it on output
    print(chr(int(npaResults[0][0])), end='')

    if SHOW_STEPS:
        cv2.imshow('test', char_image)
        cv2.waitKey(0)

if SHOW_STEPS:
    cv2.imshow('test', gray)
    cv2.imshow('test2', img)
    cv2.waitKeyEx(0)
