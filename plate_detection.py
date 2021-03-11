import cv2
import imutils
import numpy as np

# loop over the dataset images
for i in range(1, 20):
    # read and resize image
    img = cv2.imread(f'dataset/{i}.jpg')
    img = imutils.resize(img, width=800)
    # convert to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply a blur to make contours more thick
    filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    # perform canny edge detection
    edged = cv2.Canny(filtered, 10, 100) 
    # find coutours on edge detected image
    contours, hir = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by their area size so we search in biggest shapres first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate = False
    # loop over our contours
    for c in contours:
        # approximate the contour
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        # if our approximated contour has four points
        # and have big enough area, then
        # we can assume that we have found license plate
        if len(approx) == 4 and cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            if 2.5 < w / h < 4.1:
                plate = True
                cv2.drawContours(img, c, -1, (0, 255, 0), 3)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                break
    
    # if we still could'nt find the plate we keep looking with some diffrent rules
    if not plate:
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
            # if our approximated contour has more than four points
            # then we check for its radio and if its between 2.5 and 4.5
            # we could use that as plate (but its not accurate and needs work)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(c)
                if 2.5 < w / h < 4.5 and 10000 <= (w * h):
                    # b, g, r = cv2.split(img[y:y + h, x - 20:x])
                    plate = True
                    cv2.drawContours(img, c, -1, (0, 255, 0), 1)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    break

    # show the plate image
    cv2.imshow('image', img)
    cv2.imshow('edged', edged)

    # if we found any plates then crop and save that part
    if plate:
        cropped = img[y+10:y+h-10, x+10:x+w-10]
        cv2.imshow('plate', cropped)
        cv2.imwrite('plate.jpg', cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
