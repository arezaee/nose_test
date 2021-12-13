import cv2
import numpy as np
import dlib
import math
import argparse


def fx(pt1, pt2, pt3, pt4, x):
    if x < pt1[0] or x > pt4[0]:
        return pt1[1]

    if x == pt1[0]:
        p0 = pt1
        p1 = pt4
    elif x > pt1[0] and x <= pt2[0]:
        p0 = pt1
        p1 = pt2
    elif x > pt2[0] and x <= pt3[0]:
        p0 = pt2
        p1 = pt3
    elif x > pt3[0] and x <= pt4[0]:
        p0 = pt3
        p1 = pt4
        

    return int(((p1[1]-p0[1])/(p1[0]-p0[0]))*(x-p0[0])+p0[1])


def strech_top_transform(img, left_pt, right_pt, old_top1_pt, old_top2_pt, new_top1_pt, new_top2_pt):
    # print(left_pt, right_pt, old_top1_pt,
    #       old_top2_pt, new_top1_pt, new_top2_pt)
    result = img.copy()
    for x in range(left_pt[0], right_pt[0]+1):
        basey = fx(left_pt, left_pt, right_pt, right_pt, x)
        oldy = fx(left_pt, old_top1_pt, old_top2_pt, right_pt, x)
        newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, x)

        if basey == oldy:
            continue

        if basey <= newy:
            newy = basey-1

        if oldy >= newy:
            oldy = newy-1

        yprim = int(oldy - 2*(newy-oldy))

        im1 = img[yprim:oldy, x:x+1]
        im1 = cv2.resize(im1, (1, newy-yprim),
                         interpolation=cv2.INTER_NEAREST)
        result[yprim:yprim+im1.shape[0], x:x+im1.shape[1]] = im1
        im1 = img[oldy:basey, x:x+1]
        im1 = cv2.resize(
            im1, (1, basey-newy), interpolation=cv2.INTER_NEAREST)
        result[basey-im1.shape[0]:basey, x:x+im1.shape[1]] = im1

    oldy = fx(left_pt, old_top1_pt, old_top2_pt, right_pt, old_top1_pt[0])
    newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, old_top1_pt[0])
    left_yprim = (left_pt[0], int(oldy - 3*(newy-oldy)))

    oldy = fx(left_pt, old_top1_pt, old_top2_pt, right_pt, old_top2_pt[0])
    newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, old_top2_pt[0])
    right_yprim = (right_pt[0], int(oldy - 3*(newy-oldy)))

    roi = np.array([left_pt, left_yprim, right_yprim, right_pt])

    return roi_blur(img, result, roi)


def strech_top2_transform(img, left_pt, right_pt, old_top1_pt, old_top2_pt, new_top1_pt, new_top2_pt):
    # print(left_pt, right_pt, old_top1_pt,
    #       old_top2_pt, new_top1_pt, new_top2_pt)
    result = img.copy()
    for x in range(left_pt[0], right_pt[0]+1):
        basey = fx(left_pt, left_pt, right_pt, right_pt, x)
        oldy = fx(left_pt, old_top1_pt, old_top2_pt, right_pt, x)
        newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, x)
        # print(f"old_top1_pt {old_top1_pt}, old_top2_pt {old_top2_pt}")
        # print(f"new_top1_pt {new_top1_pt}, new_top2_pt {new_top2_pt}")
        # print(f"x {x}, newy {newy}, oldy {oldy}, basey {basey}")

        if basey == oldy:
            continue

        # if basey <= newy:
        #     newy = basey-1

        if oldy <= newy:
            newy = oldy-1

        yprim = int(oldy - 2*(oldy-newy))
        # print(f"x {x}, yprim, {yprim}, newy {newy}, oldy {oldy}, basey {basey}")

        im1 = img[yprim:oldy, x:x+1]
        # print(f"p1, {im1.shape}")
        im1 = cv2.resize(im1, (1, newy-yprim),
                         interpolation=cv2.INTER_NEAREST)
        # print(f"p2, {im1.shape}")

        result[yprim:yprim+im1.shape[0], x:x+im1.shape[1]] = im1
        im1 = img[oldy:basey, x:x+1]
        # print(f"p3, {im1.shape}")

        im1 = cv2.resize(
            im1, (1, basey-newy), interpolation=cv2.INTER_NEAREST)
        result[basey-im1.shape[0]:basey, x:x+im1.shape[1]] = im1
        # print(f"p4, {im1.shape}")

    # return result
    oldy = fx(left_pt, old_top1_pt, old_top2_pt, right_pt, old_top1_pt[0])
    newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, old_top1_pt[0])
    left_yprim = (left_pt[0], int(oldy - 3*(oldy - newy)))

    oldy = fx(left_pt, old_top1_pt, old_top2_pt, right_pt, old_top2_pt[0])
    newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, old_top2_pt[0])
    right_yprim = (right_pt[0], int(oldy - 3*(oldy-newy)))

    roi = np.array([left_pt, left_yprim, right_yprim, right_pt])
    # img = cv2.polylines(img, [roi], True, ( 0,255, 0), 2)

    return roi_blur(img, result, roi)


def strech_bottom_transform(img, left_pt, right_pt, old_bottom_pt, old_bottom2_pt, new_top1_pt, new_top2_pt):
    # print(left_pt, right_pt, old_bottom_pt,
    #       old_bottom2_pt, new_top1_pt, new_top2_pt)
    result = img.copy()
    for x in range(left_pt[0], right_pt[0]+1):
        basey = fx(left_pt, left_pt, right_pt, right_pt, x)
        oldy = fx(left_pt, old_bottom_pt, old_bottom2_pt, right_pt, x)
        newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, x)

        if basey == oldy:
            continue

        if basey >= newy:
            newy = basey+1

        if oldy <= newy:
            oldy = newy+1

        yprim = int(oldy - 2*(newy-oldy))

        im1 = img[oldy:yprim, x:x+1]
        im1 = cv2.resize(im1, (1, yprim-newy),
                         interpolation=cv2.INTER_NEAREST)
        result[yprim-im1.shape[0]:yprim, x:x+im1.shape[1]] = im1
        im1 = img[basey:oldy, x:x+1]
        im1 = cv2.resize(
            im1, (1, newy-basey), interpolation=cv2.INTER_NEAREST)
        result[basey:basey+im1.shape[0], x:x+im1.shape[1]] = im1

    oldy = fx(left_pt, old_bottom_pt, old_bottom2_pt, right_pt, old_bottom_pt[0])
    newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, old_bottom_pt[0])
    left_yprim = (left_pt[0], int(oldy - 3*(newy-oldy)))

    oldy = fx(left_pt, old_bottom_pt, old_bottom2_pt, right_pt, old_bottom2_pt[0])
    newy = fx(left_pt, new_top1_pt, new_top2_pt, right_pt, old_bottom2_pt[0])
    right_yprim = (right_pt[0], int(oldy - 3*(newy-oldy)))

    roi = np.array([left_pt, left_yprim, right_yprim, right_pt])

    return roi_blur(img, result, roi)


def strech_bottom2_transform(img, left_pt, right_pt, old_bottom1_pt, old_bottom2_pt, new_bottom1_pt, new_bottom2_pt):
    # print(left_pt, right_pt, old_bottom1_pt,
        #   old_bottom2_pt, new_bottom1_pt, new_bottom2_pt)
    result = img.copy()
    for x in range(left_pt[0], right_pt[0]+1):
        basey = fx(left_pt, left_pt, right_pt, right_pt, x)
        oldy = fx(left_pt, old_bottom1_pt, old_bottom2_pt, right_pt, x)
        newy = fx(left_pt, new_bottom1_pt, new_bottom2_pt, right_pt, x)
        # print(f"old_bottom1_pt {old_bottom1_pt}, old_bottom2_pt {old_bottom2_pt}")
        # print(f"new_bottom1_pt {new_bottom1_pt}, new_bottom2_pt {new_bottom2_pt}")
        # print(f"x {x}, newy {newy}, oldy {oldy}, basey {basey}")

        if basey == oldy:
            continue

        # if basey <= newy:
        #     newy = basey-1

        if oldy >= newy:
            newy = oldy+1

        yprim = int(oldy - 2*(oldy-newy))
        # print(f"x {x}, yprim, {yprim}, newy {newy}, oldy {oldy}, basey {basey}")

        im1 = img[oldy:yprim, x:x+1]
        # print(f"p1, {im1.shape}")
        im1 = cv2.resize(im1, (1, yprim-newy),
                         interpolation=cv2.INTER_NEAREST)
        # print(f"p2, {im1.shape}")

        result[yprim-im1.shape[0]:yprim, x:x+im1.shape[1]] = im1
        im1 = img[basey:oldy, x:x+1]
        # print(f"p3, {im1.shape}")

        im1 = cv2.resize(
            im1, (1, newy-basey), interpolation=cv2.INTER_NEAREST)
        result[basey:basey+im1.shape[0], x:x+im1.shape[1]] = im1
        # print(f"p4, {im1.shape}")

    # return result
    oldy = fx(left_pt, old_bottom1_pt, old_bottom2_pt, right_pt, old_bottom1_pt[0])
    newy = fx(left_pt, new_bottom1_pt, new_bottom2_pt, right_pt, old_bottom1_pt[0])
    left_yprim = (left_pt[0], int(oldy - 3*(oldy - newy)))
    # print(f"oldy {oldy} newy {newy} left_yprim {left_yprim}")

    oldy = fx(left_pt, old_bottom1_pt, old_bottom2_pt, right_pt, old_bottom2_pt[0])
    newy = fx(left_pt, new_bottom1_pt, new_bottom2_pt, right_pt, old_bottom2_pt[0])
    right_yprim = (right_pt[0], int(oldy - 3*(oldy-newy)))
    # print(f"oldy {oldy} newy {newy} right_yprim {right_yprim}")

    roi = np.array([left_pt, left_yprim, right_yprim, right_pt])
    # img = cv2.polylines(img, [roi], True, (0, 255, 0), 2)
    
    # print(f"roi {roi}")

    return roi_blur(img, result, roi)


def fy(pt1, pt2, y):
    if pt1[1] == pt2[1]:
        return pt1[1]

    y0 = pt2[1]
    if y < y0:
        return int(((pt1[0]-pt2[0])/(pt1[1]-pt2[1]))*(y-pt2[1])+pt2[0])
    else:
        return int(((pt1[0]-pt2[0])/(pt1[1]-pt2[1]))*(y0-pt2[1])+pt2[0])


def strech_right_transform(img, top_pt, bottom_pt, old_top_pt, old_mid_pt, new_top_pt, new_mid_pt):
    result = img.copy()
    for y in range(top_pt[1], bottom_pt[1]+1):
        basex = fy(top_pt, bottom_pt, y)
        oldx = fy(old_top_pt, old_mid_pt, y)
        newx = fy(new_top_pt, new_mid_pt, y)
        if oldx == newx or basex == newx:
            continue
        xprim = int(oldx + 1*(oldx - newx))

        im1 = img[y:y+1, oldx:xprim]
        im1 = cv2.resize(im1, (xprim-newx, 1),
                         interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], newx:newx+im1.shape[1]] = im1

        im1 = img[y:y+1, basex:oldx]
        im1 = cv2.resize(
            im1, (newx-basex, 1), interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], basex:basex+im1.shape[1]] = im1

    oldx = fy(old_top_pt, old_mid_pt, top_pt[1])
    newx = fy(new_top_pt, new_mid_pt, top_pt[1])
    top_xprim = (int(oldx + 2*(oldx - newx)), top_pt[1])
    
    bottom_pt = (bottom_pt[0] , bottom_pt[1] + 5)
    oldx = fy(old_top_pt, old_mid_pt, bottom_pt[1])
    newx = fy(new_top_pt, new_mid_pt, bottom_pt[1])
    bottom_xprim = (int(oldx + 2*(oldx - newx)), bottom_pt[1])
    roi = np.array([top_pt, top_xprim, bottom_xprim, bottom_pt])

    return roi_blur(img, result, roi)


def strech_left_transform(img, top_pt, bottom_pt, old_top_pt, old_mid_pt, new_top_pt, new_mid_pt):
    result = img.copy()
    for y in range(top_pt[1], bottom_pt[1]+1):
        basex = fy(top_pt, bottom_pt, y)
        oldx = fy(old_top_pt, old_mid_pt, y)
        newx = fy(new_top_pt, new_mid_pt, y)
        if oldx == newx or basex == newx:
            continue
        xprim = int(oldx + 1*(oldx - newx))

        im1 = img[y:y+1, xprim:oldx]
        im1 = cv2.resize(im1, (newx-xprim, 1),
                         interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], newx-im1.shape[1]:newx] = im1

        im1 = img[y:y+1, oldx:basex]
        im1 = cv2.resize(
            im1, (basex-newx, 1), interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], basex-im1.shape[1]:basex] = im1

    oldx = fy(old_top_pt, old_mid_pt, top_pt[1])
    newx = fy(new_top_pt, new_mid_pt, top_pt[1])
    top_xprim = (int(oldx + 2*(oldx - newx)), top_pt[1])

    bottom_pt = (bottom_pt[0] , bottom_pt[1] + 5)
    oldx = fy(old_top_pt, old_mid_pt, bottom_pt[1])
    newx = fy(new_top_pt, new_mid_pt, bottom_pt[1])
    bottom_xprim = (int(oldx + 2*(oldx - newx)), bottom_pt[1])
    roi = np.array([top_pt, top_xprim, bottom_xprim, bottom_pt])

    return roi_blur(img, result, roi)


def strech_right2_transform(img, top_pt, bottom_pt, old_top_pt, old_mid_pt, new_top_pt, new_mid_pt):
    result = img.copy()
    for y in range(top_pt[1], bottom_pt[1]+1):
        basex = fy(top_pt, bottom_pt, y)
        oldx = fy(old_top_pt, old_mid_pt, y)
        newx = fy(new_top_pt, new_mid_pt, y)
        xprim = int(newx + 1*(newx - oldx))

        if oldx == newx or basex == newx or newx == xprim:
            continue

        im1 = img[y:y+1, oldx:xprim]
        im1 = cv2.resize(im1, (xprim-newx, 1),
                         interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], newx:newx+im1.shape[1]] = im1

        im1 = img[y:y+1, basex:oldx]
        im1 = cv2.resize(
            im1, (newx-basex, 1), interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], basex:basex+im1.shape[1]] = im1

    oldx = fy(old_top_pt, old_mid_pt, top_pt[1])
    newx = fy(new_top_pt, new_mid_pt, top_pt[1])
    top_xprim = (int(newx + 2*(newx-oldx)), top_pt[1])

    bottom_pt = (bottom_pt[0] , bottom_pt[1] + 5)
    oldx = fy(old_top_pt, old_mid_pt, bottom_pt[1])
    newx = fy(new_top_pt, new_mid_pt, bottom_pt[1])
    bottom_xprim = (int(newx + 2*(newx-oldx)), bottom_pt[1])
    roi = np.array([top_pt, top_xprim, bottom_xprim, bottom_pt])

    return roi_blur(img, result, roi)


def strech_left2_transform(img, top_pt, bottom_pt, old_top_pt, old_mid_pt, new_top_pt, new_mid_pt):
    result = img.copy()
    for y in range(top_pt[1], bottom_pt[1]+1):
        basex = fy(top_pt, bottom_pt, y)
        oldx = fy(old_top_pt, old_mid_pt, y)
        newx = fy(new_top_pt, new_mid_pt, y)
        xprim = int(newx + 1*(newx - oldx))
        if oldx == newx or basex == newx or newx == xprim:
            continue

        im1 = img[y:y+1, xprim:oldx]
        im1 = cv2.resize(im1, (newx-xprim, 1),
                         interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], newx-im1.shape[1]:newx] = im1

        im1 = img[y:y+1, oldx:basex]
        im1 = cv2.resize(
            im1, (basex-newx, 1), interpolation=cv2.INTER_NEAREST)
        result[y:y+im1.shape[0], basex-im1.shape[1]:basex] = im1

    oldx = fy(old_top_pt, old_mid_pt, top_pt[1])
    newx = fy(new_top_pt, new_mid_pt, top_pt[1])
    top_xprim = (int(newx + 2*(newx-oldx)), top_pt[1])

    bottom_pt = (bottom_pt[0] , bottom_pt[1] + 5)
    oldx = fy(old_top_pt, old_mid_pt, bottom_pt[1])
    newx = fy(new_top_pt, new_mid_pt, bottom_pt[1])
    bottom_xprim = (int(newx + 2*(newx-oldx)), bottom_pt[1])
    roi = np.array([top_pt, top_xprim, bottom_xprim, bottom_pt])

    return roi_blur(img, result, roi)


def roi_blur(img, result, roi):
    dst = cv2.GaussianBlur(result, (3, 3), cv2.BORDER_DEFAULT)

    (h, w) = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # cv2.drawContours(mask, [result_pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.fillPoly(mask, [roi], 255)
    mask_inv = cv2.bitwise_not(mask)

    dst = cv2.bitwise_and(dst, dst, mask=mask)
    img_h = cv2.bitwise_and(img, img, mask=mask_inv)

    return cv2.add(dst, img_h)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def new_btm_point(btm, other):
    r = math.sqrt((btm[0] - other[0])*(btm[0] - other[0]) +
                  (btm[1] - other[1])*(btm[1] - other[1])) * 1.5
    c = math.atan((other[1]-btm[1])/(other[0]-btm[0]))

    if other[0] < btm[0]:
        c = c + math.pi

    x = r * math.cos(c) + btm[0]
    y = r * math.sin(c) + btm[1]

    return (int(x), int(y))


def new_top_point(top, btm, other):
    r = math.sqrt((btm[0] - other[0])*(btm[0] - other[0]) +
                  (btm[1] - other[1])*(btm[1] - other[1])) * 0.5
    c = math.atan((other[1]-btm[1])/(other[0]-btm[0]))

    if other[0] < btm[0]:
        c = c + math.pi

    x = r * math.cos(c) + top[0]
    y = r * math.sin(c) + top[1]

    return (int(x), int(y))


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-i", "--image", help="A photo taken by the user (face)")
parser.add_argument("-t", "--threshold",
                    help="The amount by which a face part can be increased or decreased")


# Read arguments from command line
args = parser.parse_args()

if args.image == None:
    print("Please set parameter --image")
    exit

if args.threshold == None:
    print("Please set parameter --threshold")
    exit

nose_slim_coef = float(args.threshold) / 50
eye_slim_coef = float(args.threshold) / 100


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

i = 0
c = 0

# while i < 68 and c != 27:
#     i = i+1

    # read the image
    # img = cv2.imread(f"helen/{i}.jpg")
img = cv2.imread(args.image)
img = image_resize(img, height=800)
result = img.copy()

# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
faces = detector(gray)
if len(faces) == 0:
    print("No Face Found!")
    exit

face = faces[0]

# Create landmark object
landmarks = predictor(image=gray, box=face)

nose_top = (landmarks.part(27).x, landmarks.part(27).y)
nose_bottom = (landmarks.part(30).x, landmarks.part(30).y)

# if (nose_top[0]-nose_bottom[0]) != 0:
#     angle = math.atan(
#         (nose_top[1]-nose_bottom[1])/(nose_top[0]-nose_bottom[0]))
#     angle = math.pi/2 - angle

#     rot_mat = cv2.getRotationMatrix2D(nose_bottom, angle, 1.0)
#     img = cv2.warpAffine(
#         img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

#     result = img.copy()

#     # Convert image into grayscale
#     gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#     # Use detector to find landmarks
#     faces = detector(gray)
#     if len(faces) == 0:
#         continue

#     face = faces[0]
#     x1 = face.left()  # left point
#     y1 = face.top()  # top point
#     x2 = face.right()  # right point
#     y2 = face.bottom()  # bottom point

#     faces = detector(gray)

#     if len(faces) == 0:
#         continue
#     face = faces[0]

#     # Create landmark object
#     landmarks = predictor(image=gray, box=face)

# # Loop through all the points
# for n in range(0, 68):
#     x = landmarks.part(n).x
#     y = landmarks.part(n).y

#     # Draw a circle
#     cv2.circle(img=img, center=(x, y), radius=3,
#                color=(0, 255, 0), thickness=-1)

nose_top = (landmarks.part(28).x, landmarks.part(28).y)
nose_bottom = (landmarks.part(33).x, landmarks.part(33).y)
nose_left = (landmarks.part(31).x, landmarks.part(31).y)
nose_right = (landmarks.part(35).x, landmarks.part(35).y)

nose_top_left = new_top_point(nose_top, nose_bottom, nose_left)
nose_top_right = new_top_point(nose_top, nose_bottom, nose_right)
nose_mid_left = new_btm_point(nose_bottom, nose_left)
nose_mid_right = new_btm_point(nose_bottom, nose_right)

slim_top_left = (nose_top_left[0] - int(nose_slim_coef *
                                        (nose_top_left[0] - nose_top[0])/4.0), nose_top_left[1])
slim_top_right = (nose_top_right[0] - int(nose_slim_coef *
                                            (nose_top_right[0] - nose_top[0])/4.0), nose_top_right[1])

slim_mid_left = (nose_mid_left[0] - int(nose_slim_coef *
                                        (nose_mid_left[0] - nose_left[0])), nose_mid_left[1])
slim_mid_right = (nose_mid_right[0] - int(nose_slim_coef *
                                            (nose_mid_right[0] - nose_right[0])), nose_mid_right[1])

nose_top_ = (landmarks.part(28).x, landmarks.part(28).y)
nose_bottom_ = (landmarks.part(33).x, landmarks.part(33).y+20)

d = landmarks.part(43).x - landmarks.part(42).x

eye_left = (landmarks.part(42).x-d, landmarks.part(42).y)
eye_top1 = (landmarks.part(43).x, landmarks.part(43).y)
eye_top2 = (landmarks.part(44).x, landmarks.part(44).y)
eye_right = (landmarks.part(45).x+d, landmarks.part(45).y)
eye_bottom1 = (landmarks.part(46).x, landmarks.part(46).y)
eye_bottom2 = (landmarks.part(47).x, landmarks.part(47).y)

slim_eye_top1 = (eye_top1[0], eye_top1[1] -
                    int(eye_slim_coef * (eye_top1[1] - eye_left[1])))
slim_eye_top2 = (eye_top2[0], eye_top2[1] -
                    int(eye_slim_coef * (eye_top2[1] - eye_left[1])))

slim_eye_bottom1 = (eye_bottom1[0], eye_bottom1[1] -
                    int(2*eye_slim_coef * (eye_bottom1[1] - eye_left[1])))
slim_eye_bottom2 = (eye_bottom2[0], eye_bottom2[1] -
                    int(2*eye_slim_coef * (eye_bottom2[1] - eye_left[1])))

# print(
#     f"eye_top1 {eye_top1} , slim_eye_top1 {slim_eye_top1} eye_slim_coef {eye_slim_coef}")
# print(
#     f"eye_bottom1 {eye_bottom1} , slim_eye_bottom1 {slim_eye_bottom1} eye_slim_coef {eye_slim_coef}")
# img = cv2.polylines(img, [np.array([eye_left, eye_top1,  eye_top2, eye_right])], True, ( 0,255, 0), 2)
# img = cv2.polylines(img, [np.array([eye_left, slim_eye_top1, slim_eye_top2,eye_right])], True, (0, 0,128), 2)

if nose_slim_coef >= 0:
    result = strech_right_transform(
        img, slim_top_right, nose_bottom_, nose_top_right, nose_mid_right, slim_top_right,  slim_mid_right)
    result = strech_left_transform(
        result, slim_top_left, nose_bottom_, nose_top_left, nose_mid_left, slim_top_left,  slim_mid_left)

    result = strech_top_transform(
        result, eye_left, eye_right, eye_top1,  eye_top2, slim_eye_top1, slim_eye_top2)
    result = strech_bottom_transform(
        result, eye_left, eye_right, eye_bottom1,  eye_bottom2, slim_eye_bottom1, slim_eye_bottom2)

else:
    result = strech_right2_transform(
        img, nose_top_, nose_bottom_, nose_top_right, nose_mid_right, slim_top_right,  slim_mid_right)
    result = strech_left2_transform(
        result, nose_top_, nose_bottom_, nose_top_left, nose_mid_left, slim_top_left,  slim_mid_left)

    result = strech_top2_transform(
        result, eye_left, eye_right, eye_top1,  eye_top2, slim_eye_top1, slim_eye_top2)
    result = strech_bottom2_transform(
        result, eye_left, eye_right, eye_bottom1,  eye_bottom2, slim_eye_bottom1, slim_eye_bottom2)

cv2.imshow(winname="Face", mat=img)
cv2.imshow(winname="Result", mat=result)
cv2.imwrite(f"result_{args.image}", result)

# Delay between every fram
c = cv2.waitKey(delay=0)

# Close all windows
cv2.destroyAllWindows()
