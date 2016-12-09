import cv2
import dlib
import numpy

import os
import sys
import glob

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

REFERENCE_PATH="average_portrait/portraits/01a4462309f79052d1a480170ef3d7ca7bcbd564.jpg"

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    # if len(rects) > 1:
    #     raise TooManyFaces
    # if len(rects) == 0:
    #     raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    if(if_one_face(im)!=1):
        return None, None
    s = get_landmarks(im)

    return im, s

def read_im_and_landmarks_input_image(im):
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    if(if_one_face(im)!=1):
        return None, None
    s = get_landmarks(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def if_one_face(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return 2
    if len(rects) == 0:
        return 0
    return 1    

# warping all the images in a given folder
def gen_align_lip(target_directory, FOLDER_PATH, IMAGE_FORMAT):
    REFERENCE_PATH="average_portrait/portraits/01a4462309f79052d1a480170ef3d7ca7bcbd564.jpg"
    img_ref, landmark_ref = read_im_and_landmarks(REFERENCE_PATH)
    img_index = 0;

    for f in glob.glob(os.path.join(FOLDER_PATH, "*." + IMAGE_FORMAT)):
        #print("Processing file: {}".format(f))

        #if no face, jump out the frame
        img_index += 1
        img, landmark = read_im_and_landmarks(f);
        if(img==None):
            continue
        M = transformation_from_points(landmark_ref[ALIGN_POINTS],
                                       landmark[ALIGN_POINTS])

        warped_im2 = warp_im(img, M, img_ref.shape)
        c1=300
        r1=200
        roi = warped_im2[c1:c1+60,r1:r1+120]

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        cv2.imwrite(str(target_directory)+ "/" + str(img_index)+ '.jpg', roi)

# just load it once 
# save computation time
img_ref, landmark_ref = read_im_and_landmarks(REFERENCE_PATH)
def gen_align_lip_from_video(f):
    img, landmark = read_im_and_landmarks_input_image(f);
    if(img==None):
        return None
    M = transformation_from_points(landmark_ref[ALIGN_POINTS],
                                   landmark[ALIGN_POINTS])
    warped_im2 = warp_im(img, M, img_ref.shape)
    c1=255
    r1=209
    roi = warped_im2[c1:c1+109,r1:r1+109]
    return roi
    

# if __name__ == "__main__":        
#     speaker="s9"
#     folder="/Volumes/TOSHIBA EXT/research/lipsreading/video_480_dataset/"+speaker+"/"
#     fname=folder+"a.txt"
#     with open(fname) as f:
#         content = f.readlines()
#     for ele in content:
#         print(ele.rstrip()+"-"+speaker+"-face/")
#         folder_name=ele.rstrip()+"-"+speaker+"-face/"
#         read_path=folder+folder_name
#         target_dir="/Volumes/TOSHIBA EXT/research/lipsreading/grid-lipset/"+folder_name
#         gen_align_lip(target_dir, read_path, "png")
#     #gen_align_lip("result_")

