import cv2 as cv
import numpy as np
from math import sqrt
from . import build_predictions

def show(x):
    cv.imshow("image", x)
    cv.waitKey(0)
    cv.destroyAllWindows()	

def calculate_ratio(var1):
    imgpath_folder = r"C:\Users\dell\Desktop\CDACProject\HR_GUI\results\uncertainty"
    maskpath_folder = r"C:\Users\dell\Desktop\CDACProject\HR_GUI\mask"

	#Filename fetched from html
    var2 = str(var1)
	
	#Conversion of extension from .jpg to .png
    img_name_new = list(var2)
    img_name_new[-3:] = "png"
    img_name_new = "".join(img_name_new)

	#Generate full path for segmented image and mask image
    imgpath = imgpath_folder + "\\" +str(img_name_new)
    maskpath = maskpath_folder + "\\" + str(var1)

    #Read the mask and convert it into gray format
    mask = cv.imread(maskpath)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY) #Mask B&W conversion

	#Read segmented image
    img = cv.imread(imgpath)

    #Fetch arteries and veins from the segmented image
    vein = img[:, :, 0]
    artery = img[:, :, 2]

    #Crop arteries and veins as per ROI using mask image
    vein = cv.bitwise_and(mask, vein)
    artery = cv.bitwise_and(mask, artery)
	
	#Calculation of AVR ratio begins
    img = artery
    size = np.size(img)
    arteryskel = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    while( not done ):
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        arteryskel = cv.bitwise_or(arteryskel, temp)
        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True
			
    img = vein
    size = np.size(img)
    veinskel = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS,(3, 3))
    done = False
    while( not done):
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        veinskel = cv.bitwise_or(veinskel, temp)
        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros==size:
            done = True
			
    venuole = []
    arteriole = []
    for x in range(veinskel.shape[0]):
        for y in range(veinskel.shape[1]):
            if veinskel[x][y] > 0:
                venuole.append(veinskel[x][y])
            if arteryskel[x][y] > 0:
                arteriole.append(arteryskel[x][y])
				
    venuole = sorted(venuole)
    arteriole = sorted(arteriole)
    lenven = len(venuole)
    if lenven % 2 == 1:
        Wa = venuole[lenven//2]
        if lenven//2 == 0:
            Wb = venuole[0]
        else:
            Wb = venuole[lenven//2 - 1]
    else:
        Wa = (venuole[lenven//2 - 1] + venuole[lenven//2])// 2
        Wb = venuole[lenven//2 - 2]
    print(Wa, Wb)
	
    CRVE = sqrt(0.72*(Wa**2) + 0.91*(Wb**2) + 450.02)
	
    lenart = len(arteriole)
    print("Length of artery:", lenart)

    if lenart%2 == 1:
        Wa = arteriole[lenart//2]
        if lenart//2 == 0:
            Wb = arteriole[0]
        else:
            Wb = arteriole[lenart//2 - 1]
    else:
        Wa = (arteriole[lenart//2 - 1] + arteriole[lenart//2])// 2
        Wb = arteriole[lenart//2 - 1]

    CRAE = sqrt(0.87*(Wa**2) + 1.01*(Wb**2) - .22*Wa*Wb - 10.73)
    artervenratio = CRAE/CRVE
    return artervenratio



