
#---------------------------------------------RECTANGLE DETECTION-----------------------------
import cv2
import numpy as np
import glob
from math import atan2,degrees
import os
  



def numberOfFiles(path):#get number of the files
    APP_FOLDER = path

    totalDirectories = 0

    for base, dirs, files in os.walk(APP_FOLDER):

        for directories in dirs:
            totalDirectories += 1

    return totalDirectories


images = glob.glob(r'Source\*.png')#Select all the images in the Source folder

for image1 in images:
    image=cv2.imread(image1)
    next_File=numberOfFiles('Result')+1
    filePath='Result/'+str(next_File)
    print('Image Number-->',next_File)
    os.makedirs(filePath)#Make new file

    path=filePath+"/Original Image"+".png"# imeage path
    cv2.imwrite(path,image)
    #cv2.imshow('Original', image)
    #cv2.waitKey(0)

    #Get image dimension
    dimensions = image.shape
    print('image dimension(height x width): ',dimensions[:2])

    #Get image height x width
    height=image.shape[0]
    width=image.shape[1]

    #Get middle point of the image
    XMIDDLE=width/2
    YMIDDLE=height/2

    #Create a black background for drawing
    black_image=np.ones((height, width, 3), dtype = np.uint8)
    black_image1=np.ones((height, width, 3), dtype = np.uint8)
    
    # Convert Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    path=filePath +"/Gray-Scale Image"+".png"# imeage path
    cv2.imwrite(path,gray)
    #cv2.imshow('GrayScale', gray)
    #cv2.waitKey(0)

    # Find Canny edges
    edged = cv2.Canny(gray, 100, 200)
    path=filePath +"/Canny-Edge"+".png"# imeage path
    cv2.imwrite(path,edged)
    #cv2.imshow('Canny Edge', edged)
    #cv2.waitKey(0)
    
    # Finding Contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(black_image1, contours, -1, (255,255,0), 1)#draw all the EXTERNAL CONTOURS
    path=filePath +"/All External Contours Image"+".png"# image path
    cv2.imwrite(path,black_image1)
    #cv2.imshow('All Contours', black_image1)
    #cv2.waitKey(0)

    maxsize = 0 #it indicates max contour size 
    best = 0  #max contour index
    count = 0  #counter
    for cnt in contours:  
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approximations = cv2.approxPolyDP(cnt, epsilon, True)
        #Validate if it is a rectangle
        if len(approximations)>=4:
            #Find biggest rectangle area
            if cv2.contourArea(cnt) > maxsize : 
                maxsize = cv2.contourArea(cnt)  
                best = count  
        count += 1  


    bestArea=contours[best]#biggest rectangle
    epsilon1 = 0.1 * cv2.arcLength(bestArea, True)
    approximations1 = cv2.approxPolyDP(bestArea, epsilon1, True)
    print('best rectangle area: \n',approximations1)
    #print('len approx: ',len(approximations1))


    #----------------------------------ANGLE DETECTION-----------------------------------------------
    p1=approximations1[0]
    p2=approximations1[1]
    p3=approximations1[2]
    p4=approximations1[3]
    perimeter = cv2.arcLength(approximations1, True)
    print('Perimeter: ',"%.2f"%perimeter)
    if(perimeter>100):#Check if perimeter size matches

        #Find left and right bottom point of the rectangle
        for point in approximations1:
            if(point[0,0]<XMIDDLE and point[0,1]>YMIDDLE):
                leftBottom=point
            elif(point[0,0]>XMIDDLE and point[0,1]>YMIDDLE):
                rightBottom=point
        print('Left Bottom Point: ',leftBottom)
        print('Right Bottom Point: ',rightBottom)
        xDiff = rightBottom[0,0]-leftBottom[0,0]#X axis length
        yDiff = leftBottom[0,1]-rightBottom[0,1]#Y axis length
        print('X Difference: ',xDiff)
        print('Y Difference: ', yDiff)
        angle=abs(degrees(atan2(yDiff, xDiff)))
        angle="%.2f"% angle
        print(angle,' degree difference') #Angle between points
        print('Counter Area: ', cv2.contourArea(approximations1))
        print('---------------------------------------------------------------------------------------------')
        #Draw best contour
        cv2.drawContours(black_image, [approximations1], 0,(0,255,0), 1)
        path=filePath +"/Barcode Box"+".png"# imeage path
        cv2.imwrite(path,black_image)
        #cv2.imshow('Barcode Box', black_image)
        #cv2.waitKey(0)
        cv2.putText(black_image,str(angle),(int(XMIDDLE),int(YMIDDLE)),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,255),1)
        path=filePath +"/Angle Extracted= "+str(angle)+".png"# imeage path
        cv2.imwrite(path,black_image)
        # cv2.imshow('Angle', black_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        print('--------------------------------------------------------------------------------------------')
        print('Perimeter is too small')
        print('--------------------------------------------------------------------------------------------')

