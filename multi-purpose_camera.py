import numpy as np
import cv2 as cv#Opencv library
import time
import pylibdmtx #Library for barcode reading
from pylibdmtx import pylibdmtx
from PIL import Image
import glob
import pickle

X1=0#They hold ROI for barcode area
Y1=0
X2=0
Y2=0

def setupCamera():
    width=800#camera resoution setup
    height=600
    cap = cv.VideoCapture(0,cv.CAP_V4L2)  
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)#set up resolution
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    print('resolution set')
    cap.set(cv.CAP_PROP_AUTOFOCUS,0)#turnoff autofocus
    print('autofocus off')
    focus=385#set to 8 cm distance focus
    cap.set(cv.CAP_PROP_FOCUS,focus)#set focus
    print('set focus')
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)#turn off autoexposure
    print('exposure off')
    expo=800#15.6 ms
    cap.set(15, expo)#set exposure time
    print('set exposure')
    return cap#return videoCapture
    
def getFrame(cap):#it returns current frame
    if(cap.isOpened()):
        # flush frame buffer before capture
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read() 
        frame= cv.rotate(frame, cv.ROTATE_180)#rotate the video
    return frame# return frame from video

def selectBarcodeROI(frame):#set up ROI for barcode, updates the .pickle file and saves. It returns cropped area location
    r = cv.selectROI("select the area", frame)
    X1=r[0]
    Y1=r[1]
    X2=r[2]
    Y2=r[3]
    with open('barcodeROI.pickle','wb') as f:
        pickle.dump(r,f)
    cv.destroyAllWindows()
    return r

def getBarcodeImage(frame):#it crop barcode image from the frame,it use location info from .pickle file. it returns cropped_image
    ff=open('barcodeROI.pickle','rb')
    roi=pickle.load(ff)
    ff.close
    X1,Y1,X2,Y2=roi[0],roi[1],roi[2],roi[3]
    cropped_image = frame[Y1:(Y1+Y2), X1:(X1+X2)]
    return cropped_image
    
def trainTemplate(frame,trainCount):# it saves template images that choosen from the frame. trainCount indicates count of cropping
    for x in range(trainCount):
        y1 = cv.selectROI("select the area", frame)
        template = frame[int(y1[1]):int(y1[1]+y1[3]), int(y1[0]):int(y1[0]+y1[2])]
        images = glob.glob(r'/home/pi/Templates/*.jpg')
        next_pic=len(images)+1
        template_path="/home/pi/Templates/"+str(next_pic)+".jpg"
        cv.imwrite(template_path,template)
        print("Selected Area= "+str(int(y1[1]))+":"+str(int(y1[1]+y1[3]))+" , "+str(int(y1[0]))+":"+str(int(y1[0]+y1[2])))
        print("template trainig completed, template id: "+str(next_pic))
        print("training "+str(x)+" is completed")
    print("Template training completed")
    cv.destroyAllWindows()
    
def plateDetect(frame):# this funtion uses template images fom the folder and match it.if there is a match, it means there is no plate and returns false
    start1= time.time()
    main_image_gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    images = glob.glob(r'/home/pi/Templates/*.jpg')
    loc=[[],[]]
    threshold=0.8 #should be 0.8 and more
    for templates in images:
        I=cv.imread(templates)
        template_gray=cv.cvtColor(I, cv.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]
        res = cv.matchTemplate(main_image_gray, template_gray, cv.TM_CCOEFF_NORMED)     
        test=np.where(res >= threshold)
        loc[0]=np.append(loc[0],test[0],axis=None)
        loc[1]=np.append(loc[1],test[1],axis=None)
    end1 = time.time()   
    result=loc
    result=result[::-1]  
    #print('Template match time consume: ',(end1-start1))
    if(len(result[0])!=0):
        print("NO PLATE")
        return False
    else:
        #print("PLATE FOUND")
        return True
    
def barcodeDecode(cropped_image):#it looks for barcode in the cropped image. If there is a barcode, it returns True and the barcode info
    start = time.time()
    barcodes=pylibdmtx.decode(cropped_image,max_count=1)
    end = time.time()
    #print('decoding time consume: ',"%.6f"%(end-start))
    if len(str(barcodes))>5:
        #print('Barcode Found:')
        return True,barcodes[0].data
    else:
        #print('No Barcode')
        return False
    
def saveImage(frame):
    images = glob.glob(r'/home/pi/Pictures/*.jpg')
    next_pic=len(images)+1
    template_path="/home/pi/Pictures/"+str(next_pic)+".jpg"
    cv.imwrite(template_path,frame)
    
def rearBarcode(frame,shiftValue):
    ff=open('barcodeROI.pickle','rb')
    roi=pickle.load(ff)
    ff.close
    X1,Y1,X2,Y2=roi[0],roi[1],roi[2],roi[3]
    cropped_image = frame[Y1:(Y1+Y2), (X1+shiftValue):(X1+X2+shiftValue)]
    return cropped_image
    
if __name__ == '__main__':
    
    
    cap=setupCamera()
    #frame=getFrame(cap)
    #crop=rearBarcode(frame,55)
    #cv.imshow('frame',crop)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #saveImage(crop)
    #trainTemplate(frame,2)
    #selectBarcodeROI(frame)
    plateStatus=False
    while(not plateStatus):
        frame=getFrame(cap)
        plateStatus=plateDetect(frame)
        if(plateStatus):
            cropped_image=getBarcodeImage(frame)
            barcodeStatus=barcodeDecode(cropped_image)
            if(barcodeStatus):
                print(barcodeStatus[1])
                break
            else:
                rearBarcodeCrop=rearBarcode(frame,55)
                rearBarcodeCheck=barcodeDecode(rearBarcodeCrop)
                if(rearBarcodeCheck):
                    print('Incorrect Orientation, Fix Orientation')
                else:
                    print('Please check your plate seating')
                plateStatus=False
                
            
              
              
              
              