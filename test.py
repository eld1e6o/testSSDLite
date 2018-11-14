import os, sys

sys.path.append("/opt/ocv/lib/python3.6/dist-packages/")

import api
import cv2 
import numpy as np
import sys
import imutils
import pygame
import serial
import time
# sleep a little to wait for the arduino to reset
time.sleep(1.5)

relayPresent = False

if relayPresent is True:
    # Set up serial. This needs some error handling 
    relay_device = serial.Serial('/dev/ttyUSB0',
                                baudrate=9600,
                                bytesize=serial.EIGHTBITS,
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                timeout=5,
                                xonxoff=0,
                                rtscts=0)
    time.sleep(1.5)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        print(ix, iy)
        quit()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

cv2.namedWindow('Salida', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Salida',draw_circle)

pygame.init()
time = pygame.time

sounds =  []
#sounds.append(pygame.mixer.Sound("/home/diego/Code/AI tour/alto_ahi_loca_rafa-b5P2zQHbTWA.wav"))
sounds.append(pygame.mixer.Sound("/home/diego/Code/AI tour/test.wav"))
sounds.append(pygame.mixer.Sound("/home/diego/Code/AI tour/alarm_short.wav"))
statusPlay = sounds[0].play()
sounds[0].stop()
sounds[1].stop()

cap = cv2.VideoCapture(sys.argv[1])

rotate = True 

#centerSueloInterior = (483, 644)
#axesSueloInterior = (228, 76)
centerSueloInterior = (395 - 100, 436) 
axesSueloInterior = (185, 50)

centerSueloExterior = centerSueloInterior
axesSueloExterior = (228, 276)

angle = 0

#delimitedZoneInterior = [np.array([ [255, 641], [276, 302], [652, 317], [712, 641] ], np.int32)]
delimitedZoneInterior = [np.array([ [210, 432], [178, 207], [648, 227], [579, 440] ], np.int32)] + np.array([-100, 0])# Final
delimitedZoneExterior = [np.array([ [210, 432], [178, 207], [648, 227], [579, 440] ], np.int32)] # Final
#delimitedZoneExterior = delimitedZoneExterior - np.array([100, 100], np.int32)

ret, img = cap.read() # Para obtener las dimensiones
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

maxwh = max(img.shape[:2])

imMaskZoneInterior = np.zeros((maxwh, maxwh), np.uint8)
#imMaskZoneExterior = np.zeros((maxwh, maxwh), np.uint8)
cv2.ellipse(imMaskZoneInterior,centerSueloExterior,axesSueloExterior,angle,0,180,(128), -1)
cv2.ellipse(imMaskZoneInterior,centerSueloInterior,axesSueloInterior,angle,0,180,(255), -1)
cv2.fillPoly(imMaskZoneInterior, delimitedZoneInterior, (255))

kernel = np.ones((5,5),np.uint8)
imMaskZoneInterior = cv2.dilate(imMaskZoneInterior,kernel,iterations = 1)
imMaskZoneInterior = cv2.dilate(imMaskZoneInterior,kernel,iterations = 1)

cv2.imshow('Zona 1', imMaskZoneInterior)
#cv2.imshow('Zona 2', imMaskZoneExterior)

alertLevel = 0
alertLevelActual = 0
playingLevel = 0
relayStatus = 0
needToPlay = 0
needRelay = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
lineType               = 2

logo = cv2.imread("/home/diego/Code/AI tour/logo.png")
logo = cv2.resize(logo, (0,0), fx=0.1, fy=0.1) 

faceCascade = cv2.CascadeClassifier("/opt/ocv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")


peopDet = None
facesLast = None
    
repeats = 0

imMaskRGB = cv2.cvtColor(imMaskZoneInterior, cv2.COLOR_GRAY2BGR)


while(True):
    
    ret, img = cap.read()
    if img is None:
        if relayPresent is False:
            quit()
        relay_data_encode = str.encode("relay_3\n")
        relay_device.write(relay_data_encode)
        relay_data_encode = str.encode("off\n")
        relay_device.write(relay_data_encode)
        quit()
    
    (hImg, wImg) = img.shape[:2]

    img = img[int(hImg*0.05):int(hImg*0.95), int(wImg*0.25):int(wImg*0.9)]
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    img = cv2.flip(img, 1)
    
    worstZone = 0
    centerPeople=(0,0)
    
    if (type(img) is not np.ndarray):
        break

    if rotate == True:
        rotated90 = imutils.rotate_bound(img, 90)
    else:
        rotated90 = img
        
    (h, w) = rotated90.shape[:2]

    maxwh = max(w,h)
    minwh = min(w,h)
    
    #toFill = (maxwh - minwh)/2

    imdest = np.zeros([maxwh, maxwh, 3], dtype=img.dtype)

    if (h>w):# and 0:
        imdest[:, int((maxwh - w)/2) : int((maxwh - w)/2 + w)] = rotated90[:,:]
        rotated90 = imdest

    else:
        imdest[int((maxwh - h)/2) : int((maxwh - h)/2 + h), :]=rotated90[:,:]#int((maxwh - w)/2) : int((maxwh - w)/2 + w)] = rotated90[:,:]
        rotated90 = imdest
        
    imgOrig = rotated90

    imMaskPeople = np.zeros(imdest.shape[:2], np.uint8)
        
    cv2.ellipse(rotated90,centerSueloInterior,axesSueloInterior,angle,0,180,(255,0,0), 1)
    cv2.ellipse(rotated90,centerSueloExterior,axesSueloExterior,angle,0,180,(255, 64, 0), 2)
        
    alertLevel = 0
    
    bbox_list = api.api(rotated90, 0.35)
    
    for i in bbox_list:
        #if tamaño continue
        #Imprimo persona
        cv2.rectangle(rotated90, i[0], i[1], (125, 255, 51), thickness=2)

        #Para imprimir límites
        cx,cy = int((i[0][0] + i[1][0])/2), int((i[0][1] + i[1][1])/2)
        ax1,ax2 =  int((i[1][0] - i[0][0])/2), int((i[1][1]-i[0][1])/2)
        center = (cx,cy)        
        axes = (ax1,ax2)
    
        bottomLeftCornerOfText = (int(i[0][0]), int(i[0][1]))
        fontColor              = (0,255,0)
        text = str(i[1][0] - i[0][0]) + " x " +  str(i[1][1] - i[0][1])
        cv2.putText(rotated90,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        peopCropped = np.copy(rotated90[int(i[0][1]):int(i[1][1]), int(i[0][0]):int(i[1][0])])
        gray = cv2.cvtColor(peopCropped, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(peopCropped, (x, y), (x+w, y+h), (0, 255, 0), 2)
               
        #cv2.imshow('crop', peopCropped)
                
        centerPeople = (int((i[1][0] + i[0][0])/2), int(i[1][1]))
        if centerPeople[1]<imMaskZoneInterior.shape[1] and centerPeople[0]<imMaskZoneInterior.shape[0]:
            toDetectZone = imMaskZoneInterior[centerPeople[1], centerPeople[0]]
        else:
            toDetectZone = 0
        
        if toDetectZone == 128:
            
            if alertLevel < 1:
                peopDet = peopCropped
                if len(faces) == 1:
                    #Defino facesLast
                    facesLast = peopCropped[y:y+h,x:x+w]
                    #cv2.imshow('cara', facesLast)
            
            cv2.circle(rotated90, centerPeople, 15, (255, 255, 255), -1)
            
            #cv2.ellipse(rotated90,center,axes,angle,0,360,(255,0,0), 1)
            cv2.ellipse(imMaskPeople,center,axes,angle,0,360,(255), -1)

            alertLevel = 1

            cv2.circle(rotated90, (int(i[1][0] + 40), int(i[0][1]) + 40), 18, (0,255, 255), -1)
            cv2.circle(rotated90, (int(i[1][0] + 40), int(i[0][1])), 15, (0,0, 0), -1)
            cv2.circle(rotated90, (int(i[1][0] + 40), int(i[0][1]) + 80), 15, (0,0, 0), -1)


        elif toDetectZone == 255:
            cv2.rectangle(rotated90, i[0], i[1], (0, 0, 255), thickness=2)

            
            peopDet = peopCropped
            
            if len(faces) == 1:
                #Defino facesLast
                facesLast = peopCropped[y:y+h,x:x+w]
                #cv2.imshow('cara', facesLast)

            cv2.circle(rotated90, centerPeople, 15, (255, 255, 255), -1)
            #cv2.ellipse(rotated90,center,axes,angle,0,360,(0,0,255), 1)

            cv2.ellipse(imMaskPeople,center,axes,angle,0,360,(255), -1)

            alertLevel = 2
            cv2.circle(rotated90, (int(i[1][0] + 40), int(i[0][1])+40), 15, (0, 0, 0), -1)
            cv2.circle(rotated90, (int(i[1][0] + 40), int(i[0][1]) ), 18, (0,0, 255), -1)
            cv2.circle(rotated90, (int(i[1][0] + 40), int(i[0][1]) + 80), 15, (0,0, 0), -1)
            

    rotated90 = cv2.addWeighted(rotated90, 0.7, imMaskRGB[0:624, 0:624], 0.3, 0)
            
    #Hasta acá imprime
    toFill = (maxwh - minwh+200)/2
    (h, w) = rotated90.shape[:2]
    imdest = np.zeros([maxwh, maxwh+200, 3], dtype=img.dtype)
    imdest[:,100:w+100] = rotated90
    rotated90 = imdest
    
    if alertLevel == 2:
        if alertLevelActual <2:
            #forzar los cambios:
            needToPlay = 2
            needRelay = 1

    elif alertLevel == 1:
        needToPlay = 1
        needRelay = 0

    else:
        needToPlay = 0
        needRelay = 0
            
    #Control de audios
    
    # si actual < alerta => comienzo alerta
    if not statusPlay.get_busy():
        nowPlaying = 0
        
    if needRelay == 0 and relayStatus == 1 and nowPlaying == 0:
        if relayPresent is True:
            relay_data_encode = str.encode("relay_3\n")
            relay_device.write(relay_data_encode)
            relay_data_encode = str.encode("off\n")
            relay_device.write(relay_data_encode)
        relayStatus = 0
        
    if needToPlay == 2 and nowPlaying != 2:
        sounds[0].stop()
        sounds[1].stop()
        statusPlay = sounds[1].play()
        nowPlaying = 2
        if needRelay == 1 and relayStatus == 0:
            if relayPresent is True:
                relay_data_encode = str.encode("relay_3\n")
                relay_device.write(relay_data_encode)
                relay_data_encode = str.encode("on\n")
                relay_device.write(relay_data_encode)
            relayStatus = 1

    elif needToPlay == 1 and nowPlaying == 0:
        statusPlay = sounds[0].play()
        nowPlaying = 1
        
    #if nowPlaying == 0:
    #    rotated90[:,:] = rotated90[:,:]/2
    
    #imprimo texto
    if nowPlaying == 1:
        bottomLeftCornerOfText = (int(toFill) + 10, 30)
        fontColor              = (0,255,0)
        
        cv2.putText(rotated90,'Persona en zona de alerta!', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
    elif nowPlaying == 2:
        bottomLeftCornerOfText = (int(toFill) + 10, 30)
        fontColor              = (0,0,255)
        
        cv2.putText(rotated90,'Persona en zona prohibida!', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

    fontColor              = (255,255,255)
    cv2.putText(rotated90,'REGISTRO:', 
        (10, 10), 
        font, 
        fontScale,
        fontColor,
        lineType)        

    alertLevelActual = alertLevel
    
    #rotated90 = cv2.bitwise_and(rotated90, imgOrig, mask = imMaskPeople)
    
    (logoH, logoW) = logo.shape[:2]
    (hFinal, wFinal) = rotated90.shape[:2]
    
    #Pego el cuerpo entero al costado
    startX = 50
    
            
    if facesLast is not None:
        (faceH, faceW) = facesLast.shape[:2]
        factor = toFill / faceW
        facesDetResized = cv2.resize(facesLast, (0,0), fx=factor, fy=factor)
        #cv2.imshow('im', facesDetResized)
        (faceH, faceW) = facesDetResized.shape[:2]
        if faceH + startX>=hFinal:
            facesDetResized = facesDetResized[0:hFinal - startX, :]
            facesH = hFinal - startX
                
        rotated90[startX:faceH + startX,0:faceW] = facesDetResized

        startX = startX + faceH
        
    if peopDet is not None:
        (peopH, peopW) = peopDet.shape[:2]
        factor = toFill / peopW
        peopDetResized = cv2.resize(peopDet, (0,0), fx=factor, fy=factor)
        #cv2.imshow('im', peopDetResized)
        (peopH, peopW) = peopDetResized.shape[:2]
        if peopH+ startX>=hFinal:
            peopDetResized = peopDetResized[0:hFinal - startX, :]
            peopH = hFinal - startX
        
        
        rotated90[startX:peopH + startX,0:peopW] = peopDetResized
        
    
    #Pego la cara al costado
    
    rotated90[hFinal - logoH:hFinal, wFinal - logoW:wFinal]=logo[:,:]


    #print(logoH, logoW)
    #cv2.imshow('logo', logo)
    
    cv2.imshow('Zona 1', imMaskZoneInterior)    
    cv2.imshow('Salida', rotated90)

    #cv2.imshow('mask1', imMaskPeople)
    #cv2.imshow('orig', imgOrig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break#plt.imshow(img[:,:,::-1])
        
if relayPresent is False:
    quit()
relay_data_encode = str.encode("relay_3\n")
relay_device.write(relay_data_encode)
relay_data_encode = str.encode("off\n")
relay_device.write(relay_data_encode)
quit()

