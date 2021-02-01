''' PREDICTING FACIAL EXPRESSION AND PLAYING RELATED YT VIDEO '''

# first import vlc
# then go to the E:\ML_projects\Face-Expression_Recognition

# importing the required libraries
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import vlc
import pafy
import time
import json
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import face_recognition

''' Predciting the Facial Expression '''

# loading the trained model
model = load_model('E:/ML_projects/Face-Expression_Recognition/finalModel.h5')
 
# list of expressions
expressions = ['Anger', 'Fear', 'Happy', 'Sad', 'Surprise']

# getting the live camera feed
vc = cv2.VideoCapture(0)
while(True):
    
    _, frame = vc.read()
    
    rgb_frame = frame[:, :, ::-1] # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    for (startY, endX, endY, startX) in face_locations:
        
        processed_frame = frame[startY:endY, startX:endX] # locating the face
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.resize(processed_frame, (48, 48))
        processed_frame = img_to_array(processed_frame)
        processed_frame = processed_frame.reshape(1, 48, 48, 1) #(BS, frame_ROWS, frame_COLS, NUMBER_OF_CHANNELS)
        processed_frame = processed_frame.astype('float32')
        processed_frame = processed_frame / 255.0
        
        prediction = model.predict(processed_frame)
        index = np.argmax(prediction, axis=1)
        expression = expressions[index[0]]
        
        cv2.putText(frame, expression, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    # saving the expression captured in the image
    cv2.imwrite('expression.jpg', frame)
    
    key = cv2.waitKey(1) & 0xFF
    # press 'Q' to take a snap & then stop the webcam feed
    if key == ord('q'):
        break
    
# Cleanup
cv2.destroyAllWindows()
vc.release()

''' Playing related YT video via VLC ''' 

option = Options()
option.headless = False

driver = webdriver.Chrome('E:/ML_projects/Face-Expression_Recognition/chromedriver.exe', options=option)
driver.implicitly_wait(5)


URL = 'https://www.youtube.com/results?search_query='

link_list = [] # list of the URls of the videos from the Youtube search result page

driver.get(URL + expression + 'Chainsmokers Song')
    
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()
# appending URLs of all the videos in the search result page to a list
for vid in soup.select("a#thumbnail"):
    try :
        link_list.append('https://www.youtube.com' + vid.get('href')) # 'href' contains the video URL
    except:
        print("href not found in a tag")
        
        
videolink = pafy.new(link_list[1]) # playing the 3rd video
bestlink = videolink.getbest() # getting the best video format
media = vlc.MediaPlayer(bestlink.url)
media.play()
time.sleep(15) # playing the video for 20s
media.stop()        

        






