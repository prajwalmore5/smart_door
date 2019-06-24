import os
import cv2
import glob
import shutil
import pickle
import numpy as np
from glob import glob
from PIL import Image
from time import sleep

def train_model():
      BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #stores directory name
      image_dir = os.path.join(BASE_DIR, "images") #stores path of images

      face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
      recognizer = cv2.face.LBPHFaceRecognizer_create()

      current_id = 0
      label_ids = {}
      y_labels = []
      x_train = []


      for root, dirs, files in os.walk(image_dir):
            for file in files:
                  if file.endswith("png") or file.endswith("jpg"):
                        path = os.path.join(root, file)
                        label = os.path.basename(os.path.dirname(path)).replace(" ","_").lower()
                        #print(label, path)
                        if not label in label_ids:
                              label_ids[label] = current_id
                              current_id += 1
                        id_ = label_ids[label]
                        #print(label_ids)
                        #y_labels.append(label)
                        #x_train.append(path)
                        pil_image = Image.open(path).convert("L") #gives image stored in path and converts it into grayscale
                        size = (550,500)
                        #final_image = pil_image.resize(size, Image.ANTIALIAS)
                        image_array = np.array(pil_image, "uint8")
                        #image_array = np.array(final_image, "uint8") #converts grayscale image into numpy array. It stores pixel values in an array
                        #print(image_array)
                        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                        for (x,y,w,h) in faces:
                              roi = image_array[y:y+h, x:x+w]
                              x_train.append(roi)
                              y_labels.append(id_)
                              
      with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)
            
      recognizer.train(x_train, np.array(y_labels))
      recognizer.save("trainer.yml")
      #print(y_labels)
            
def face_recog():
      face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
      #eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
      verify = 0
      recognizer = cv2.face.LBPHFaceRecognizer_create()
      recognizer.read("trainer.yml")
      labels = {}
      with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {value:key for key,value in og_labels.items()}
            
      cap = cv2.VideoCapture(0)

      while True:
            ret, frame = cap.read() #capture frame by frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                  #print(x,y,w,h)
                  roi_gray = gray[y:y+h, x:x+w]
                  roi_color = frame[y:y+h, x:x+w]

                  #recognize
                  id_, conf = recognizer.predict(roi_gray)
                  if (conf >= 45):# and conf <= 85):
                        #print(id_)
                        #print(labels[id_])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name = labels[id_]
                        color = (255,255,255) #bgr
                        stroke = 2
                        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                  color = (255,0,0) #bgr
                  stroke = 2
                  end_cord_x = x+w
                  end_cord_y = y+h
                  cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y),color, stroke)
            #display resulting frame
            cv2.imshow('frame',frame)
            try:
                  if labels[id_] in labels.values():
                        print("verify")
                        if verify == 4:
                              print("Welcome "+str(labels[id_]))
                              break
                        verify+=1
            except UnboundLocalError :
                  print("Cannot detect face.")
                  main()
            if cv2.waitKey(20) & 0xFF == ord('q'):
                  break
      cap.release()
      cv2.destroyAllWindows()

def new_face_data():
      new_user = str(input("enter your name"))
      path = 'D:/yolo/Facial Recognition/facial_recognition/new_user'
      images_path = 'D:/yolo/Facial Recognition/facial_recognition/images'
      try:
        if not os.path.exists("new_user"):
            os.makedirs("new_user")
      except OSError:
        print ('Error: Creating directory. '+new_user)
      shutil.move(path, images_path)
      user_path = 'D:/yolo/Facial Recognition/facial_recognition/images/new_user'
      user_path2 = images_path+'/'+new_user
      os.rename(user_path, user_path2)
      
      cap = cv2.VideoCapture(0)
      count = 0
      while (count<=60):
            ret, frame = cap.read()
            frame_roi = frame[165:420,205:445]
            for i in range (1,15):
                  new_face_image = str(i)+".png"
                  blue = (255,0,0) #bgr
                  stroke = 2
                  x = 300
                  y = 280
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  message = "Keep your face inside the box."
                  white = (255,255,255) 
                  message_stroke = 2
                  cv2.putText(frame, message, (100,100), font, 1, white, message_stroke, cv2.LINE_AA)
                  #cv2.circle(frame,(x,y), 140, color, stroke)
                  cv2.rectangle(frame,(200,160),(450,425),blue,2)
                  cv2.imwrite(os.path.join(user_path2 , new_face_image),frame_roi)
            cv2.imshow('frame',frame)
            count = count+1
            if cv2.waitKey(20) & 0xFF == ord('q'):
                  break
      cap.release()
      cv2.destroyAllWindows()

def delete_face_data():
      images_path = 'D:/yolo/Facial Recognition/facial_recognition/images'
      print(os.listdir(images_path))
      folder_name = str(input("Select the face data that you want to delete."))
      shutil.rmtree(images_path+'/'+folder_name)
      
     
def main():
      a = int(input("press 1 to scan\npress 2 to add new face data\npress 3 to delete previous face data"))
      if (a==1):
            face_recog()
      elif (a==2):
            new_face_data()
            train_model()
            #train_model()
            print("Successfully added new face data.")
            main()
      elif (a==3):
            delete_face_data()
            print("Face data deleted successfully.")
            main()
      else:
            print("Select correct option!")
            sleep(2)
            main()
            
main()
