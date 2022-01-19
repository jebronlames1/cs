from unicodedata import east_asian_width
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
from nembir_pilate_localijation import main as number_plate_localizer
from save_in_csv import *
from PIL import Image
from imaje_resije import conTO28x28, image_resize_sklearn


#*********************************************************************************************************************************************************************************

#FLASK_APP
def flask_app():
    from flask import Flask, render_template, request
    import numpy as np
    import os
    from imutils.contours import sort_contours
    import numpy as np
    import argparse
    import imutils
    import cv2
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model("Combined_Resnet_50_Epochs")
    print("model is loaded")

    app = Flask(__name__)
    @app.route("/", methods=['GET', 'POST'])
    def home():
        return render_template('index.html')


    @app.route("/predict", methods=['GET', 'POST'])
    def predict():
        if request.method == 'POST':
            file = request.files['image']
            filename = file.filename
            file_path = os.path.join('static/user uploaded', filename)
            file.save(file_path)
            test_image = tf.keras.preprocessing.image.load_img(file_path)
            src = cv2.imread(file_path)
            print(src)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 150)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sort_contours(cnts, method="left-to-right")[0]
            chars = []
            for c in cnts:
                # compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                # filter out bounding boxes, ensuring they are neither too small
                # nor too large
                if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
                    # extract the character and threshold it to make the character
                    # appear as *white* (foreground) on a *black* background, then
                    # grab the width and height of the thresholded image
                    roi = gray[y:y + h, x:x + w]
                    thresh = cv2.threshold(roi, 0, 255,
                                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    (tH, tW) = thresh.shape
                    # if the width is greater than the height, resize along the
                    # width dimension
                    if tW > tH:
                        thresh = imutils.resize(thresh, width=32)
                        # otherwise, resize along the height
                    else:
                        thresh = imutils.resize(thresh, height=32)
                    # re-grab the image dimensions (now that its been resized)
                    # and then determine how much we need to pad the width and
                    # height such that our image will be 32x32
                    (tH, tW) = thresh.shape
                    dX = int(max(0, 32 - tW) / 2.0)
                    dY = int(max(0, 32 - tH) / 2.0)
                    # pad the image and force 32x32 dimensions
                    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                                value=(0, 0, 0))
                    padded = cv2.resize(padded, (32, 32))
                    # prepare the padded image for classification via our
                    # handwriting OCR model
                    padded = padded.astype("float32") / 255.0
                    padded = np.expand_dims(padded, axis=-1)
                    # update our list of characters that will be OCR'd
                    chars.append((padded, (x, y, w, h)))
            boxes = [b[1] for b in chars]
            chars = np.array([c[0] for c in chars], dtype="float32")
            # OCR the characters using our handwriting recognition model
            preds = model.predict(chars)
            # define the list of label names
            labelNames = "0123456789"
            labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            labelNames = [l for l in labelNames]

            output = ""
            for (pred, (x, y, w, h)) in zip(preds, boxes):
                i = np.argmax(pred)
                prob = pred[i]
                label = labelNames[i]
                output += label

            print("output",output)

            return render_template('sec.html', pred_output=output, user_image=file_path)

    if __name__ == "__main__":
        app.run(threaded=False)

#*********************************************************************************************************************************************************************************


def stream_lit_app():
#STREAMLIT_APP
    try:
        import cv2
        from nembir_pilate_localijation import main as number_plate_localizer
        import streamlit as st  #Web App
        from PIL import Image #Image Processing
        import numpy as np #Image Processing 
        import os
        import easyocr
#title
        st.title("NUMBER PLATE RECOGNITION")
#subtitle
        st.markdown("")

#image uploader
        image = st.sidebar.file_uploader(label = "Upload the image of the car here",type=['png','jpg','jpeg'])
        if image is not None:
            #CHECKING IF IMAGE EXISTS OR NOT 

            input_image = Image.open(image) #read image
            file_details = image.name,image.type
            with open(os.path.join("tempdir" ,"temp.png"),"wb") as f:
                f.write(image.getbuffer())
            #saving numberplate in a directory called tempdir
            
            Type = st.radio("" , ["EasyOCR" , "ML Model"])

        
            st.sidebar.image(input_image) 
        #display image
            st.sidebar.write(file_details[0])
            
        #displaying name of file user uploaded
        
            st.sidebar.success("Image successfully uploaded!")
            st.balloons()
            if Type == "ML Model":
                if st.button("Click here to read the numer plate!"):
                    number_plate_localizer("tempdir/temp.png")
                    format='.png'
                    myDir = "plates"
                    def createFileList(myDir, format='.png'):
                        fileList = []
                        for root, dirs, files in os.walk(myDir, topdown=False):
                           for name in files:
                             # print(name)
                              if name.endswith(format):
                                 fullName = os.path.join(root, name)
                                 fileList.append(fullName)
                        return fileList
                    plates = createFileList(myDir)
                    if plates == []:
                        st.write("NumberPlate Not Found")
                    else:
                       try:
                        for image in createFileList(myDir):
                           pred = predict_image(image)
                           if len(pred) > 3:
                             img=Image.open(image)
                             st.image(img , caption="Localized Numberplate")
                     
                             st.write("prediction =",  pred)
                             save_in_csv()
                       
               
                       except:                
                        st.write("Can't Read The Numberplate")
            elif Type == "EasyOCR":
                if st.button('Click Here To Read The numberplate') :
                    number_plate_localizer("tempdir/temp.png")
                    format='.png'
                    myDir = "plates"
                    def createFileList(myDir, format='.png'):
                        fileList = []
                        for root, dirs, files in os.walk(myDir, topdown=False):
                           for name in files:
                             # print(name)
                              if name.endswith(format):
                                 fullName = os.path.join(root, name)
                                 fileList.append(fullName)
                        return fileList
                    plates = createFileList(myDir)
                    if plates == []:
                        st.write("NumberPlate Not Found")
                    else:
                       try:
                        for image in createFileList(myDir):
                            Reader = easyocr.Reader(['en'])
                            text =Reader.readtext(image , paragraph= False)
                            text_ = ""
                            accuracy=0
                            for i in range(len(text)):
                            
                               text_+=text[i][1].rstrip("\n")
                               accuracy += float(text[i][2])
                               
                            st.write(text_)
                       except:
                            st.write("Number Plate Not Found")
                   

        else:
         st.sidebar.write("Upload an Image")

       
    except:
        #st.write("Can't Read the Numberplate")
        pass
       
                     

        st.caption('''By Jebronlames 
        and Brick Freak''')     

#*********************************************************************************************************************************************************************************



stream_lit_app()

            
          





