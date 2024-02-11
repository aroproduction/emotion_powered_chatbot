import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import requests
from localStoragePy import localStoragePy
from deepface import DeepFace

localStorage = localStoragePy('my_app', 'json')

model = "@cf/meta/llama-2-7b-chat-int8"
account_id = "22d5746a5309e439f88bd2acc163466c"
api_token = "uEknbTWt4BvUEqKQMgAqr9RFoU3Ejnzlot6n2rIV"

# max_indexes = []
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

st.title('Emotion Powered Chatbot')

class VideoTransformer(VideoTransformerBase):
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    def transform(self, frame):


        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

        # Load the age detection model
        age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        VideoTransformer.model.load_weights('pr_model.h5')

        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # Find haar cascade to draw bounding box around face
        frame = frame.to_ndarray(format="bgr24")
        # if not ret:
        #     break
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        global emotion_dict

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = VideoTransformer.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            # Gender detection
            face = frame[y:y + h, x:x + w] # cropping the face found
            result = DeepFace.analyze(face, actions = ['gender'], enforce_detection=False)
            gender = "Male" if result[0]['dominant_gender'] else "Female"

            # Preprocess the ROI for age detection
            roi = cv2.resize(frame[y:y+h,x:x+w], (227, 227))
            blob = cv2.dnn.blobFromImage(roi, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Display the gender, age and emotion on the frame in a comma-separated format
            cv2.putText(frame, f"{gender}, {age}, {emotion_dict[maxindex]}", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # global max_indexes 

            localStorage.setItem('key', maxindex)

        return frame


# Face Analysis Application #
activiteis = ["Home", "Webcam Face Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activiteis)
if choice == "Home":
    html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Gender, Age and Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                        </div>
                                        </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)
    st.write("""
                The application has two functionalities.

                1. Real time face detection using web cam feed.

                2. Real time face emotion recognization.

                """)
elif choice == "Webcam Face Detection":
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    user_input = st.text_input("Ask something:")
    if st.button('Send'):
        if user_input != '':
            pass
            # Get a value from local storage
            value = localStorage.getItem('key')
            # print(value)
            # st.write("Max_index:", value, emotion_dict)
            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
                headers={"Authorization": f"Bearer {api_token}"},
                json={"messages": [
                    {"role": "system", "content": f"You are a emotion powered chatbot. Your responses are influenced by the user emotions. Currently the user is  {emotion_dict[int(value)]}!"},
                    {"role": "user", "content": user_input}
                ]}
            )

            inference = response.json()
            st.markdown(inference["result"]["response"])
        else:
            st.chat_input('Please enter a question.')


elif choice == "About":
    st.subheader("About this app")
    html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                </div>
                                </br>"""
    st.markdown(html_temp_about1, unsafe_allow_html=True)

else:
    pass

# Remove a value from local storage
# localStorage.removeItem('key')

# Clear all values from local storage
localStorage.clear()