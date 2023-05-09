# importing packages
import streamlit as st 
from streamlit_option_menu import option_menu 
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import time
from tensorflow import keras
model = keras.models.load_model('./otherFiles/trafficSignClassifier.h5')

#with open('style.css') as f:
#    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

## TO DO:
## change icons
## add heic compatibity

# images used
banner = Image.open("./pictures/websiteBanner.png")
autopilotCameraPerception = Image.open("./pictures/AutopilotCameraFeed.png")
ethicalIssue = Image.open("./pictures/ethicalIssue.png")
objectDetectionClassification = Image.open("./pictures/objectDetection.png")
signClassification = Image.open("./pictures/signClassification.png")
laneDetectionI = Image.open("./pictures/laneDetection.png")

# image and menu bar at the top of the site
st.image(banner)
menu = option_menu(menu_title = None, options = ["Future of Autonomous Vehicles", "Object Detection System", "Traffic Sign Classification System", "Lane Detection System"], 
menu_icon="cast", default_index = 0, icons = ["house", "hdd-network", "clipboard-data", "people-fill"], orientation="horizontal",
    styles={
        "container": {"padding": "1em 1.5em", "background-color": "#6b36a5"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "15px", "text-align": "middle", "margin":"3px", "--hover-color": "#917ed6", "writing-mode": "horizontal-tb", "display": "flex",  "align-items": "center", "justify-content": "center"},
        "nav-link-selected": {"background-color": "#4c519f"},
    })     

# Page 1
def home():
    # section 1: intro
    st.title("Future of Autonomous Vehicles")

    # section 2: impact, ethics, and future development
    st.header("Maximizing Social Benefit while Addressing Safety and Security Concerns:") 
    st.markdown(""" 
    """) 
    st.image(ethicalIssue, caption="Ethical Issue Posed by Autonoumous Vehicles: Is there a \"correct\" choice that artificial intelligence should choose?", use_column_width=True)
    st.markdown(""" 
    The emergence of autonomous vehicles has the potential to cause significant societal benefit
    such as increased access to transportation, reduced traffic congestion, and improved road
    safety. Self-driving cars can provide a more accessible means of transportation for 
    elderly and disabled individuals who are unable to drive. Because of vehicle-to-vehicle communication 
    systems, they also have the potential to reduce the number of accidents caused by human error. However, 
    there are also negative implications to consider. As with any new technology, there are significant 
    safety and security concerns associated with autonomous vehicles. The potential for system mishaps and 
    cyberattacks raise important questions about how to ensure the safety and security of several parties in 
    society. Moreover, there are ethical considerations around the deployment of autonomous vehicles, 
    including questions about who is responsible in fatal accidents, and how to address information 
    privacy issues. As the development and deployment of autonomous vehicles continues to progress, 
    it is crucial that we consider both benefits and concerns, and work together to ensure that this
    technology is used in a responsible and beneficial manner.
    """) 
    st.markdown(""" 
    """) 

    # section 3: role of ai
    st.header("The Crucial Role of Deep Learning and Machine Learning Systems in Autonomous Vehicles:") 
    st.markdown(""" 
    """) 
    st.image(autopilotCameraPerception, caption="Camera Feed of Tesla's \"Autopilot\" Software", use_column_width=True)
    st.markdown(""" 
    Deep learning and machine learning systems are advanced types of artificial intelligence that enable computers to learn how to make 
    highly-accurate predictions and decisions from data. The importance of deep learning and machine learning systems in autonomous vehicles cannot be overstated. 
    These technologies enable self-driving cars to interpret their environment in a way that mimics 
    human perception, allowing them to make real-time decisions and take action to avoid potential hazards. 
    Without these advanced systems, autonomous vehicles would be unable to navigate complex 
    roadways or respond to unpredictable situations. Deep learning and machine learning systems in autonomous 
    vehicles include computer vision algorithms for object recognition, natural language processing for human-machine 
    interaction, and reinforcement learning for decision-making and control. As we continue to develop and refine these technologies, it is crucial that 
    we strike a balance between maximizing the safety and security of autonomous vehicles and upholding ethics. By doing so, we can harness the full potential of deep-learning and machine-learning 
    systems to create a future of transportation that is both safe and secure for all parties.
    """)  

# function for object detection in images
def imageod():
    # image uploader
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    # starting process once file is uploaded
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)
        # displaying original image
        st.divider()
        st.image(img1, caption = "Original Image")
        st.divider()
        # sidebar title
        sideTitle = st.sidebar.title("Settings:")
        # bar to set confidence threshold
        confThreshold =st.sidebar.slider('Confidence Threshold:', 0, 100, 50)
        whT = 320
        classesFile = "./otherFiles/coconames.txt"
        f = open(classesFile, "r")
        classes = []
        for line in f.readlines():
            line = line.replace("\n", "")
            classes.append(line)
        net = cv2.dnn.readNetFromDarknet('./otherFiles/yolov3.cfg', './otherFiles/yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs,img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, 20/100)
            obj_list=[]
            confi_list =[]
            #drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img2, (x, y), (x+w,y+h), (240, 54 , 230), 2)
                obj_list.append(classes[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(img2,f'{classes[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
            if st.checkbox("Show Object's list" ):
                
                st.write(df)
            if st.checkbox("Show Confidence bar chart" ):
                st.subheader('Bar chart for confidence levels')
                
                st.bar_chart(df["Confidence"])
           
        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,img2)
    
        st.image(img2, caption='Processed Image.')

# Object Detection in Videos
def videood():
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    font_scale = 1
    thickness = 1
    classesFile = "./otherFiles/coconames.txt"
    f = open(classesFile, "r")
    classes = []
    for line in f.readlines():
        line = line.replace("\n", "")
        classes.append(line)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet('./otherFiles/yolov3.cfg', './otherFiles/yolov3.weights')

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Original Video")
        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        h, w = image.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
        count = 0
        while True:
            _, image = cap.read()
            if _ != False:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count +=1
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # loop over each of the layer outputs
                for output in layer_outputs:
                    # loop over each of the object detections
                    for detection in output:
                        # extract the class id (label) and confidence (as a probability) of
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # discard weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # perform the non maximum suppression given the scores defined before
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

                font_scale = 0.6
                thickness = 1

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = image.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

                out.write(image)
                cv2.imshow("image", image)
                
                if ord("q") == cv2.waitKey(1):
                    break
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()


# Page 2
def objectDetection(): 
    st.title("Object Detection and Classification") 
    st.markdown(""" Object detection and classification systems are essential components 
    of autonomous vehicles, enabling them to perceive and understand their environment in 
    real-time. These systems use sensors to identify and track 
    objects such as pedestrians, vehicles, and obstacles, allowing self-driving cars to make 
    informed decisions. The accuracy and reliability of these 
    systems are crucial to the safety and success of autonomous vehicles, as even small errors 
    or delays in detecting objects can have significant consequences. As the risk of sophisicated
    cyberattacks and other challenges increase, ensuring the robustness and reliability of these systems is critical to 
    achieve safe and effective deployment of autonomous vehicles in our streets and highways. """)
    st.image(objectDetectionClassification, caption="Example of the YOLO Object Detection and Classification Algorithm", use_column_width=True)
    st.divider()

    st.header("Replication of System:")
    choice  = st.selectbox("File Type",("Image","Video"))
    
    if choice == "Image":
        imageod()
    if choice == "Video":
        videood()

# Page 3
def trafficSignClassification():
    # section 1: info
    st.title("Traffic Sign Classification")  
    st.markdown("""
    Traffic sign classification systems are an integral component of autonomous vehicles, enabling 
    them to understand the rules of the road when navigating to a desination. By recognizing and classifying 
    traffic signs such as speed limits and 
    pedestrian crossings, these systems allow self-driving cars to respond appropriately to road regulations 
    and minimize potential conflicts with other motorists or pedestrians. The reliability of these systems 
    are vital to the reduce accidents on public roads. 
    Nonetheless, these systems are not immune to errors and failures. Adverse weather conditions, 
    obstructed views, incorrect positioning of signs, or even intentional vandalism or sabotage may 
    result in misinterpretations, which could lead to catastrophic consequences if left unaddressed. 
    Therefore, continuous monitoring of these system and addressing related issues must 
    remain integral parts of the development process to ensure their widespread safety. 
    """)
    st.image(signClassification, caption="Example of a Traffic Sign Classification Algorithm", use_column_width=True)
    # section 2: system demo
    st.divider()
    st.header("Replication of System:")
    file1 = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    data = []
    if file1 != None:
        img = Image.open(file1)
        st.image(img, caption = "Inputted Image", use_column_width=True)
        image = img.resize((30,30))
        data.append(np.array(image))
        pred = np.array(data)
    classes = { 1:'Speed limit (20km/h)',
           2:'Speed Limit (30km/h)',
           3:'Speed Limit (50km/h)',
           4:'Speed Limit (60km/h)',
           5:'Speed Limit (70km/h)',
           6:'Speed Limit (80km/h)',
           7:'End of Speed Limit (80km/h)',
           8:'Speed Limit (100km/h)',
           9:'Speed Limit (120km/h)',
           10:'No Passing',
           11:'No Passing Vehicle Over 3.5 Tons',
           12:'Right-of-way at Intersection',
           13:'Priority Road',
           14:'Yield',
           15:'Stop',
           16:'No Vehicles',
           17:'Vehicles > 3.5 Tons Prohibited',
           18:'No Entry',
           19:'General Caution',
           20:'Dangerous Curve Left',
           21:'Dangerous Curve Right',
           22:'Double Curve',
           23:'Bumpy Road',
           24:'Slippery Road',
           25:'Road Narrow to the Right',
           26:'Road Work',
           27:'Traffic Signals',
           28:'Pedestrian Crossing',
           29:'Children Crossing',
           30:'Bicycles Crossing',
           31:'Beware of Ice/Snow',
           32:'Wild Animals Crossing',
           33:'End of Speed and Passing Limits',
           34:'Turn Right Ahead',
           35:'Turn Left Ahead',
           36:'Ahead Only',
           37:'Go Straight or Right',
           38:'Go Straight or Left',
           39:'Keep Right',
           40:'Keep Left',
           41:'Mandatory Roundabout',
           42:'End of No Passing',
           43:'End of No Passing Vehicles with Weights Greater than 3.5 Tons' }
    Button = st.button("Run")
    if Button == True:
        predict_input = model.predict(pred)
        pred = np.argmax(predict_input,axis=1)
        for x in pred:
            sign = classes[x + 1]
            st.header(f"Model Prediction: {sign} Sign")

# Page 4
def laneDetection():
    def make_coordinates(image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

    def canny(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def display_lines(image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def region_of_interest(image):
        height = image.shape[0]
        polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    # section 1: info
    st.title("Lane Detection")  
    st.markdown("""
    Lane detection systems are a significant component of autonomous vehicles, facilitating 
    navigation by discerning between lanes marked on public roads. Detection methods 
    leverage imagery and cloud data drawn from sensors to generate precise maps of each lane's boundaries 
    in real time. This information feeds directly into decision-making processes governing steering and 
    acceleration choices. As with all complex automated technologies, there are numerous threats to the 
    optimal functioning of this subsystem. Inclement weather, foggy 
    conditions, construction work obscuring lane markings, debris accumulation, and deliberate 
    attempts at interference are a few factors that might compromise 
    performance. Given present and future challenges, continued research into solutions 
    aimed at improving adaptability across different environmental and 
    operational conditions is necessary. 
    """)
    st.image(laneDetectionI, caption="3D Tesla Visualization of Lane Detection System", use_column_width=True)
    # section 2: system demo
    st.divider()
    st.header("Replication of System:")
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    # starting process once file is uploaded
    if file != None:
        img = file
        file = cv2.imread('/Users/dhruvbejugam/Desktop/AutonomousSystemsWebsite/laneDetection/' + file.name)
        file = np.copy(file)
        # displaying original image
        st.divider()
        st.image(img, caption = "Original Image")
        st.divider()
        canny_image = canny(file)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(file, lines)
        line_image = display_lines(file, averaged_lines)
        combo_image = cv2.addWeighted(file, 0.8, line_image, 1, 1)
        st.image(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB), caption = "Processed Image")

if menu == "Future of Autonomous Vehicles":
    home() 
elif menu == "Object Detection System":
    objectDetection()
elif menu == "Traffic Sign Classification System":
    trafficSignClassification() 
elif menu == "Lane Detection System":
    laneDetection() 
