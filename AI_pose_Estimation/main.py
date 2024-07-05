import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils # drawing utils
mp_pose = mp.solutions.pose # importing pose estimation models

### Make Detection
# VIDEO FEED
cap = cv2.VideoCapture(0)

# Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read() # frame -> actual image from webcam, ret -> return value not used
#         # cv2.imshow('Mediapipe Feed', frame) #name -> 'Mediapipe Feed'

#         # detection stuff and render
#         # recolor image
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False # performance tuning

#         results = pose.process(image) # making detection

#         # recoloring image back to intial color format of opencv
#         image.flags.writeable = True
#         image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # print(results)
#         # render the detections
#         mp_drawing.draw_landmarks(
#             image=image, 
#             landmark_list=results.pose_landmarks, 
#             connections=mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # custom colors for dots
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 280), thickness=2, circle_radius=2)) # custom colors for connections

#         cv2.imshow('Mediapipe feed', image)
    
#         if cv2.waitKey(10) & 0xFF == ord('q'): # if close or 'q' key pressed
#             break

#     cap.release()
#     cv2.destroyAllWindows


# extract landmarks
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read() # frame -> actual image from webcam, ret -> return value not used

#         # detection stuff and render
#         # recolor image
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False # performance tuning

#         results = pose.process(image) # making detection

#         # recoloring image back to intial color format of opencv
#         image.flags.writeable = True
#         image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # extract landmarks
#         try: 
#             landmarks = results.pose_landmarks.landmark
#         except:
#             pass

#         # render the detections
#         mp_drawing.draw_landmarks(
#             image=image, 
#             landmark_list=results.pose_landmarks, 
#             connections=mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # custom colors for dots
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 280), thickness=2, circle_radius=2)) # custom colors for connections

#         cv2.imshow('Mediapipe feed', image)
    
#         if cv2.waitKey(10) & 0xFF == ord('q'): # if close or 'q' key pressed
#             break

#     cap.release()
#     cv2.destroyAllWindows

# print(len(landmarks))
# print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) # left shoulder landmark value (mp_pose.PoseLandmark.LEFT_SHOULDER.value -> return index value of left shoulder landmark)

def calculate_angle(a, b, c):
    # calculate the angle between the lines of any there points
    a = np.array(a) #first 
    b = np.array(b) #mid
    c = np.array(c) #end
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if (angle > 180.0):
        angle = 360-angle

    return angle

# shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
# print(shoulder, elbow, wrist)
# print(calculate_angle(shoulder, elbow, wrist))
# calculate_angle()

# angle tracking
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read() # frame -> actual image from webcam, ret -> return value not used

#         # detection stuff and render
#         # recolor image
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False # performance tuning

#         results = pose.process(image) # making detection

#         # recoloring image back to intial color format of opencv
#         image.flags.writeable = True
#         image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # extract landmarks
#         try: 
#             landmarks = results.pose_landmarks.landmark

#             # get coordinates
#             hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            
#             # calculate angle
#             angle = calculate_angle(hip, shoulder, elbow)
#             # visualise angle
#             cv2.putText(image, 
#                 str(angle), 
#                 tuple(np.multiply(elbow, [640, 480]).astype(int)), # calculate coordinates for text to be elbow * (resolution of input image)
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#         except:
#             pass

#         # render the detections
#         mp_drawing.draw_landmarks(
#             image=image, 
#             landmark_list=results.pose_landmarks, 
#             connections=mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # custom colors for dots
#             connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 280), thickness=2, circle_radius=2)) # custom colors for connections

#         cv2.imshow('Mediapipe feed', image)
    
#         if cv2.waitKey(10) & 0xFF == ord('q'): # if close or 'q' key pressed
#             break
#     cap.release()
#     cv2.destroyAllWindows

# curl counting
counter = 0
stage = None
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        try: 
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            cv2.putText(image, 
                str(angle), 
                tuple(np.multiply(elbow, [640, 480]).astype(int)), # calculate coordinates for text to be elbow * (resolution of input image)
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
            

            # curl counter logic
            if angle > 160:
                stage = 'down'
            if angle < 30 and stage == 'down':
                stage = 'up'
                counter += 1
                print(counter)
        except:
            pass

        # render curl counter
        # setup render box -> top left corner and ending at (225, 73)
        cv2.rectangle(image, (0,0), (225, 73), (245, 117, 16), -1)

        # rep data
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # stage data
        cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(stage), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=results.pose_landmarks, 
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # custom colors for dots
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 280), thickness=2, circle_radius=2)
            ) # custom colors for connections

        cv2.imshow('Mediapipe feed', image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'): # if close or 'q' key pressed
            break

    cap.release()
    cv2.destroyAllWindows 