import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import csv
import numpy as np

def write_landmarks_to_csv(landmarks, frame_number, csv_data, data_dist, shape):
    #print(f"Landmark coordinates for frame {frame_number}:")
    #for normalize pose landmark
    image_height, image_width, _ = shape
    #if frame_number == 0:
    first = 29
    framenum = 20
    label = 0
    act = 1
    firstact = 0
    lastact = 0

    if frame_number == first:
        print("Start")
        data_dist[3] = 0
        data_dist[4] = 0
        data_dist[5] = 0
        data_dist[6] = 0
        data_dist[7] = 0
        data_dist[8] = 0
        #data firstact
        data_dist[9] = 0
        #data_aksi
        data_dist[10] = 0
        #data_aksi_global
        data_dist[11] = 0
    #else:
        #print(data_dist[3], data_dist[4], data_dist[5])

    for idx, landmark in enumerate(landmarks):
        #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x*100}, y: {landmark.y*100}, z: {landmark.z*100})")
        #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        data_dist[11] = 0

        if mp_pose.PoseLandmark(idx).name == 'LEFT_WRIST' :
            #LEFT_WRIST x,y,z 3,4,5.
            if frame_number > 0:
                point1 = (landmark.x, landmark.y)
                point2 = (data_dist[3], data_dist[4])
                distance = np.linalg.norm(np.array(point1) - np.array(point2))*100
                if distance > 7:
                    lastact = frame_number
                    if (lastact-data_dist[9])>framenum:
                        data_dist[10] = data_dist[10]+1
                        data_dist[9] = frame_number
                        data_dist[11] = data_dist[10]

                    print("Euclidean distance Left Wrist:", distance, "framenumber ", frame_number, "aksi ", data_dist[10])
                    cv2.imwrite('output_img_raw/train/frame%d.jpg' % frame_number, frame)
                    label=1


            data_dist[3] = landmark.x
            data_dist[4] = landmark.y
            data_dist[5] = landmark.z
            #print(data_dist[3],'asdsd')

        if mp_pose.PoseLandmark(idx).name == 'RIGHT_WRIST' :
            #RIGHT_WRIST x,y,z 6,7,8.
            if frame_number > 0:
                point1 = (landmark.x, landmark.y)
                point2 = (data_dist[6], data_dist[7])
                distance = np.linalg.norm(np.array(point1) - np.array(point2))*100
                if distance > 7:
                    lastact = frame_number
                    if (lastact - data_dist[9]) > framenum:
                        data_dist[10] = data_dist[10] + 1
                        data_dist[9] = frame_number
                        data_dist[11] = data_dist[10]

                    print("Euclidean distance Right Wrist:", distance, "framenumber ", frame_number, "aksi ", data_dist[10])
                    cv2.imwrite('output_img_raw/train/frame%d.jpg' % frame_number, frame)
                    label=1

            data_dist[6] = landmark.x
            data_dist[7] = landmark.y
            data_dist[8] = landmark.z

        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z, data_dist[10],label])
        #skip dulu biar cpt
        #cv2.imwrite('output_img_raw/fed_front/frame%d.jpg' % frame_number, frame)



video_path = 'input_videos/fed_front_miami.mp4'
output_csv = 'output/bnf/output_video_fed.csv'


# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
distance = 0
csv_data = []
data_dist = []
for i in range(18):
    data_dist.append([])

dist = 0
xtemp = 0
ytemp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    # capture from 165-190 : 25 frame.

    #supaya cepat
    #cv2.imwrite('output_img_raw/fed_front/frame%d.jpg' % frame_number, ret)

    if result.pose_landmarks:
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data, data_dist, frame_rgb.shape)
        #print('xxtemp :',xtemp)
        # mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Add the landmark coordinates to the list and print them
        text_put = str(frame_number)
        #print(dist)
        #text_put = str(frame_number) + " " + str(dist)
        cv2.putText(frame, text_put, (70, 50), cv2.FONT_HERSHEY_PLAIN, 4,
                    (255, 255, 0), 3)
        #supaya cepat di disable
        #cv2.imwrite("output_img/fed_front/frame%d.jpg" % frame_number, frame)


    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

# Save the CSV data to a file
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z','action','label'])
    csv_writer.writerows(csv_data)