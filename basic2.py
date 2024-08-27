import cv2
import mediapipe as mp
import csv

def write_landmarks_to_csv(landmarks, frame_number, csv_data, shape):
    print(f"Landmark coordinates for frame {frame_number}:")
    #for normalize pose landmark
    image_height, image_width, _ = shape
    for idx, landmark in enumerate(landmarks):
        #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x*image_width}, y: {landmark.y*image_height}, z: {landmark.z*image_width})")
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        #if mp_pose.PoseLandmark(idx).name == 'NOSE' :
            #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x*image_width, landmark.y*image_height, landmark.z*image_width])
    print("\n")


#video_path = 'input_videos/input_video.mp4'
#output_csv = 'output/output_video.csv'

video_path = 'input_videos/input_video_fed_bnf.mp4'
output_csv = 'output/bnf/output_video.csv'


# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
csv_data = []

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
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, str(frame_number), (70, 50), cv2.FONT_HERSHEY_PLAIN, 4,
                    (255, 255, 0), 3)
        cv2.imwrite("output_img/frame%d.jpg" % frame_number, frame)

        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data, frame_rgb.shape)

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
    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
    csv_writer.writerows(csv_data)