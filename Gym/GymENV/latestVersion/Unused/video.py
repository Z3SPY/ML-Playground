import cv2
import os

# Define the folder where videos are stored.
video_folder = 'videos'

# List all files in the video folder that end with '.mp4'
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

if not video_files:
    print("No video files found in the 'videos' folder.")
    exit()

# For this example, we use the first video file found.
video_path = os.path.join(video_folder, video_files[0])
print("Displaying video:", video_path)

# Create a VideoCapture object to read from the video file.
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop until the video ends or the user presses 'q'
while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame from the video.
    if not ret:
        print("End of video or error reading the frame.")
        break

    # Display the frame in a window named 'Lunar Lander Video'
    cv2.imshow('Lunar Lander Video', frame)
    
    # Wait for 25ms for a key press. If 'q' is pressed, break the loop.
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Quitting video playback.")
        break

# Release the VideoCapture object and close display window(s).
cap.release()
cv2.destroyAllWindows()
