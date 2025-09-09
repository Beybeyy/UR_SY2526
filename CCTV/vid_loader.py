import cv2

video_path = r"C:\Users\Bea\UR_SY2526\CCTV\record_clear.mp4"  # Use raw string or other safe format

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    cv2.imshow("Video Playback", frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
