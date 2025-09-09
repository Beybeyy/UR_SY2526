import cv2

# Path to your image file
image_path = r"C:\Users\Bea\UR_SY2526\CCTV\image.png"

# Load the image
img = cv2.imread(image_path)

# Check if image was loaded successfully
if img is None:
    print("❌ Error: Could not load image.")
    exit()

# Show the image in a window
cv2.imshow("Image", img)

# Wait indefinitely until any key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
