import cv2
import numpy as np

def create_invisibility_cloak():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Allow the camera to warm up
    for i in range(30):
        ret, frame = cap.read()

    # Capture the background frame
    _, background = cap.read()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of red color in HSV (two ranges due to wrap-around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Refine the mask
        mask = cv2.medianBlur(mask, 5)

        # Invert the mask
        inv_mask = cv2.bitwise_not(mask)

        # Use the mask to extract the red regions
        red_regions = cv2.bitwise_and(background, background, mask=mask)

        # Extract non-red regions from the current frame
        non_red_regions = cv2.bitwise_and(frame, frame, mask=inv_mask)

        # Combine the red regions from background and non-red regions from current frame
        result = cv2.add(red_regions, non_red_regions)

        # Display the result
        cv2.imshow('Invisibility Cloak', result)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_invisibility_cloak()