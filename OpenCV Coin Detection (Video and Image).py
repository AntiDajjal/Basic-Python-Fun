import cv2
import numpy as np
from tkinter import Tk, filedialog

def detect_coins(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coins = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0 and 4 * np.pi * area / (perimeter ** 2) > 0.7 and area > 500:
            # Calculate the bounding circle of the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            coins.append((center, radius))

    return coins

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load the image.")
        return

    detected_coins = detect_coins(image)
    for coin in detected_coins:
        center, radius = coin
        cv2.circle(image, center, radius, (0, 255, 0), 2)

    # Display the image with detected coins
    cv2.imshow('Detected Coins', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Failed to open the video.")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            break

        detected_coins = detect_coins(frame)
        for coin in detected_coins:
            center, radius = coin
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        output_video.write(frame)

        cv2.imshow('Detected Coins', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

def select_file():

    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select Image or Video", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("Video Files", "*.mp4;*.avi;*.mov")])
    file_extension = file_path.lower().split('.')[-1]

    if file_extension in ['jpg', 'jpeg', 'png']:
        process_image(file_path)
    elif file_extension in ['mp4', 'avi', 'mov']:
        process_video(file_path)
    else:
        print("Unsupported file format.")

select_file()
