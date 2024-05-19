import cv2
import pytesseract

# Load an image from file
image_path = './download.jpeg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(gray_image)
text = pytesseract.image_to_string(gray_image)


# Print the recognized text
print("Recognized Text:")
print(text)
