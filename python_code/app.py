import easyocr
import argparse

def read_text_from_image(image_path):
    """
    Read text from an image using EasyOCR.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        list: A list of tuples containing the detected text, bounding box coordinates, and confidence score.
    """
    # Create an OCR reader object
    reader = easyocr.Reader(['en'])
    
    # Read text from the image
    result = reader.readtext(image_path)
    
    return result

def print_text(text):
    """
    Print the detected text.
    
    Args:
        text (list): A list of tuples containing the detected text, bounding box coordinates, and confidence score.
    """
    for detection in text:
        # if detection[2] > 0.5:
            print(detection[1])

def main():
    """
    Extract text from an image using EasyOCR.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Extract text from an image using EasyOCR')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Read text from the image and print the extracted text
    text = read_text_from_image(args.image_path)
    print_text(text)

if __name__ == "__main__":
    main()

