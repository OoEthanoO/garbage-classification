import requests

def send_image_for_detection(image_path, server_url):
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post(server_url, files=files)
        if response.status_code == 200:
            with open('detected_image.jpg', 'wb') as f:
                f.write(response.content)
            print("Image processed successfully. Saved as 'detected_image.jpg'.")
        else:
            print(f"Failed to process image. Status code: {response.status_code}")
            print(response.json())

if __name__ == "__main__":
    image_path = '/Users/ethanxu/hack-the-north/garbage-classification/IMG_0087.jpeg'  # Replace with the path to your image
    server_url = 'https://cuddly-otters-worry.loca.lt/detect'
    send_image_for_detection(image_path, server_url)