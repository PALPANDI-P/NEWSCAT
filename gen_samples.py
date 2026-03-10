import os
from PIL import Image, ImageDraw, ImageFont

def generate_image():
    # Create an image with text "Apple announces record breakthrough in artificial intelligence"
    img = Image.new('RGB', (800, 400), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Just draw basic text
    d.text((50, 150), "Apple announces a record breakthrough in artificial intelligence", fill=(0, 0, 0))
    d.text((50, 200), "The new technology will revolutionize the tech industry and markets.", fill=(0, 0, 0))
    d.text((50, 250), "Stock prices are soaring as investors rush to buy shares.", fill=(0, 0, 0))
    
    os.makedirs('data_samples', exist_ok=True)
    img.save('data_samples/test_image.jpg')
    print("Created test_image.jpg")

if __name__ == '__main__':
    generate_image()
