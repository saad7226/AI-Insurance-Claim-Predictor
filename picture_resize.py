from PIL import Image
import os

image_dir = 'static/images/'
target_size = (3024, 4032)  
for filename in ['member1.jpg', 'member2.jpg', 'member3.jpg']:
    filepath = os.path.join(image_dir, filename)
    if os.path.exists(filepath):
        with Image.open(filepath) as img:
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_resized.save(filepath)