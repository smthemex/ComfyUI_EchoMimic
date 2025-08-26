# pip install retina-face
# we recommand tensorflow==2.15
# 


from retinaface import RetinaFace
import sys
from PIL import Image
import numpy as np

def get_mask_coord(img):

  #img = Image.open(image_path).convert("RGB")
  img = np.array(img)[:,:,::-1]
  if img is None:
    raise ValueError(f"Exception while loading {img}")

  height, width, _ = img.shape
  
  facial_areas = RetinaFace.detect_faces(img)  
  if len(facial_areas) == 0:
    print ('has no face detected!')
    return None
  else:
    face = facial_areas['face_1']
    x,y,x2,y2 = face["facial_area"]
    
    return y,y2,x,x2,height,width

# if __name__ == "__main__":
#   image_path = sys.argv[1]
#   y,y2,x,x2,height,width = get_mask_coord(image_path)
#   print (y,y2,x,x2,height,width)
