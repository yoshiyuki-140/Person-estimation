from deepface import DeepFace
from pprint import pprint


# 顔属性
face_img1 = './data/face_image1.jpg'
face_img2 = './data/face_image2.jpg'

objs1 = DeepFace.analyze(img_path=face_img1, actions=[
    'age', 'gender', 'race', 'emotion'])

# 顔認識(同一人物判定)
result = DeepFace.verify(img1_path=face_img1, img2_path=face_img2)

pprint(objs1)
pprint(result)
