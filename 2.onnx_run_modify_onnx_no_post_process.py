import onnxruntime as rt
import cv2
import numpy as np

sess = rt.InferenceSession("rtdetr_hgnetv2_l_6x_coco-modify.onnx")
img = cv2.imread("./dog.jpg")
org_img = img
h, w = img.shape[:2]
print("img shape: ", img.shape)
img = cv2.resize(img, (640,640))
image = img.astype(np.float32) / 255.0
input_img = np.transpose(image, [2, 0, 1])
image = input_img[np.newaxis, :, :, :]


results = sess.run(['scores',  'boxes'], {'image': image})

print("scores: ", tuple(results[0].shape))
print("bboxes: ", tuple(results[1].shape))

scores, boxes = [o[0] for o in results]

index = scores.max(-1)
boxes, scores = boxes[index>0.5], scores[index>0.5]
labels = scores.argmax(-1)
scores = scores.max(-1)

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
    cv2.rectangle(org_img, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(org_img, f"{int(label)}: {score:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.imwrite('./result.jpg', org_img)