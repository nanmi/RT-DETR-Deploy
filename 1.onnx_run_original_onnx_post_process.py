import onnxruntime as rt
import cv2
import numpy as np

sess = rt.InferenceSession("rtdetr_hgnetv2_l_6x_coco-sim.onnx")
img = cv2.imread("./dog.jpg")
print(img.shape)
org_img = img
im_shape = np.array([[float(img.shape[0]), float(img.shape[1])]]).astype('float32')
img = cv2.resize(img, (640,640))
scale_factor = np.array([[float(640/img.shape[0]), float(640/img.shape[1])]]).astype('float32')
img = img.astype(np.float32) / 255.0
input_img = np.transpose(img, [2, 0, 1])
image = input_img[np.newaxis, :, :, :]
result = sess.run(["reshape2_83.tmp_0"], {'im_shape': im_shape, 'image': image, 'scale_factor': scale_factor})
print(np.array(result[0].shape))
for value in result[0]:
    if value[1] > 0.5:
        cv2.rectangle(org_img, (int(value[2]), int(value[3])), (int(value[4]), int(value[5])), (255,0,0), 2)
        cv2.putText(org_img, str(int(value[0]))+": "+str(value[1]), (int(value[2]), int(value[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.imwrite("./result.png", org_img)