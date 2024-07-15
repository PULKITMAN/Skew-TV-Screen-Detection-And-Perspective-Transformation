```python
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for proper visualization
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        warped = cv2.flip(warped, 1)
        ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))  # Convert to RGB for proper visualization
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M


img = cv2.imread("drive-download-20240317T164434Z-001\IMG_6691.JPG")
img = cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR)
model = YOLO('best2.pt')
results = model.track(source=img, show=True, conf=0.5, save=True)
# from PIL import Image

# r = results[0]
# im_array = r.plot()  # plot a BGR numpy array of predictions
# im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
# cv2.imshow(im)
xy = results[0].keypoints.xy
print(xy)
p1,p2,p3,p4=xy[0]
src = np.float32([(p1[0],     p1[1]),
                  (p2[0],  p2[1]),
                  (p4[0],  p4[1]),
                  (p3[0],    p3[1])
                  ])

dst = np.float32([(640, 0),
                  (0, 0),
                  (640, 640),
                  (0, 640)])

unwarp(img, src, dst, True)

cv2.imshow("so", img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
```
