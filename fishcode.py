import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import numpy as np
from PIL import Image
import depth_pro
import os
 
from sklearn import linear_model
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import normalize
import math
 
fl = 0.05  # 50mm standard
t = 0.5  # threshold for fish length
# Weight prediction formula coefficients
a = 8.2 * 10 ** -6  # Weight formula coefficient
b = 3.2  # Weight formula exponent
 
threshold = 0.5
image_path = "/content/drive/My Drive/yolv8segmentation/data/images/train/u.png"
image_input = cv2.imread(image_path)
model_path = "/content/drive/My Drive/yolv8segmentation/runs/segment/train/weights/last.pt"
 
model = YOLO(model_path)
results = model(image_input)[0]
 
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        cv2.rectangle(image_input, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image_input, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()
# load and preprocess an image
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)
# Run inference.
prediction = model.infer(image, f_px=f_px)
depth_np = prediction["depth"]  # Depth in [m].
depth_np = depth_np.squeeze().cpu().numpy()
num_points = 1000  # Number of random points
 
for box in results.boxes.xyxy.tolist():
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) // 2)  # force to int
    center_y = int((y1 + y2) // 2)  # force to int
    depth_value = depth_np[center_y, center_x]
 
    text = f"Depth: {depth_value:.2f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
 
    text_x = int(x1)
    text_y = int(y1 - 10)
    rect_x1 = int(text_x - 5)
    rect_y1 = int(text_y - text_size[1] - 10)
    rect_x2 = int(text_x + text_size[0] + 5)
    rect_y2 = int(text_y + 5)
    cv2.rectangle(image_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.putText(image_input, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    mu0 = 1.003
    mu1 = 1.5
    mu2 = 1.333
    d0 = 0.25
    d1 = 0.012  # cm glass thickeness
    dm = depth_value
    d1m = d1 / (mu1 / mu0)
    d2m = dm - d0 - d1m
    d2 = d2m * (mu2 / mu0)
    dr = d0 + d1 + (dm - d0 - d1 / (mu1 / mu0)) * (mu2 / mu0)
    alpha = d0 / dr
    beta = d1 / dr
    gamma = d2 / dr
    mu_r = d0 * mu0 * alpha + d1 * mu1 * beta + d2 * mu2 * gamma
    pix_L = abs(x2 - x1)
    sin_theta2 = pix_L / math.sqrt(pix_L ** 2 + fl ** 2)
    C = sin_theta2 * (mu0 / mu_r)
    pix_R = C * fl / (math.sqrt(1 - C ** 2))
    box_l = pix_R * dr / fl
    depth_variation = depth_value * 0.1
    random_z = np.random.uniform(depth_value - depth_variation, depth_value + depth_variation, num_points)
    # Compute random X and Y coordinates
    length = x2 - x1
    width = y2 - y1
    smaller_dimension = min(length, width)
    range_limit = smaller_dimension / 4
    random_x = np.random.uniform(center_x - range_limit, center_x + range_limit, num_points)
    random_y = np.random.uniform(center_y - range_limit, center_y + range_limit, num_points)
    # Combine into a point cloud
    points = np.column_stack((random_x, random_y, random_z))
    # Step 2: Apply RANSAC regression
    ransac: RANSACRegressor = linear_model.RANSACRegressor(linear_model.LinearRegression())
    ransac.fit(points[:, :2], points[:, 2])  # Fit X (random_x, random_y) to Z (random_z)
    # the plane equation
    z = lambda x, y: (-ransac.estimator_.intercept_ - ransac.estimator_.coef_[0] * x - ransac.estimator_.coef_[1] * y) / \
                     ransac.estimator_.coef_[2]
    a, b = ransac.estimator_.coef_  # Coefficients for x and y
    c = ransac.estimator_.intercept_  # Intercept
    # Step 3: Calculate the normal vector of the fitted plane Plane equation: Ax + By + Cz + D = 0
    # Normal vector is [A, B, C]
    [A, B, C] = a, b, c
    normal_vector = np.array([A, B, C])
    normalized_vector = normalize(normal_vector.reshape(1, -1))[0]
    # Step 4: Calculate the angle between the normal vector and Z-axis
    z_axis = np.array([0, 0, 1])
    cross_product = np.cross(z_axis, normalized_vector)
    sin_theta = np.linalg.norm(cross_product) / (np.linalg.norm(z_axis) * np.linalg.norm(normalized_vector))
    Obj_L = box_l / math.sqrt(1 - math.sin(math.radians(sin_theta)) ** 2)
    text = f"Obj_L: {Obj_L:.2f}"
    cv2.putText(image_input, text, (int(x1) + 5, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
 
cv2_imshow(image_input)
cv2.imwrite("depth.jpg", image_input)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
depth_np_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
inv_depth_np_normalized = 1.0 - depth_np_normalized
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
 
cv2_imshow(depth_colormap)
cv2.imwrite("depth_colormap.jpg", depth_colormap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
####################################3Dcloudprocessing###########################