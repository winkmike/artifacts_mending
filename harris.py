import cv2 
import numpy as np
import os 

# PARAMS 
gaussian_blur_kernel_size = (5, 5)
gaussian_blur_sigma_x = 0

canny_threshold1 = 100 
canny_threshold2 = 200 

harris_block_size = 2 
harris_ksize = 3 
harris_k = 0.04
harris_filter_threshold = 0.1

# PATH
piece1_path = "./data/piece-1-images/images/"
piece2_path = "./data/piece-2-images/images/"
output = "./output/"

def read_images(img_path): 
    ''' 
    Get all image paths within a folder. 
    '''
    imgs = [] 
    valid_imgs = [".jpg",".gif",".png"]

    for f in os.listdir(img_path): 
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_imgs: 
            continue 
        imgs.append(os.path.join(img_path, f))
        imgs.sort()

    return imgs

def preprocess_img(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, gaussian_blur_kernel_size, gaussian_blur_sigma_x)

    return img_blur

def canny_edge_detection(img_blur):
    edges = cv2.Canny(image=img_blur, threshold1=canny_threshold1, threshold2=canny_threshold2)
    
    return edges

def harris_corner_detection(edges): 
    dst = cv2.cornerHarris(edges, harris_block_size, harris_ksize, harris_k)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, harris_filter_threshold*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(edges,np.float32(centroids),(5,5),(-1,-1),criteria)

    return corners

def find_harris_corners(piece1_path, piece2_path): 
    imgs_1 = read_images(piece1_path)
    imgs_2 = read_images(piece2_path)

    res = "" 
    for piece_imgs in [imgs_1, imgs_2]:
        for img_path in piece_imgs: 
            img_blur = preprocess_img(img_path)
            edges = canny_edge_detection(img_blur)
            corners = harris_corner_detection(edges)

            # extract image name from path
            img_name = os.path.basename(img_path)
            res += f"{img_name} "
            for corner in corners: 
                res += f"{corner[0]} {corner[1]}" 
                res += " "

            res += "\n"
        
        filename = "piece-1-harris" if piece_imgs == imgs_1 else "piece-2-harris"
        with open(f"{output}{filename}.txt", "w") as f: 
            f.write(res)


if __name__ == "__main__": 
    find_harris_corners(piece1_path, piece2_path)

