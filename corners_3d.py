import numpy as np
from scipy.spatial.distance import cdist

# INPUT PATH
piece1_harris_path = "./output/piece-1-harris.txt"
piece2_harris_path = "./output/piece-2-harris.txt"

piece1_images_path = "./data/piece-1-txt/txt/images.txt"
piece2_images_path = "./data/piece-2-txt/txt/images.txt"

piece1_points_path = "./data/piece-1-txt/txt/points3D.txt"
piece2_points_path = "./data/piece-2-txt/txt/points3D.txt"

piece1_paths = [piece1_harris_path, piece1_images_path, piece1_points_path]
piece2_paths = [piece2_harris_path, piece2_images_path, piece2_points_path]

# OUTPUT PATH
piece1_2d_colmap_path = "./output/piece-1-colmap-2d.txt"
piece2_2d_colmap_path = "./output/piece-2-colmap-2d.txt"

piece1_3d_corners_path = "./output/piece-1-corner-3d.txt"
piece2_3d_corners_path = "./output/piece-2-corner-3d.txt"

def read_harris_corners_txt(txt_path): 
    ''' 
    Read image name and Harris corner coordinates from text file.
    '''
    harris_corners = dict() 

    with open(txt_path) as file: 
        for line in file: 
            img_name = line.split()[0] 
            corners = line.split()[1:]

            harris_corners[img_name] = [] 

            for i in range(0, len(corners), 2):
                harris_x, harris_y = float(corners[i]), float(corners[i+1])
                harris_corners[img_name].append((harris_x, harris_y))

    return harris_corners

def read_colmap_images_txt(txt_path): 
    ''' 
    Read image name, pixels, and point id from COLMAP images.txt file.
    '''
    f = open(txt_path,'r')
    lines = f.readlines()[4:] # data starts from 5th line in COLMAP images.txt

    colmap_pixels = dict() 
    for i in range(len(lines)): 
        if i % 2 == 0: 
            img_name = lines[i].split()[-1]
            colmap_pixels[img_name] = [] 
        else:
            pixels = lines[i].split()
            for j in range(0, len(pixels), 3): 
                pixel_x, pixel_y, point_id = float(pixels[j]), float(pixels[j+1]), int(pixels[j+2])
                if point_id != -1: 
                    colmap_pixels[img_name].append((pixel_x, pixel_y, point_id))

    return colmap_pixels

def read_colmap_points3d_txt(txt_path): 
    ''' 
    Read point id, point coords and colors from COLMAP points3D.txt file.
    '''
    f = open(txt_path, 'r')
    lines = f.readlines()[3:] # data starts from 4th line in COLMAP points3D.txt 

    colmap_3d = dict() 
    for line in lines: 
        point_id = int(line.split()[0])
        x, y, z, r, g, b = [float(i) for i in line.split()[1:7]]
        colmap_3d[point_id] = [x, y, z, r, g, b]
    
    return colmap_3d

def nearest_colmap_pixel(harris_corners, colmap_pixels): 
    ''' 
    Find the nearest pixel in COLMAP images.txt file for each Harris corner.
    '''
    colmap_2d = dict()

    for img_name, corners in harris_corners.items(): 
        if not img_name in colmap_pixels: 
            continue

        pixels = colmap_pixels[img_name]

        # find pairwise distance matrix between harris corners vs colmap pixels
        dist_mat = cdist(np.array(corners, dtype=np.float32), np.array(pixels, dtype=np.float32)[:, :2], metric='euclidean')

        # find nearest colmap pixels to each corner
        nearest_pixels_id = np.unique(np.argmin(dist_mat, axis=1))
        nearest_pixels = np.array(pixels)[nearest_pixels_id]

        colmap_2d[img_name] = nearest_pixels

    return colmap_2d

def colmap_2d_to_3d(colmap_2d, colmap_3d): 
    ''' 
    Map 2D pixels from COLMAP images.txt file to 3D corners in COLMAP points3D.txt
    '''
    corners_3d = [] 
    for img_name, pixels in colmap_2d.items(): 
        for pixel in pixels: 
            x, y, point_id = pixel
            if point_id not in colmap_3d: 
                continue 
            point = colmap_3d[point_id]
            corners_3d.append(point)

    return corners_3d

def write_3d_corners(corners_3d, output_path): 
    ''' 
    Write 3d corners extracted to txt file.
    '''
    res = "" 
    for corner in corners_3d: 
        res += " ".join([str(i) for i in corner])
        res += "\n"

    with open(f'{output_path}', "w") as f: 
        f.write(res)
    
def extract_3d_corners(piece1_paths, piece2_paths):
    for paths in [piece1_paths, piece2_paths]: 
        harris_path, img_path, points_path = paths

        harris_corners = read_harris_corners_txt(harris_path)
        colmap_pixels = read_colmap_images_txt(img_path)
        colmap_3d = read_colmap_points3d_txt(points_path)

        colmap_2d = nearest_colmap_pixel(harris_corners, colmap_pixels)
        corners_3d = colmap_2d_to_3d(colmap_2d, colmap_3d)

        output_path = piece1_3d_corners_path if paths == piece1_paths else piece2_3d_corners_path 
        write_3d_corners(corners_3d, output_path)

if __name__ == "__main__": 
    extract_3d_corners(piece1_paths, piece2_paths)
