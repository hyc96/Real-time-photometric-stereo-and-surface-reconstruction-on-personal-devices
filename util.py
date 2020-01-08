import cv2
import numpy as np
import screeninfo
import os
from PIL import Image
from scipy.sparse.linalg import svds
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from mayavi import mlab

INPUT_DIR = "input/"
NUM_IN = 4
IM_RATIO = 1

def load_image(fname):
    im = Image.open(fname).convert(mode="L")
    new_w = im.size[0] // IM_RATIO
    new_h = im.size[1] // IM_RATIO
    im = im.resize((new_w, new_h), Image.ANTIALIAS)
    return np.asarray(im)

def capture(cap,img):
    '''
    img: input image number
    cap: video capture

    '''
    cam, frame = cap.read()
    if(not cam):
        print('Error: webcam not running!')
    cv2.imwrite(filename=INPUT_DIR + str(img) + '.jpg', img=frame)


def display(index):
    screen_id = 0
    is_color = False

    # get the size of the screen
    screen = screeninfo.get_monitors("osx")[screen_id]
    width, height = int(screen.width), int(screen.height)
 
    # create image
    image = np.zeros((height, width))
    mid_height = height // 2
    mid_width = width // 2
    if index == 1:
        for i in range(0, height):
            for j in range(0, mid_width):
                image[i][j] = 255
    elif index == 2:
        for i in range(0, mid_height):
            for j in range(0, width):
                image[i][j] = 255
    elif index == 3:
        for i in range(0, height):
            for j in range(mid_width, width):
                image[i][j] = 255
    elif index == 4:
        for i in range(mid_height, height):
            for j in range(0, width):
                image[i][j] = 255

    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
    cv2.waitKey(40)
    
def get_image():
    cap = cv2.VideoCapture(0)
    for i in range(NUM_IN):
        display(i + 1)
        capture(cap, i + 1)      
    cap.release()
    cv2.destroyAllWindows()

def get_input():
    ims = ()
    for i in range(NUM_IN):
        imarray = load_image(INPUT_DIR+str(i+1) + ".jpg")
        imx = imarray.shape[0]
        imy = imarray.shape[1]
        I = np.reshape(imarray, (imx * imy, 1),order="C")
        ims += (I,)
    return np.column_stack(ims), (imx, imy)
        

def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    imx = surface_normals.shape[1]
    imy = surface_normals.shape[0] #flipped for indexing
    #height_map = np.zeros((imx,imy)
    fx = surface_normals[:,:,0] / surface_normals[:,:,2]
    fy = surface_normals[:,:,1] / surface_normals[:,:,2]
    fy = np.nan_to_num(fy)
    fx = np.nan_to_num(fx)
    row = np.cumsum(fx,axis=1)
    column = np.cumsum(fy,axis=0)
    if integration_method == 'row':
        row_temp = np.vstack([row[0,:]]*imy)
        height_map = column + row_temp     
        #print(np.max(height_map))
    if integration_method == 'column':
        col_temp = np.stack([column[:,0].T]*imx,axis=1)
        height_map = row + col_temp   
        #print(height_map.T)
    if integration_method == 'average':
        row_temp = np.vstack([row[0,:]]*imy)
        col_temp = np.stack([column[:,0].T]*imx,axis=1)
        height_map = (row + column + row_temp + col_temp) / 2
        
    if integration_method == 'random':
        iteration = 10
        height_map = np.zeros((imy,imx))
        for x in range(iteration):
            print(x)
            for i in range(imy):
                print(i)
                for j in range(imx):
                    id1 = 0
                    id2 = 0
                    val = 0
                    path = [0] * i + [1] * j
                    random.shuffle(path)
                    for move in path:
                        if move == 0:
                            id1 += 1
                            if id1 > imy - 1: id1 -= 1
                            val += fy[id1][id2]
                            #print(val,fx[id1][id2])
                        if move == 1:
                            id2 += 1
                            if id2 > imx - 1: id2 -= 1
                            val += fx[id1][id2]
                    height_map[i][j] += val
                    #print(i,j,val)
        height_map = height_map / iteration
        #print(np.max(height_map))
    # print(height_map)
    return height_map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

# def display_output(albedo_image, height_map):
#     fig = plt.figure()
#     plt.imshow(albedo_image, cmap='gray')
#     plt.axis('off')
    
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.gca(projection='3d')
#     ax.view_init(20, 20)
#     X = np.arange(albedo_image.shape[0])
#     Y = np.arange(albedo_image.shape[1])
#     X, Y = np.meshgrid(Y, X)
#     H = np.flipud(np.fliplr(height_map))
#     A = np.flipud(np.fliplr(albedo_image))
#     A = np.stack([A, A, A], axis=-1)
#     A /= np.max(A)
#     ax.xaxis.set_ticks([])
#     ax.xaxis.set_label_text('Z')
#     ax.yaxis.set_ticks([])
#     ax.yaxis.set_label_text('X')
#     ax.zaxis.set_ticks([])
#     ax.yaxis.set_label_text('Y')
#     surf = ax.plot_surface(
#         H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
#     set_aspect_equal_3d(ax)
#     plt.show()

                
def find_surface(N, size, h_map, K = 2000, p = False):
    h, w = size
    ret = h_map
    fx = N[:,:,0] / N[:,:,2]
    fx = np.nan_to_num(fx)
    fy = N[:,:,1] / N[:,:,2]
    fy = np.nan_to_num(fy)
    fxp = np.roll(fx,1,axis = 1)
    fxp[:,0] = 0
    deriv_x = fxp - fx
    fyp = np.roll(fy,1,axis = 0)
    fyp[0,:] = 0
    deriv_y = fyp - fy

    for i in range(K):
        dx = 1
        if p:
            dx = (-15.0 / 49.0) * i + 16
        dy = dx
        left = np.roll(ret,1,axis=1)
        left[:,0] = 0
        right = np.roll(ret,-1,axis=1)
        right[:,-1] = 0
        up = np.roll(ret,1,axis=0)
        up[0,:] = 0
        down = np.roll(ret,-1,axis=0)
        down[-1,:] = 0
        neighbor = 1/4 * (up + left + right + down)
        deriv = 1/4 * (deriv_x * dx + deriv_y * dy)
        ret = neighbor + deriv

    return ret

def surface_normals(I, size):
    h, w = size
    mat = np.matmul(I.T,I).astype(np.float)

    _,_,v = np.linalg.svd(mat)
    B = v[[1,2,0]]
    N = (B@I.T).T

    N = N.reshape(h, w, 3, order="C")
    albedo = np.linalg.norm(N, axis = -1)
    N /= albedo.reshape(h, w, 1)

    # #gradients (partial derivatives)
    # g_x = -N[:,:,0] / N[:,:,2]
    # g_y =  N[:,:,1] / N[:,:,2]
    # g_x = np.nan_to_num(g_x)
    # g_y = np.nan_to_num(g_y)

    h_map = get_surface(N, 'average')
    h_map = find_surface(N, size, h_map)
    mlab.surf(h_map)
    mlab.show()
    # display_output(albedo, h_map)

def threshold(h_map):
    h_map += abs(np.min(h_map))
    # h_map /= np.max(h_map)
    h_map[h_map < np.max(h_map) * 0.6] = np.max(h_map) * 0.6
    return h_map 

def surface_normal(I, size):

    h, w = size
    u, _, _ = svds(I.astype(np.float), k=3)

    surface_normal = u.reshape(h, w, 3, order="C")
    surface_normal = surface_normal[:, :, [1,0,2]]

    albedo = np.linalg.norm(surface_normal, axis = -1)

    surface_normal /= albedo.reshape(h, w, 1)

    surface_normal = np.nan_to_num(surface_normal)
    h_map = get_surface(surface_normal, 'average')
    h_map = find_surface(surface_normal, size, h_map)
    
    h_map = threshold(h_map)
    mlab.surf(h_map)
    mlab.show()
    # display_output(albedo, h_map)

    return surface_normal

def display_frames():
    for i in range(11):
        frame = np.load("./frames/"+str(i)+".npy")
        print(frame)
        mlab.surf(frame)
        mlab.show()


def real_time_stereo():
    cap = cv2.VideoCapture(0)
    for i in range(NUM_IN):
        display(i + 1)
        capture(cap, i + 1)    
    I, size = get_input()
    h, w = size
    u, _, _ = svds(I.astype(np.float), k=3)

    surface_normal = u.reshape(h, w, 3, order="C")
    surface_normal = surface_normal[:, :, [1,0,2]]
    albedo = np.linalg.norm(surface_normal, axis = -1)
    surface_normal /= albedo.reshape(h, w, 1)
    surface_normal = np.nan_to_num(surface_normal)
    h_map = get_surface(surface_normal, 'average')
    h_map = find_surface(surface_normal, size, h_map, 1000)
    h_map = threshold(h_map)
    np.save("./frames/0.npy", h_map)
    
    for i in range(10):
        display(i % NUM_IN + 1)
        capture(cap, i % NUM_IN + 1)
        surface_normal = u.reshape(h, w, 3, order="C")
        surface_normal = surface_normal[:, :, [1,0,2]]
        albedo = np.linalg.norm(surface_normal, axis = -1)
        surface_normal /= albedo.reshape(h, w, 1)
        surface_normal = np.nan_to_num(surface_normal)
        h_map = find_surface(surface_normal, size, h_map, 50, True)
        h_map = threshold(h_map)
        np.save("./frames/"+str(i+1)+".npy", h_map)
    
    cap.release()
    cv2.destroyAllWindows()
