import random

import numpy as np


def train_test(quadrant, canvas_size, pad, train_num, test_num):
    quadrant_range = {1 : {'x' : (canvas_size // 2, canvas_size-pad), 'y' : (pad, canvas_size // 2)}, 
                      2 : {'x' : (pad, canvas_size // 2), 'y' : (pad, canvas_size // 2)}, 
                      3 : {'x' : (pad, canvas_size // 2), 'y' : (canvas_size // 2, canvas_size-pad)}, 
                      4 : {'x' : (canvas_size // 2, canvas_size-pad), 'y' : (canvas_size // 2, canvas_size-pad)}}
    
    train_coords, test_coords = [], []
    train_imgs, test_imgs = [], []

    if quadrant:
        for i in range(train_num):
            while True:
                x, y = random.randint(pad, canvas_size-pad-1), random.randint(pad, canvas_size-pad-1)
                x_min, x_max = quadrant_range[quadrant]['x']
                y_min, y_max = quadrant_range[quadrant]['y']

                if (x_min <= x < x_max) and (y_min <= y < y_max): continue
                else: break

            img = np.array([[0] * canvas_size for _ in range(canvas_size)])
            img[y, x] = 255
            
            train_coords.append((x, y))
            train_imgs.append(img)

        for i in range(test_num):
            while True:
                x, y = random.randint(pad, canvas_size-pad-1), random.randint(pad, canvas_size-pad-1)
                x_min, x_max = quadrant_range[quadrant]['x']
                y_min, y_max = quadrant_range[quadrant]['y']

                if (x_min <= x < x_max) and (y_min <= y < y_max): break
                else: continue

            img = np.array([[0] * canvas_size for _ in range(canvas_size)])
            img[y, x] = 255

            test_coords.append((x, y))
            test_imgs.append(img)

    else:
        for i in range(train_num):
            x, y = random.randint(pad, canvas_size-pad-1), random.randint(pad, canvas_size-pad-1)

            img = np.array([[0] * canvas_size for _ in range(canvas_size)])
            img[y, x] = 255

            train_coords.append((x, y))
            train_imgs.append(img)

        for i in range(test_num):
            x, y = random.randint(pad, canvas_size-pad-1), random.randint(pad, canvas_size-pad-1)

            img = np.array([[0] * canvas_size for _ in range(canvas_size)])
            img[y, x] = 255

            test_coords.append((x, y))
            test_imgs.append(img)
        
    return train_coords, test_coords, train_imgs, test_imgs