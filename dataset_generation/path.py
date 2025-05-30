import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import io
from PIL import Image

def generate_paths(num_paths, num_points, coord_range, unit_length):
    paths = []
    possible_coords = np.arange(coord_range[0], coord_range[1] + unit_length, unit_length)
    possible_coords = possible_coords.astype(int)
    for _ in range(num_paths):
        path = []
        points_set = set()
        while len(points_set) < num_points:
            x = random.choice(possible_coords)
            y = random.choice(possible_coords)
            point = (x, y)
            points_set.add(point)
        path = list(points_set)
        if path[0] == path[-1]:
            while True:
                x = random.choice(possible_coords)
                y = random.choice(possible_coords)
                new_end_point = (x, y)
                if new_end_point != path[0]:
                    path[-1] = new_end_point
                    break
        paths.append(path)
    return paths

def plot_path(path, config, image_size):
    plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor(config['background_color'])
    x_coords, y_coords = zip(*path)
    
    plt.plot(x_coords, y_coords, marker='o', color='blue', linestyle='-')
    plt.plot(x_coords[0], y_coords[0], marker=(5,1,0), markersize=config['marker_size']*1.5, color='green', label='Start')
    plt.plot(x_coords[-1], y_coords[-1], marker='^', markersize=config['marker_size']*1.5, color='red', label='End')
    
    if len(path) > 2:
        plt.plot(x_coords[1:-1], y_coords[1:-1], 'o', markersize=config['marker_size'], color='black', label='Intermediate')
    
    if config.get('include_reference_lines', False):
        for x, y in path:
            ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    
    plt.xlim(config['range'][0], config['range'][1])
    plt.ylim(config['range'][0], config['range'][1])
    
    if config['show_grid']:
        plt.grid(True)
    else:
        plt.grid(False)
    
    if config['show_axes']:
        ticks = np.arange(config['range'][0], config['range'][1]+config['unit_length'], config['unit_length'])
        plt.xticks(ticks, fontsize=32) #FIXME: fontsize
        plt.yticks(ticks, fontsize=32) #FIXME: 
    else:
        plt.xticks([])
        plt.yticks([])
    
    ax.set_xlabel(f"X-axis (Unit: {config['unit_length']})", fontsize=32) #FIXME: 
    ax.set_ylabel(f"Y-axis (Unit: {config['unit_length']})", fontsize=32) #FIXME: 
    
    #plt.legend(fontsize=32) #FIXME:
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def generate_dataset(config):
    paths = generate_paths(config['num_charts'], config['num_points'], config['range'], config['unit_length'])
    
    for img_size in config['image_size']:
        data_list = []
        with tqdm(total=len(paths), desc=f"Generating Dataset for image size {img_size}") as pbar:
            for path in paths:
                pil_image = plot_path(path, config, img_size)
                data_element = {
                    "image": pil_image,
                    "path": str(path),
                    "num_points": len(path),
                    "unit_length": config['unit_length'],
                    "range": str(config['range']),
                    "background_color": config['background_color'],
                    "image_size": str(img_size),
                    "marker_size": config['marker_size'],
                    "show_grid": config['show_grid'],
                    "show_axes": config['show_axes'],
                    "include_reference_lines": config['include_reference_lines']
                }
                data_list.append(data_element)
                pbar.update(1)
        
        # 创建数据集
        dataset = DatasetDict({
            "test": Dataset.from_list(data_list)
        })
        
        # 为每个图像尺寸创建单独的目录
        dataset_type = ("num-points" + str(config['num_points']) + 
                        "-range-" + str(config['range']).replace('(','').replace(')','').replace(', ','-') + 
                        '-img-size-' + str(img_size[0]))
        save_path = os.path.join('eval_dataset/path_dataset', dataset_type)
        
        # 保存数据集
        dataset.save_to_disk(save_path)

if __name__ == "__main__":
    for num_p in [2,3,4,5,6]:
        for ra in [(-5, 5), (-10, 10), (0, 10), (0,20)]:
            config = {
                'num_charts': 100,
                'num_points': num_p,
                'range': ra,
                'unit_length': 1,
                'background_color': 'white',
                # 'image_size': [(1024, 1024), (768, 768), (512, 512)],  # 可以在这里添加多个尺寸
                'image_size': [(512, 512)],
                'marker_size': 8,
                'show_grid': True,
                'show_axes': True,
                'include_reference_lines': True
            }
            generate_dataset(config)
            # num_points: 2,3,4,5,6
            # range: -5,5,-10,10,0,10,0,20
            # image size
