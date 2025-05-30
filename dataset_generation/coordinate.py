import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import io
from PIL import Image

def generate_points(start, end, precision, dimension):
    grid = np.arange(start, end + precision, precision)
    if dimension == 1:
        points = [(x,) for x in grid]
    elif dimension == 2:
        points = [(x, y) for x in grid for y in grid]
    elif dimension == 3:
        points = [(x, y, z) for x in grid for y in grid for z in grid]
    else:
        raise ValueError("Unsupported dimension")
    return points

def plot_point(coordinate, dimension, config):
    size_in_inches = config['image_size'][0] / 100  # 将像素转换为英寸
    dpi = 100  # 分辨率
    background_color = config.get('background_color', 'white')  # 默认白色背景

    # 计算坐标轴范围，扩大 1 个单位
    x_min = config['range'][0] - 1
    x_max = config['range'][1] + 1

    if dimension == 1:
        x0 = coordinate[0]
        plt.figure(figsize=(size_in_inches, size_in_inches), dpi=dpi, facecolor=background_color)

        # Plot the point at y=0
        plt.scatter([x0], [0], color='red', s=config['object_size'] * 10)
        plt.axhline(0, color='black', linewidth=0.5)  # Add horizontal axis
        if config['include_reference_lines']:
            plt.axvline(x=x0, color=config['reference_line_color'], linestyle='--', linewidth=1)

        if config['precise_labeling']:
            # plt.xticks(range(x_min, x_max), fontsize=64)
            plt.xticks(range(int(x_min), int(x_max)))

        plt.yticks([])

        # Hide unnecessary spines
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Position the x-axis at y=0
        plt.gca().spines['bottom'].set_position(('data', 0))

        plt.xlim(x_min, x_max)
        # Adjust y-limits to a small range around zero to keep the point visible
        plt.ylim(-1, 1)

        # Add title, labels, legend, and grid
        plt.title("1D Coordinate System")
        plt.xlabel("X-axis")
        # plt.legend(loc='upper right')
        if config.get('include_grid', False):
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            plt.grid(False)
        plt.gca().set_aspect('auto')  # Adjust aspect ratio as needed

    elif dimension == 2:
        # 2D 绘图逻辑
        x0, y0 = coordinate
        plt.figure(figsize=(size_in_inches, size_in_inches), dpi=dpi, facecolor=background_color)

        # 绘制点
        plt.scatter([x0], [y0], color='red', s=config['object_size'] * 10)
        plt.axhline(0, color='black', linewidth=0.5)  # 添加 x 轴
        plt.axvline(0, color='black', linewidth=0.5)  # 添加 y 轴

        if config['include_reference_lines']:
            plt.axhline(y=y0, color=config['reference_line_color'], linestyle='--', linewidth=1)  # 添加水平参考线
            plt.axvline(x=x0, color='green', linestyle='--', linewidth=1)  # 添加垂直参考线

        # 设置坐标轴范围
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)  # 假设 y 轴范围与 x 轴一致，保持正方形

        # 添加网格
        if config.get('include_grid', False):
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            plt.grid(False)

        # 添加标题和图例
        plt.title(f"2D Coordinate System") #FIXME:
        plt.xlabel("X-axis") #FIXME:
        plt.ylabel("Y-axis") #FIXME:
        # plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')  # 保持背景正方形比例

        # 设置刻度
        if config['precise_labeling']:
            plt.xticks(range(int(x_min), int(x_max)), fontsize=12) #FIXME:
            plt.yticks(range(int(x_min), int(x_max)), fontsize=12) #FIXME:

    elif dimension == 3:
        # 3D 绘图逻辑
        x0, y0, z0 = coordinate
        fig = plt.figure(figsize=(size_in_inches, size_in_inches), dpi=dpi, facecolor=background_color)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点
        ax.scatter(x0, y0, z0, c='r', marker='o', s=config['object_size'] * 20)

        # 标注点
        ax.text(x0, y0, z0, f'({x0}, {y0}, {z0})', color='red')

        if config['include_reference_lines']:
            # 添加参考线
            ax.plot([x0, x0], [y0, y0], [x_min, z0], linestyle="--", color=config['reference_line_color'])
            ax.plot([x0, x0], [x_min, y0], [z0, z0], linestyle="--", color='green')
            ax.plot([x_min, x0], [y0, y0], [z0, z0], linestyle="--", color='purple')

        # 设置坐标轴标签和范围
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.set_zlim(x_min, x_max)  # 假设 z 轴范围与 x 和 y 轴一致

        # 添加图例和标题
        ax.legend()
        plt.title('3D Coordinate System')

        if config['include_grid']:
            ax.grid(True)  
        else:
            ax.grid(False)

        # 设置刻度
        if config['precise_labeling']:
            ticks = range(int(x_min), int(x_max))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_zticks(ticks)

    # 保存为 PIL 图像
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=background_color)
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def generate_dataset(config):
    dimensions = config['dimensions']
    data_list = []
    total_images = 0

    # 计算总图像数量以更新进度条
    for dim in dimensions:
        num_points = len(generate_points(config['range'][0], config['range'][1], config['precision'], dim))
        total_images += num_points

    with tqdm(total=total_images, desc="Generating Dataset") as pbar:
        for dim in dimensions:
            points = generate_points(config['range'][0], config['range'][1], config['precision'], dim)
            for point in points:
                # 生成图像并添加到数据集中
                pil_image = plot_point(point, dim, config)
                data_element = {
                    "image": pil_image,
                    "coordinate": str(point),
                    "question": "",
                    "dimension": dim,
                    "range": str(config['range']),
                    "precision": config['precision'],
                    "precise_labeling": config['precise_labeling'],
                    "object_to_place": config['object_to_place'],
                    "object_size": config['object_size'],
                    "background_color": config['background_color'],
                    "reference_line_color": config['reference_line_color'],
                    "image_size": str(config['image_size']),
                    "include_reference_lines": config['include_reference_lines'],
                    "include_grid": config['include_grid']  # 添加 include_grid
                }
                data_list.append(data_element)
                pbar.update(1)

    # 创建包含测试集的数据集
    dataset = DatasetDict({
        "test": Dataset.from_list(data_list)
    })

    dataset_type = "range-" + str(config['range']).replace('(','').replace(')','').replace(', ','-') + "img_size-" \
                     + str(config['image_size'][0]) 
    if config['include_reference_lines']:
        dataset_type = dataset_type + "-reference"
    if config['include_grid']:
        dataset_type = dataset_type + "-grid"

    # 保存数据集
    dataset.save_to_disk(os.path.join('eval_dataset/coordinate_dataset', dataset_type))

if __name__ == "__main__":

    # image_size_list = [(512, 512), (768, 768), (1024, 1024)]
    image_size_list = [(512, 512)]
    range_list = [(-5,5),(-10,10),(0,10),(0,20)]
    reference_list = [True, False]
    grid_list = [True, False]
    for image_size in image_size_list:
        for ra in range_list:
            for reference in reference_list:
                for grid in grid_list:
                    config = {
                        'dimensions': [1, 2],  # 如需添加三维，可以在此处添加 3 # 3d的有必要测吗？
                        'range': ra,
                        'precision': 1,
                        'precise_labeling': True, #这个参数没有意义，无需关注
                        'object_to_place': 'point',
                        'object_size': 8,
                        'background_color': 'white',
                        'reference_line_color': 'blue',
                        'image_size': image_size,
                        'include_reference_lines': reference,
                        'include_grid': grid  # 添加 include_grid 选项
                    }
                    generate_dataset(config)

#     colors = [
#     'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
#     'gray', 'darkgray', 'lightgray', 'brown', 'orange', 'pink', 'purple',
#     'gold', 'lime', 'navy', 'teal', 'indigo', 'violet', 'beige', 'azure',
#     'ivory', 'maroon', 'olive', 'chocolate', 'coral', 'crimson', 'orchid',
#     'plum', 'salmon', 'silver', 'tan', 'turquoise', 'wheat'
# ]
