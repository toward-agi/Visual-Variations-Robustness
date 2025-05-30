import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PIL.Image import Resampling
from datasets import Dataset, DatasetDict, Features, Value, Image as DatasetImage
from io import BytesIO
from tqdm import tqdm

obj_dir = "obj_imgs"
bg_dir = "bg_imgs"

# photo_I_path_list = ['shiba.png','cat.png', 'bear.png', 'eagle.png', 'snake.png', 'panda.png', 'turtle.png', 'fish.png', 'car.png', 'plane.png']
# object_name_list = ['shiba,dog', 'cat', 'bear', 'eagle', 'snake', 'panda', 'turrle', 'fish', 'car', 'plane']
# original_direction_list = ['up','up','up','up','up','up','up','up','up','up']

photo_I_path_list = ['shiba.png']
object_name_list = ['shiba,dog']
original_direction_list = ['up']

# 这三个是一起的，都是关于object本身
background_image_config_list = [
    ("color", "white"),
    # ("color", "red"),
    # ("color", "blue"),
    # ("color", "green"),
    # ("color", "yellow"),
    # ("image", "hongkong.jpg"),
    # ("image", "NY.jpg"),
    ("image", "SF.jpg"),
    # ("image", "valley.jpg"),
    # ("image", "sky.jpg")
    # (type, value)
]
color_to_value = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0)
}

# background_size_list = [(512, 512), (768, 768), (1024, 1024)]
background_size_list = [(512, 512)]

ratio_list = [2,3,5,10,15,20]


def create_dataset(photo_I_path, object_name, original_direction, bg_type, bg_value, background_size, ratio, output_folder):

    # 这些都整成一个循环来测试模型
    # 输入参数
    # photo_I_path = 'shiba.png'  # 照片 I 的路径
    # object_name = 'shiba,dog'  # 照片 I 中的物体名称
    # original_direction = "up"
    background_B_path = None  # 背景 B 的路径，如果没有则设置为 None
    # bg_type = "white"
    # background_size = (512, 512)  # 背景的大小
    # ratio = 5  # 背景相对于照片 I 的大小倍数
    transparency = 0.8  # 照片 I 的透明度（1.0 为不透明，0.0 为完全透明）
    # output_folder = 'object_dataset'  # 输出文件夹名称
    filters_to_apply = ['NONE']  # 要应用的滤镜列表

    dataset_type = "ratio-" + str(ratio) + "_" + "bg" + str(background_size).replace(', ','-').replace('(','').replace(')','') + "-bgtype-" + bg_type \
                    + object_name + bg_value.replace('.jpg','').replace('/','-')
    
    print(dataset_type)
    
    save_dir = os.path.join(output_folder, dataset_type)
    if os.path.exists(save_dir):
        return

    # 加载照片 I
    photo_I = Image.open(photo_I_path).convert('RGBA')

    # 加载或创建背景 B
    if bg_type == 'color':
        background_B = Image.new('RGBA', background_size, color_to_value[bg_value])
        background_type = {'type': 'color', 'value': bg_value}
    elif bg_type == 'image':
        background_B = Image.open(bg_value).convert('RGBA').resize(background_size, Resampling.LANCZOS)
        background_type = {'type': 'image', 'value': bg_value}
    # TODO: 这里是修改了background_type的，之后得修改一下。
    # if background_B_path and os.path.exists(background_B_path):
    #     background_B = Image.open(background_B_path).convert('RGBA').resize(background_size, Resampling.LANCZOS)
    #     background_type = {'image_path': background_B_path,'color':""}
    # else:
    #     background_B = Image.new('RGBA', background_size, (255, 255, 255))
    #     background_type = {'image_path':"",'color': str((255, 255, 255))}

    # 将背景图像转换为 bytes
    background_buffer = BytesIO()
    background_B.save(background_buffer, format='PNG')
    background_image_bytes = background_buffer.getvalue()

    # 计算照片 I 的尺寸
    width_B, height_B = background_B.size
    photo_I_width = width_B // ratio
    photo_I_height = height_B // ratio

    # 调整照片 I 的大小
    photo_I = photo_I.resize((photo_I_width, photo_I_height), Resampling.LANCZOS)

    # 保存输入照片的尺寸
    input_photo_size = f'({photo_I_width}, {photo_I_height})'

    # 调整照片 I 的透明度
    alpha = photo_I.split()[3]
    alpha = alpha.point(lambda p: p * transparency)
    photo_I.putalpha(alpha)

    # 定义旋转角度
    rotations = [0, 45, 90, 135, 180, 225, 270, 315]

    # 定义位置坐标
    positions = []
    for j in range(ratio):
        for i in range(ratio):
            positions.append((i, j))

    # 自动计算步长
    if ratio > 1:
        step_x = (width_B - photo_I_width) / (ratio - 1)
        step_y = (height_B - photo_I_height) / (ratio - 1)
    else:
        step_x = 0
        step_y = 0

    # 定义滤镜函数
    def apply_filter(image, filter_name):
        if filter_name == 'NONE':
            return image
        elif filter_name == 'GRAYSCALE':
            return ImageOps.grayscale(image).convert('RGBA')
        elif filter_name == 'SEPIA':
            sepia_image = ImageOps.colorize(ImageOps.grayscale(image), '#704214', '#C0A080')
            return sepia_image.convert('RGBA')
        elif filter_name == 'INVERT':
            r, g, b, a = image.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            inverted_image = ImageOps.invert(rgb_image)
            r2, g2, b2 = inverted_image.split()
            return Image.merge('RGBA', (r2, g2, b2, a))
        elif filter_name == 'BLUR':
            return image.filter(ImageFilter.BLUR)
        elif filter_name == 'SHARPEN':
            return image.filter(ImageFilter.SHARPEN)
        elif filter_name == 'EDGE_ENHANCE':
            return image.filter(ImageFilter.EDGE_ENHANCE)
        else:
            return image  # 未知滤镜，返回原图

    # 计算总的迭代次数用于进度条
    total_iterations = len(positions) * len(rotations) * len(filters_to_apply)

    # 生成图片
    image_data_list = []
    progress_bar = tqdm(total=total_iterations, desc='Processing images')

    for pos in positions:
        i, j = pos
        pos_x = i * step_x
        pos_y = j * step_y
        for angle in rotations:
            rotated_I = photo_I.rotate(angle, expand=True)
            rotated_width, rotated_height = rotated_I.size
            # 调整位置，使旋转后的图像中心对准原位置
            adjusted_pos_x = pos_x + (photo_I_width - rotated_width) / 2
            adjusted_pos_y = pos_y + (photo_I_height - rotated_height) / 2
            for filter_name in filters_to_apply:
                filtered_I = apply_filter(rotated_I, filter_name)
                # 创建背景的副本
                composite_image = background_B.copy()
                # 将照片 I 粘贴到背景上
                composite_image.paste(filtered_I, (int(adjusted_pos_x), int(adjusted_pos_y)), filtered_I)
                # 将合成图像转换为 bytes
                image_buffer = BytesIO()
                composite_image.save(image_buffer, format='PNG')
                image_bytes = image_buffer.getvalue()
                # 记录图片属性
                image_data = {
                    'question': '',
                    'image': image_bytes,
                    'object_name': object_name,
                    'original_direction': original_direction,
                    'position': f'({i}, {j})',
                    'rotation_angle': angle,
                    'filter': filter_name,
                    'transparency': transparency,
                    'ratio': ratio,
                    'background_type': background_type,
                    'background_size': f'({height_B}, {width_B})',
                    'input_photo_size': input_photo_size,
                    # 'background_image': background_image_bytes,
                }
                image_data_list.append(image_data)
                progress_bar.update(1)
    progress_bar.close()

    # 定义特征
    features = Features({
        'question': Value('string'),
        'image': DatasetImage(),  # 合成图片
        'object_name': Value('string'),
        'original_direction': Value('string'),
        'position': Value('string'),
        'rotation_angle': Value('int32'),
        'filter': Value('string'),
        'transparency': Value('float32'),
        'ratio': Value('int32'),
        'background_type': {
            'type': Value('string'),
            'value': Value('string'),
        },
        'background_size': Value('string'),
        'input_photo_size': Value('string'),
        # 'background_image': DatasetImage(),  # 背景图片
    })

    # 创建数据集
    dataset = Dataset.from_list(image_data_list, features=features)

    # 创建 DatasetDict 并添加到 'test' split
    dataset_dict = DatasetDict({'test': dataset})

    # 保存数据集
    dataset_dict.save_to_disk(os.path.join(output_folder, dataset_type))

for object_img_path, object_name, original_direction in zip(photo_I_path_list, object_name_list, original_direction_list):
    object_img_path = os.path.join(obj_dir, object_img_path)
    print(object_img_path)
    for bg_type, bg_value in background_image_config_list:
        if bg_type == "image":
            bg_value = os.path.join(bg_dir, bg_value)
        for bg_size in background_size_list:
            for ratio in ratio_list:
                create_dataset(
                    photo_I_path=object_img_path,
                    object_name=object_name,
                    original_direction=original_direction,
                    bg_type=bg_type,
                    bg_value=bg_value,
                    background_size=bg_size,
                    ratio=ratio,
                    output_folder='eval_dataset/object_dataset'
                )
