import os
import idx2numpy
from PIL import Image
from tqdm import tqdm
 
def save_images(images, labels, target_dir):
    for label in range(10):
        label_dir = os.path.join(target_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
 
        # 获取当前标签的所有图像
        label_images = images[labels == label]
 
        # 为当前标签的每张图像显示进度条
        for i, img in enumerate(tqdm(label_images, desc=f"Processing {target_dir}/{label}", ascii=True)):
            img_path = os.path.join(label_dir, f"{i}.jpg")
            img = Image.fromarray(img)
            img.save(img_path)
 
#! MNIST数据集文件路径
train_img_path = r'/Users/kinghua/mnist/train-images.idx3-ubyte'
train_lbl_path = r'/Users/kinghua/mnist/train-labels.idx1-ubyte'
test_img_path = r'/Users/kinghua/mnist/t10k-images.idx3-ubyte'
test_lbl_path = r'/Users/kinghua/mnist/t10k-labels.idx1-ubyte'
 
# 读取数据集
train_images = idx2numpy.convert_from_file(train_img_path)
train_labels = idx2numpy.convert_from_file(train_lbl_path)
test_images = idx2numpy.convert_from_file(test_img_path)
test_labels = idx2numpy.convert_from_file(test_lbl_path)
 
# 保存图像
save_images(train_images, train_labels, 'train')
save_images(test_images, test_labels, 'test')