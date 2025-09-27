import os
import shutil
import random

def split_dataset(src_dir, dest_dir, split_ratio=0.8):
    # فولدر train و test
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # نعدي على كل فولدر (كلاس)
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # الصور
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        # نعمل فولدرات الكلاسات
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # ننسخ الصور
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

        print(f"✅ {class_name}: {len(train_images)} train, {len(test_images)} test")

# شغل الكود هنا
src_dir = r"C:\Users\kirolosAnwar\Desktop\AMIT\archive (1)\asl_alphabet_train"
dest_dir = r"C:\Users\kirolosAnwar\Desktop\AMIT\data"
split_dataset(src_dir, dest_dir, split_ratio=0.8)
