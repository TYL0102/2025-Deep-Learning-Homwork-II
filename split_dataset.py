import random
from pathlib import Path

ROOT_DIR = Path("/home/neat/deep_learning_hw2/KSDD2")
TRAIN_DIR = ROOT_DIR / "train/images"
TEST_DIR = ROOT_DIR / "test/images"

def generate_path_lists(val_ratio = 0.1) : 
    # get absolute path of all images from train dir
    all_images = [str(file.absolute()) for file in TRAIN_DIR.glob("*") if file.suffix.lower() == ".png"]

    # shuffle images
    random.shuffle(all_images)

    # split dataset
    val_count = int(len(all_images) * val_ratio)
    val_list = all_images[:val_count]
    train_list = all_images[val_count:]

    # write train_list.txt
    with open(ROOT_DIR / "train_list.txt", "w", encoding = "utf-8") as file : 
        file.write("\n".join(train_list))
    
    # write val_list.txt
    with open(ROOT_DIR / "val_list.txt", "w", encoding = "utf-8") as file : 
        file.write("\n".join(val_list))
    
    # get absolute path of all images from test dir
    test_list = [str(file.absolute()) for file in TEST_DIR.glob("*") if file.suffix.lower() == ".png"]

    # write test_list.txt
    with open(ROOT_DIR / "test_list.txt", "w", encoding = "utf-8") as file : 
        file.write("\n".join(test_list))

    # print result
    print("splitting finished.")
    print(f"train: {len(train_list)} images.")
    print(f"val  : {len(val_list)} images.")

def create_yaml() : 
    # generate yaml content
    yaml_content = "# pointed to generated .txt files\n" + f"train: {(ROOT_DIR / 'train_list.txt').as_posix()}\n" + f"val: {(ROOT_DIR / 'val_list.txt').as_posix()}\n" + f"test: {(ROOT_DIR / 'test_list.txt').as_posix()}\n" + "\n" + "# class\n" + "names:\n" + "  0: defect"
    with open(ROOT_DIR / "data.yaml", "w", encoding = "utf-8") as file : 
        file.write(yaml_content)
    
    # print result
    print("yaml updated.")

if __name__ == "__main__":
    generate_path_lists()
    create_yaml()