from ultralytics import RTDETR

def train_model() : 
    # load RT-DETR pre-trained model
    # use version-l (Large), fit well training on RTX4080
    model = RTDETR("rtdetr-l.pt")

    # start training
    model.train(
        # basic settings
        data = "/home/neat/deep_learning_hw2/KSDD2/data.yaml", # yaml containing the data settings
        epochs = 100,                                          # max epochs
        imgsz = 640,                                           # image size (as a square)
        batch = 16,                                            # batch size
        rect = True,                                           # rectangle training
        cache = "disk",                                        # cache images
        workers = 4,                                           # working processes

        # output settings
        project = "KSDD2_Detection",                           # project name
        name = "rtdetr_l_run",                                 # run name
        save = True,                                           # save best.pt and last.pt
        patience = 20,                                         # stop if mAP stop increasing for 20 epochs
        deterministic = False,                                 # alow PyTorch use undeterministic algorithm to speedup training

        # model settings
        lr0 = 0.0002,                                          # initial learning rate
        amp = True,                                            # auto mixed precision
        warmup_epochs = 5.0,                                   # warmup epoch (avoid gradient explosion)
        optimizer = "AdamW",                                   # use AdamW optimizer

        # data augmentation
        flipud = 0.5,                                          # flip up/down (50%)
        fliplr = 0.5,                                          # flip left/right (50%)
        mosaic = 1.0                                           # mosaic augmentation (splice 4 images into one)
    )

if __name__ == "__main__" : 
    train_model()