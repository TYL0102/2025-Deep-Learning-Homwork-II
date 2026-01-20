import tkinter as tk
from tkinter import filedialog # select files
from PIL import Image, ImageTk # image processing
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from ultralytics import RTDETR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class KSDD2App : 
    def __init__(self, root) : 
        self.root = root
        self.root.title("KSDD2 Defect Detection")
        self.root.geometry("1200x800")

        self.current_img_path = None
        self.det_model = None
        self.seg_model = None

        self.init_models()

        self.setup_ui()

    def init_models(self) : 
        # Segmentation
        self.seg_model = smp.Unet(
            encoder_name = "resnet50", 
            encoder_weights = None, 
            in_channels = 3, 
            classes = 1
        ).to(DEVICE)
        self.seg_model.load_state_dict(torch.load("best_unet_ksdd2.pth", map_location = DEVICE, weights_only = True))
        self.seg_model.eval()

        self.seg_transform = A.Compose([
            # letterbox logic
            A.LongestMaxSize(max_size = 640),  # resize long side to 640
            A.PadIfNeeded(                     # padding with black pixel
                min_height = 640, 
                min_width = 640, 
                border_mode = 0, 
                fill = (0, 0, 0), 
                fill_mask = 0
            ), 
            
            # normalize
            A.Normalize(mean = (0, 0, 0), std = (1, 1, 1)), 
            ToTensorV2(), 
        ])

        # Detection
        self.det_model = RTDETR("KSDD2_Detection/rtdetr_l_run/weights/best.pt")

    def setup_ui(self) : 
        # top frame (buttons)
        top_frame = tk.Frame(self.root, bg = "#dcdcdc", height = 80)
        top_frame.pack(side = tk.TOP, fill = tk.X)

        btn_load = tk.Button(top_frame, text = "Load Image", font = ("Arial", 14), padx = 20, pady = 10, command = self.load_image)
        btn_load.pack(side = tk.LEFT, padx = 20, pady = 20)

        btn_inf = tk.Button(top_frame, text = "Inference", font = ("Arial", 14), padx = 20, pady = 10, command = self.inference)
        btn_inf.pack(side = tk.LEFT, padx = 20, pady = 20)

        # bottom frame (display)
        display_frame = tk.Frame(self.root)
        display_frame.pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

        frame_org = tk.Frame(display_frame, relief = "sunken")
        frame_org.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        tk.Label(frame_org, text = "Original", font = ("Arial", 14, "bold")).pack(side = tk.TOP)
        self.lbl_org = tk.Label(frame_org, text = "[ Image Area ]", font = ("Arial", 14), bg = "gray")
        self.lbl_org.pack(expand = True, fill = tk.BOTH, padx = 5, pady = 5)

        frame_det = tk.Frame(display_frame, relief="sunken")
        frame_det.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        tk.Label(frame_det, text = "Object Detection", font = ("Arial", 14, "bold")).pack(side = tk.TOP)
        self.lbl_det = tk.Label(frame_det, text = "[ Image Area ]", font = ("Arial", 14), bg = "gray")
        self.lbl_det.pack(expand = True, fill = tk.BOTH, padx = 5, pady = 5)

        frame_seg = tk.Frame(display_frame, relief="sunken")
        frame_seg.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        tk.Label(frame_seg, text = "Segmentation", font = ("Arial", 14, "bold")).pack(side = tk.TOP)
        self.lbl_seg = tk.Label(frame_seg, text = "[ Image Area ]", font = ("Arial", 14), bg = "gray")
        self.lbl_seg.pack(expand = True, fill = tk.BOTH, padx = 5, pady = 5)
    
    def load_image(self) : 
        file_path = filedialog.askopenfilename(
            filetypes = [("Image files", "*.png *.jpg *.jpeg")]
        )
        if file_path : 
            self.current_img_path = file_path
            img_pil = Image.open(file_path)
            self.display_on_label(img_pil, self.lbl_org)
            self.lbl_det.config(image = "", text = "[ Image Area ]")
            self.lbl_seg.config(image = "", text = "[ Image Area ]")

    def display_on_label(self, img_pil, label_widget) : 
        self.root.update()

        target_w = label_widget.winfo_width()
        target_h = label_widget.winfo_height()
        if target_w < 50 : 
            target_w, target_h = 380, 500
        
        org_w, org_h = img_pil.size
        ratio = min(target_w / org_w, target_h / org_h)
        new_w = int(org_w * ratio)
        new_h = int(org_h * ratio)
        
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_resized)

        label_widget.config(image = tk_img, text = "")
        label_widget.image = tk_img
    
    def inference(self) : 
        if self.current_img_path : 
            image_org = cv2.imread(self.current_img_path)
            h_org, w_org = image_org.shape[:2]
            image_rgb = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)

            # Detection
            det_results = self.det_model.predict(self.current_img_path, imgsz = 640, conf = 0.25)[0]
            det_plotted = det_results.plot(labels = True, boxes = True)
            det_rgb = cv2.cvtColor(det_plotted, cv2.COLOR_BGR2RGB)
            det_pil = Image.fromarray(det_rgb)
            self.display_on_label(det_pil, self.lbl_det)

            # Segmentation
            augmented = self.seg_transform(image = image_rgb)
            input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad() : 
                output = self.seg_model(input_tensor)
                pred_prob = torch.sigmoid(output).cpu().squeeze().numpy()
            
            size = 640
            scale = size / max(h_org, w_org)
            new_h, new_w = int(h_org * scale), int(w_org * scale)

            top = (size - new_h) // 2
            left = (size - new_w) // 2
            
            pred_mask_unpadded = pred_prob[top:top+new_h, left:left+new_w]
            pred_mask_final = cv2.resize(pred_mask_unpadded, (w_org, h_org), interpolation = cv2.INTER_LINEAR)

            mask_bin = (pred_mask_final > 0.5).astype(np.uint8)
            overlay = image_rgb.copy()
            overlay[mask_bin == 1] = [0, 0, 255]

            blended_rgb = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)

            seg_pil = Image.fromarray(blended_rgb)
            self.display_on_label(seg_pil, self.lbl_seg)

if __name__ == "__main__" : 
    root = tk.Tk()
    app = KSDD2App(root)
    root.mainloop()