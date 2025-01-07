import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageSelector:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Selector")
        
        self.result_path = "Subset_2/Gen_results"
        self.mask_path = "Subset_2/masks_tag"
        self.save_path = "Subset_2/Gen_results_filtered"
        
        self.mask_list = []
        self.image_list = []
        self.current_image = 0
        self.current_mask = 368

        self.image_labels = []
        for _ in range(3):  # 显示3张图片
            label = tk.Label(master)
            label.pack(side=tk.LEFT)
            self.image_labels.append(label)
        
        self.keep_button = tk.Button(master, text="保留", command=self.keep_image)
        self.keep_button.pack(side=tk.LEFT)

        self.delete_button = tk.Button(master, text="删除", command=self.delete_image)
        self.delete_button.pack(side=tk.LEFT)

        self.keep_mask_button = tk.Button(master, text="保留mask", command=self.keep_mask)
        self.keep_mask_button.pack(side=tk.RIGHT)

        self.next_mask_button = tk.Button(master, text="删除mask", command=self.next_mask)
        self.next_mask_button.pack(side=tk.RIGHT)

        self.get_masks()


    def get_masks(self):
        for id in os.listdir(self.mask_path):
            for mask in os.listdir(os.path.join(self.mask_path, id)):
                if mask == 'anotated_img.png':
                    continue
                else:
                    self.mask_list.append([os.path.join(self.mask_path, id, mask), id, int(mask.rstrip('.png').lstrip('mask_'))-1])
        self.show_mask()

    def show_mask(self):
        print(self.current_mask)
        if self.current_mask < len(self.mask_list):
            [img_path, image_id, mask_id] = self.mask_list[self.current_mask]
            ano_img = Image.open(os.path.join(os.path.dirname(img_path), 'anotated_img.png'))
            img = Image.open(img_path)
            img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.image_labels[1].config(image=img_tk)
            self.image_labels[1].image = img_tk

            ano_img.thumbnail((400, 400))
            ano_img_tk = ImageTk.PhotoImage(ano_img)
            self.image_labels[0].config(image=ano_img_tk)
            self.image_labels[0].image = ano_img_tk
        else:
            self.img_label.config(text="没有更多mask图片")

    def show_image(self):
        if self.current_image < len(self.image_list):
            img_path = self.image_list[self.current_image]
            img = Image.open(img_path)
            img.thumbnail((400, 400))  # Adjust thumbnail size as needed
            img_tk = ImageTk.PhotoImage(img)
            self.image_labels[2].config(image=img_tk)
            self.image_labels[2].image = img_tk
        else:
            self.current_mask += 1
            self.show_mask()

    def keep_image(self):
        current_img = self.image_list[self.current_image]
        save_path = current_img.replace('Gen_results', 'Gen_results_filtered')

        [_, image_id, _] = self.mask_list[self.current_mask]
        if not os.path.exists(os.path.join(self.save_path, str(image_id))):
            os.mkdir(os.path.join(self.save_path, str(image_id)))
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        shutil.copy(current_img, save_path)
        self.next_image()

    def delete_image(self):
        self.next_image()

    def next_image(self):
        self.current_image += 1
        self.show_image()

    def keep_mask(self):
        [img_path, image_id, mask_id] = self.mask_list[self.current_mask]
        img_path = os.path.dirname(img_path.replace(self.mask_path, self.result_path))
        img_path = os.path.join(img_path, str(mask_id))
        if os.path.exists(img_path):
            image_list = os.listdir(img_path)
            self.image_list = [os.path.join(img_path, i) for i in image_list]
            self.current_image = 0
            self.show_image()
        else:
            self.current_mask += 1
            self.show_mask()

    def next_mask(self):
        self.current_mask += 1
        self.show_mask()
        


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelector(root)
    root.mainloop()
