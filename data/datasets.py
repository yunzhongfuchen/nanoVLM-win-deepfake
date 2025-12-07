import torch
from PIL import Image
from torch.utils.data import Dataset

import models.config as cfg


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle image (it's a list)
        image_data = item['images']
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Process text (also a list)
        text_data = item['texts']
        if isinstance(text_data, list) and len(text_data) > 0:
            text = text_data[0]
        else:
            text = text_data

        question = text['user']
        # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        answer = text['assistant'] + self.tokenizer.eos_token

        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
            
        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        
        question = item['question']
        answer = item['answer'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        
        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer:"
        
        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


import os
from pathlib import Path


class AuthFolderDataset(Dataset):
    """
    从一个大目录下的三个子文件夹(real, tampered, full_synthetic)读取图片，
    并把类别转成文本答案，例如:
        real          -> "The image is real"
        tampered      -> "The image is tampered"
        full_synthetic-> "The image is full synthetic"

    输出格式与 VQADataset 一致：
        {
            "image": processed_image,   # tensor [3, H, W]
            "text_data": formatted_text,# 问句 Prompt
            "answer": answer_text       # 文本答案 + eos_token
        }
    这样就能直接接 VQACollator 使用。
    """

    def __init__(self, root_dir, tokenizer, image_processor):
        """
        :param root_dir: 根目录路径，下面有 real / tampered / full_synthetic 三个子目录
        :param tokenizer: 训练用 tokenizer，用来拿 eos_token
        :param image_processor: 官方 get_image_processor 得到的图像预处理
        """
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # 文件夹名 -> 文本答案
        self.class_to_answer = {
            "real": "The image is real",
            "tampered": "The image is tampered",
            "full_synthetic": "The image is full synthetic",
        }
        self.class_names = list(self.class_to_answer.keys())

        # 收集样本 (path, class_name)
        self.samples = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                print(f"[WARN] 子目录不存在: {cls_dir}")
                continue

            for fname in os.listdir(cls_dir):
                path = cls_dir / fname
                if path.suffix.lower() not in exts:
                    continue
                self.samples.append((path, cls))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"在 {self.root_dir} 下没有找到任何图片（real/tampered/full_synthetic）"
            )

        print(f"[INFO] AuthFolderDataset: 共 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name = self.samples[idx]

        # --- 读图 ---
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] 打开图片失败: {img_path}, error: {e}")
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size
            )
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)

        # --- 构造 QA 文本 ---
        # 问题你可以随便改，这里给一个示例
        question = "Classify the authenticity of this image as real, tampered, or full synthetic."
        # 根据文件夹生成答案
        answer_text = self.class_to_answer[cls_name]

        # 加 eos，让模型学会何时停
        answer = answer_text + self.tokenizer.eos_token * 3

        # 跟官方 VQADataset 的格式保持一致
        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer,
        }
