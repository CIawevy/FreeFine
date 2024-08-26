import clip
import torch
import spacy
import cv2

class Myclip():
    def __init__(self,type="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(type,device=self.device)
        self.model = model
        self.preprocess = preprocess

    @torch.no_grad()
    def ClipWordMatching(self, img, mask, full_prompt):
        nlp = spacy.load("en_core_web_sm")

        # 使用 spaCy 处理句子
        doc = nlp(full_prompt)
        omit_list = ['photo']

        # 提取名词，排除指定的词
        words = [token.text for token in doc if token.pos_ == "NOUN" and token.text not in omit_list]

        # 通过mask区域的boundary box裁剪图像
        cropped_img = self.crop_image_with_mask(img, mask)
        image = self.clip_process(self.numpy_to_pil(cropped_img)[0]).unsqueeze(0).to(self.device)

        # 对每个词进行编码
        text_tokens = clip.tokenize(words).to(self.device)

        # 计算图像的特征向量
        image_features = self.clip.encode_image(image)

        # 计算每个词的特征向量
        text_features = self.clip.encode_text(text_tokens)

        # 计算相似度
        similarities = torch.cosine_similarity(image_features, text_features)

        # 找到最匹配的词的索引
        best_match_index = torch.argmax(similarities).item()

        # 最匹配的词
        best_match_word = words[best_match_index]

        return best_match_word
    def __call__(self,image_path,mask_path,prompt, *args, **kwargs):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        object_text = self.ClipWordMatching(img=original_image, mask=mask, full_prompt=prompt)
        return object_text
