from numpy import np
from tqdm import tqdm
import torch
import clip
import json

class RetrievalMethod():
    def __init__(self, clip_model, feature_extractor, clip_tokenizer, device) -> None:
        # CLIP tools
        self.clip_model = clip_model
        self.feature_extractor = feature_extractor
        self.clip_tokenizer = clip_tokenizer
        self.device = device
    
    def coco_text_2_text(self, coco_path):
        bs = 256
        coco_data = json.load(open(coco_path))['images']

        captions = []
        for item in coco_data:
            if item['split'] == 'restval':
                item['split'] = 'train'
            if item['split'] == 'train':
                for sentence in item['sentences']:
                    captions.append({'image_id': item['cocoid'],  'caption': ' '.join(sentence['tokens']), 'sentid': sentence['sentid']})

        query_caption_img_ids = [img['image_id'] for img in captions]

        encoded_captions = []
        for idx in tqdm( range(0, len(query_caption_img_ids), bs)):
            captions = [caption['caption'] for caption in captions[idx:idx+bs]]
            with torch.no_grad():
                if self.clip_tokenizer == None:
                    input_ids = clip.tokenize(captions).to(self.device)
                else:
                    input_ids = self.clip_tokenizer(captions).to(self.device)
                encoded_captions.append( self.clip_model.encode_text(input_ids).cpu().numpy() )

        return np.concatenate(encoded_captions)
