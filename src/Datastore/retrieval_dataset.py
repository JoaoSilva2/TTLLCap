from transformers import AutoTokenizer

from tqdm import tqdm
import json

class RetrievalDataset():

    def __init__(self, coco_path, web_path) -> None:
        print("Loading data for datastore")
        self.images, self.captions = [], []

        coco_data = json.load(open(coco_path))['images']

        web_data = []
        with open(web_path, 'r') as web_data_file:
            web_data.append(json.load(web_data_file))

        # Loading coco data into datastore
        for item in coco_data:
            if item['split'] == 'restval':
                item['split'] = 'train'
            if item['split'] == 'train':
                for sentence in item['sentences']:
                    self.captions.append({'image_id': item['cocoid'],  'caption': ' '.join(sentence['tokens']), 'sentid': sentence['sentid']})
            self.images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})


        # Used for img-text retrieval
        self.query_image_ids = [img['image_id'] for img in self.images]
        # Used for text-text retrieval
        self.query_caption_img_ids = [img['image_id'] for img in self.captions]
        self.query_caption_ids = [cap['sentid'] for cap in self.captions]

        # Loading web data
        id = -1
        for data in web_data:
            for instance in data:
                self.captions.append({'image_id': id, 'caption': instance['caption']})
                id -= 1

        # Ids for captions in datastore
        self.batch_image_ids = []
    

    def preprocess(self):
        bs = 512

        print("Removing captions that have a gpt2 encoder length higher than 25")
        decoder_name = 'gpt2'
        tokenizer = AutoTokenizer.from_pretrained(decoder_name)

        captions_image_ids  = self.query_caption_img_ids
        captions = [cap['caption'] for cap in self.captions]

        encodings = []
        for idx in tqdm(range(0, len(self.captions), bs)):
            encodings += tokenizer.batch_encode_plus(captions[idx:idx+bs], return_tensors='np')['input_ids'].tolist()
        
        assert len(captions_image_ids) == len(captions) and len(captions) == len(encodings)

        # Captions with length lower or equal than 25
        filtered_image_ids, filtered_captions = [], []

        for image_id, cap, encoding in zip(captions_image_ids, captions, encodings):
            if len(encoding) <= 25:
                filtered_image_ids.append(image_id)
                filtered_captions.append(cap)
        
        print("Then: ",  len(captions_image_ids), " Now: ", len(filtered_image_ids))
        self.captions = filtered_captions

        # List of image ids of captions that will be present in the datastore
        self.batch_image_ids = filtered_image_ids