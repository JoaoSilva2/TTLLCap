from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import numpy as np
import random

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "

PLACE_LENGTH = 2
OBJECT_LENGTH = 4

# FOR PROMPT DEBUGGING
show_prompt = True
# --------------------

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, retrieved_places=None, retrieved_objects=None, k=None, pk=None, ok=None, is_test=False, max_length=None, prompt=False, shuffle=False, reverse=False, self_att=False):
    global show_prompt

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True

    if retrieved_caps is not None:
        retrieval_prompt = retrieved_caps[:k]

        if shuffle:
            random.shuffle(retrieval_prompt)

        if reverse:
            infix = '\n\n'.join(retrieval_prompt[::-1]) + '.'
        else:
            infix = '\n\n'.join(retrieval_prompt) + '.'

        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX
        
    if retrieved_places is not None and retrieved_objects is not None:
        for i in range(pk):
            prefix = prefix.replace(f"_place{i+1}_", retrieved_places[i])
        for i in range(ok):
            prefix = prefix.replace(f"_object{i+1}_", retrieved_objects[i])
    
    if prompt:
        return prefix
    # For debugging ------------------
    if show_prompt:
        print(prefix)
        show_prompt = not show_prompt
    # --------------------------------

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    
    
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    if not self_att:
        label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id]
    else:
        # If model is not SmallCap shift is done by the model
        label_ids = [-100] * (len_prefix) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    

    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred

def get_cap_indexes(input_ids, caps, decoder_name):
    indexes = []

    for cap in caps:

        found = False
        max_sim = 0
        backup = (0, 1)

        # Remove EOS token from beggining
        if "opt" in decoder_name:
            cap = cap[1:]
        for i in range(len(input_ids)):
            subset = input_ids[i:len(cap)+i]

            sim = len(set(subset) & set(cap))
            if sim > max_sim:
                sim = max_sim
                backup = (i, i+len(cap))

            if subset == cap:
                indexes.append((i, i+len(cap)))
                found = True
                break
        
        if found == False:
            indexes.append(backup)

    return indexes

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, sm=False, template_path=None, k=None, pk=None, ok=None, max_caption_length=25, shuffle=False, reverse=False, self_att=False):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.max_target_length = max_caption_length
        self.shuffle = shuffle
        self.reverse = reverse
        self.self_att = self_att
        self.count = 0


        if rag:
            self.template = open(template_path).read().strip() + ' '
            if sm:
                self.max_target_length = (max_caption_length  # target caption
                                        + max_caption_length * k # retrieved captions
                                        + PLACE_LENGTH * pk # retrieved places
                                        + OBJECT_LENGTH * ok # retrieved objects
                                        + len(tokenizer.encode(self.template)) # template
                                        + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                        )
            else:
                self.max_target_length = (max_caption_length  # target caption
                                        + max_caption_length * k # retrieved captions
                                        + len(tokenizer.encode(self.template)) # template
                                        + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                        )

            assert k is not None
            assert pk is not None
            assert ok is not None
            self.k = k
            self.pk = pk
            self.ok = ok
        self.rag = rag
        self.sm = sm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.count == 0:
            print("\n\n\n\n" , self.df['cocoid'][idx], "\n\n\n\n")
            self.count = 1

        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            places = self.df['places'][idx]
            objects = self.df['objects'][idx]

            if self.sm:
                decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, retrieved_places=places, retrieved_objects=objects, 
                                                     k=self.k, pk=self.pk, ok=self.ok, max_length=self.max_target_length, shuffle=self.shuffle,
                                                     reverse=self.reverse, self_att=self.self_att)
            else:
                decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length, shuffle=self.shuffle,
                                                     reverse=self.reverse, self_att=self.self_att)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)

        # load precomputed features
        encoder_outputs = self.features[self.df['sentid'][idx]][()]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training(annot_path, caps_path=None, cats_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    if cats_path is not None:
        retrieved_cats = json.load(open(cats_path))
    data = {'train': [], 'val': []}
    
    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        # Captions --------------------------------------
        caps = None
        places = None
        objects = None
        #------------------------------------------------
        samples = []
        for sentence in item['sentences']:
            caps = []
            places = []
            objects = []
            if item["split"] == "train" or item['split'] == 'restval':
                caps = retrieved_caps[str(sentence['sentid'])]

                if cats_path is not None:
                    cats = retrieved_cats[str(sentence['sentid'])]
                    places = cats['places'][:-1]
                    objects = cats['objects'][:-1]

            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens']), 'places': places, 'objects': objects, 'sentid': str(sentence["sentid"])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 

def load_data_for_inference(annot_path, caps_path=None, cats_path=None, split='val'):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    if cats_path is not None:
        retrieved_cats = json.load(open(cats_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        if item["split"] == split:
            file_name = item['filename'].split('_')[-1]
            if caps_path is not None:
                caps = retrieved_caps[str(item['cocoid'])]
            else:
                caps = None

            # Categories ------------------------------------
            if cats_path is not None:
                cats = retrieved_cats[str(item['cocoid'])]
                places = cats['places'][:-1]
                objects = cats['objects'][:-1]
            else:
                places = None
                objects = None
            #------------------------------------------------

            image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid']), 'places': places, 'objects': objects}
            if item['split'] == 'test':
                data['test'].append(image)
            elif item['split'] == 'val':
                data['val'].append(image)

    return data      

