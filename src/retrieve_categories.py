from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import faiss
import torch
import clip
import json


object_retrieval_template = "I think there might be a _."
place_retrieval_template = "I think this image was taken at a _."
replace_token = "_"

def encode_text(text_arr, bs, clip_model, clip_tokenizer, device):
    assert (lambda x: ((x & (x-1)) == 0) and x != 0)(bs), f'batch size {bs} is not a power of 2.'

    encoded_text = []
    for idx in tqdm(range(0, len(text_arr), bs)):
        with torch.no_grad():
            if clip_tokenizer == None:
                text_ids = clip.tokenize(text_arr[idx:idx+bs]).to(device)
            else:
                text_ids = clip_tokenizer(text_arr[idx:idx+bs]).to(device)
            encoded_text.append(clip_model.encode_text(text_ids).cpu().numpy())

    return np.concatenate(encoded_text)


#####################################################################################################################
#                                            PROCESS CATEGORY DATASETS                                              #
#####################################################################################################################
def process_category_datasets(model, tokenizer, device):
    global place_retrieval_template, object_retrieval_template, replace_token

    # Places ----------------------------------------------------------------------------------
    data_path = 'data/places/categories_places365.txt'

    bs = 256
    places = np.loadtxt(data_path, dtype=str)
    
    place_texts = []
    for place in places[:, 0]:
        place = place.split('/')[2:]
        if len(place) > 1:
            place = place[1] + ' ' + place[0]
        else:
            place = place[0]
        place = place.replace('_', ' ')
        place_texts.append(place)
    
    encoded_places = encode_text([place_retrieval_template.replace(replace_token, p) for p in place_texts], bs, model, tokenizer, device)
    #------------------------------------------------------------------------------------------


    # Objects ---------------------------------------------------------------------------------
    data_path = 'data/objects/dictionary_and_semantic_hierarchy.txt'

    with open(data_path) as fid:
        object_categories = fid.readlines()
    object_texts = []
    for object_text in object_categories[1:]:
        object_text = object_text.strip()
        object_text = object_text.split('\t')[3]
        # Maintain one variant
        variants = object_text.split(',')
        if len(variants) > 0:
            first_variant = variants[0].strip()
            object_texts.append(first_variant)
    object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
    encoded_objects = encode_text([object_retrieval_template.replace(replace_token, o) for o in object_texts], bs, model, tokenizer, device)
    #------------------------------------------------------------------------------------------

    return place_texts, encoded_places, object_texts, encoded_objects





#####################################################################################################################
#                                            RETRIEVE NEAREST NEIGHBOURS                                            #
#####################################################################################################################
def get_category_nns(categories, images, k=7):
    xq = images.astype(np.float32)
    xb = categories.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 
    return I, D

"""
Obtain text from nearest neighbours
"""
def nns_to_categories(place_texts, place_nns, place_similarities, object_texts, object_nns, object_similarities, xq_image_ids):
    retrieved_categories = {}

    for place_nns_list, object_nns_list, image_id in zip(place_nns, object_nns, xq_image_ids):
        retrieved_categories[image_id] = {"places": [], "objects": []}

        for place_idx in place_nns_list:
            retrieved_categories[image_id]["places"].append(place_texts[place_idx])
        for object_idx in object_nns_list:
            retrieved_categories[image_id]["objects"].append(object_texts[object_idx])
        
        image_index = xq_image_ids.index(image_id)
    
        retrieved_categories[image_id]["places"].append( [str(sim) for sim in place_similarities[image_index]] )
        retrieved_categories[image_id]["objects"].append( [str(sim) for sim in object_similarities[image_index]] )
    
    return retrieved_categories





#####################################################################################################################
#                                                     UTILS                                                         #
#####################################################################################################################
"""
Get maximum size of both places and objects categories encoddings
"""
def get_cat_max_size(place_texts, object_texts):

    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    bs = 512

    place_encodings = []
    for idx in range(0, len(place_texts), bs):
        place_encodings += tokenizer.batch_encode_plus(place_texts[idx:idx+bs], return_tensors='np')['input_ids'].tolist()
    
    object_encodings = []
    for idx in range(0, len(object_texts), bs):
        object_encodings += tokenizer.batch_encode_plus(object_texts[idx:idx+bs], return_tensors='np')['input_ids'].tolist()

   
    return max(map(len, place_encodings)), max(map(len, object_encodings))


"""
Retrieve categories from given datasets
---------------------------------------

Parameters:
- clip_model: Clip model used to encode the category strings, also the same used to encode the images
- clip_version: Version of used clip model
- xq_image_ids: Ids of the images
- encoded_images: Vector containing the encodded images
- pk: Number of place categories to retrieve
- ok: Number of object categories to retrieve
"""
def retrieve_categories(clip_model, clip_tokenizer, xq_image_ids, encoded_images, categories_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get the places and objects strings as well as their encoddings
    place_texts, encoded_places, object_texts, encoded_objects = process_category_datasets(clip_model, clip_tokenizer, device)


    #####################################################################################################################
    #                                                  Retrieval                                                        #
    #####################################################################################################################
    print('Retrieving places')
    place_nns, place_similarities = get_category_nns(encoded_places, encoded_images)
    print('Retrieving objects')
    object_nns, object_similarities = get_category_nns(encoded_objects, encoded_images)

    # Get categories from nearest neighbours
    retrieved_categories = nns_to_categories(place_texts, place_nns, place_similarities, object_texts, object_nns, object_similarities, xq_image_ids)


    #####################################################################################################################
    #                                                    UTILS                                                          #
    #####################################################################################################################
    # Get maximum sizes for object and places categories encoddings
    max_place_encodding_size, max_obj_encodding_size = get_cat_max_size(place_texts, object_texts)

    print("Maximum place encodding size: ", max_place_encodding_size)
    print("Maximum object encodding size: ", max_obj_encodding_size)
    

    #####################################################################################################################
    #                                                 SAVED FILES                                                       #
    #####################################################################################################################
    # Save file with categories
    json.dump(retrieved_categories, open(categories_path, 'w'), indent=4)
