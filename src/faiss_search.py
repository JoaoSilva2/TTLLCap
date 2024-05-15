import faiss
import json
import os

"""
Class to retrieve captions in the datastore with Human Labeled + Web data. Captions from here are not tied to any image id so there is no need to preprocess the dataset.
Images are still from the COCO dataset.
"""

class FaissSearch():
    def __init__(self, datastore):
        self.datastore = datastore

    def save_files(self, index, nns, similarities):
        # If nearest neighbours and distances have never been retrieved before save them for future use
        if not os.path.exists(self.nns_path) or not os.path.exists(self.similarities_path):
            print("Writing COCO nearest neighbours file")
            json.dump([[str(s) for s in nn] for nn in nns], open(self.nns_path, 'w'))

            print("Writing COCO similarities file")
            json.dump([[str(f) for f in sim] for sim in similarities], open(self.similarities_path, 'w'))
    
    def log(self):
        print("----------------------------------------------------------------------------------------------------------------------------\n" + \
            f"                                    NOW RETRIEVING CAPTIONS FOR THE COCO DATASET (Human + Web)                               \n" + \
            "----------------------------------------------------------------------------------------------------------------------------\n"   + \
            f"-> {self.k} captions per image                                                                                              \n"  + \
            f"-> {self.n} neighbours will be retrieved                                                                                    \n"  + \
            f"-> {self.target} neighbours will be maintained                                                                              \n"  + \
            "-----------------------------------------------------------------------------------------------------------------------------\n")
        
    def retrieve_captions(self, target):
        self.log()
        
        #self.encoded_images = self.encode_all_captions()

        # Encode captions and images
        self.encode_captions()
        self.encode_images()

        # Retrieve nearest neighbours
        index, I, D = self.get_nns()
        self.save_files(index, I, D)

        # Filter nearest neighbours
        filtered_nns, filtered_similarities = self.filter_nns(I, D)

        return index, filtered_nns, filtered_similarities
