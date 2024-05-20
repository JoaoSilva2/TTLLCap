import numpy as np
import faiss
import json
import os

"""
Class to retrieve captions in the datastore with Human Labeled + Web data. Captions from here are not tied to any image id so there is no need to preprocess the dataset.
Images are still from the COCO dataset.
"""

class FaissSearch():
    def __init__(self, datastore, num_neighbors, num_target):
        self.datastore = datastore
        self.num_neighbors = num_neighbors
        self.num_target = num_target

    def save_files(self, index, nns, similarities):
        # If nearest neighbours and distances have never been retrieved before save them for future use
        if not os.path.exists(self.nns_path) or not os.path.exists(self.similarities_path):
            print("Writing COCO nearest neighbours file")
            json.dump([[str(s) for s in nn] for nn in nns], open(self.nns_path, 'w'))

            print("Writing COCO similarities file")
            json.dump([[str(f) for f in sim] for sim in similarities], open(self.similarities_path, 'w'))
    

    def nn_search(self, target_embeddings, batch_embeddings):
        xq = target_embeddings.astype(np.float32)
        xb = batch_embeddings.astype(np.float32)
        faiss.normalize_L2(xb)

        quantizer = faiss.IndexFlatIP(xb.shape[1])
        index = faiss.IndexIVFFlat(quantizer, xb.shape[1], 256, faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        faiss.normalize_L2(xq)

        index.nprobe = 32
        D, I = index.search(xq, self.num_neighbors)

        return index, I, D

    def filter_matching_pairs(self, nns, similarities, target_ids, batch_ids):
        """ We filter out nearest neighbors which are actual captions for the query image, keeping x neighbors per image."""
        filtered_nns, filtered_similarities = [], []
        for nns_list, sims_list, image_id in zip(nns, similarities, target_ids):
            good_nns, good_sims = [], []

            for nn, sim in zip(nns_list, sims_list):
                if batch_ids[nn] == image_id:
                    continue
                good_nns.append(nn)
                good_sims.append(sim)
                if len(good_nns) == self.num_target:
                    break

            assert len(good_nns) == self.num_target
            filtered_nns.append(good_nns)
            filtered_similarities.append(good_sims)
        
        print(self.target, all([len(nns) == self.target for nns in filtered_nns]))
        return filtered_nns, filtered_similarities


    def retrieve_captions(self, target_embeddings, target_ids, batch_embeddings, batch_ids):
        print("Retrieving captions using FAISS. This process can take several hours.")
        
        # Retrieve nearest neighbours
        index, neighbor_indices, similarities = self.nn_search(target_embeddings, batch_embeddings)

        # Filter nearest neighbours
        filtered_nns, filtered_similarities = self.filter_matching_pairs(neighbor_indices, similarities, target_ids, batch_ids)

        return filtered_nns, filtered_similarities
