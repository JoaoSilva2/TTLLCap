from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class CaptionReranking():
    def __init__(self, nns, similarities, encoded_captions, lambda_param) -> None:
        self.nns = nns
        self.similarities = similarities
        self.encoded_captions = encoded_captions
        self.lambda_param = lambda_param

    def rerank_individual(self, idx):
        # Get list of nearest neighbours for current instance
        nn_list = self.nns[idx]

        # Get list of encoded captions for current instance
        encoded_nn = [self.encoded_captions[nn] for nn in nn_list]

        # Compute matrix containing the similarities between nearest neighbours.
        caption_sim_matrix = cosine_similarity(encoded_nn, encoded_nn)

        # Similarities between current instance and captions
        sim = self.similarities[idx]

        # Reranking

        # The most accurate caption will always be used
        currently_retrieved = [nn_list[0]]
        instance_similarities = [sim[0]]

        # Copy the list so we can change the new one
        nn_copy = list(nn_list)[1:]

        while len(currently_retrieved) < self.k:
            mmr_scores = []

            # Loop through remaining nearest neighbours
            for nn in nn_copy:
                # Horizontal index for caption similarity matrix
                y = nn_list.index(nn)
                # Vertical indexes for caption similarity matrix
                x_list = [nn_list.index(retr_nn) for retr_nn in currently_retrieved]

                # Similarity between caption and image
                instance_similarity = sim[y]

                # Get similarity value of the most similar captions between the one being processed now and the ones already retrieved
                max_sim = max([caption_sim_matrix[y][x] for x in x_list])

                # mmr score
                mmr_score = self.mmr(instance_similarity, max_sim)

                mmr_scores.append((nn, mmr_score))

            # Select neighbour with highest mmr score
            nn_to_add, _ = max(mmr_scores, key=lambda l: l[1])

            # Add nearest neighbour with highest mmr
            currently_retrieved.append(nn_to_add)

            idx_to_add = nn_list.index(nn_to_add)
            instance_similarities.append(sim[idx_to_add])

            # Delete selected caption to avoid duplicates
            idx_to_add = nn_copy.index(nn_to_add)
            del nn_copy[idx_to_add]
        
        
        return currently_retrieved, instance_similarities
    
    def rerank_all(self):
        reranked_nns = []
        reranked_similarities = []

        # Loop through all instances
        for idx in tqdm(range( len(self.nns) )):
            
            # Rerank caption retrieved for a single image
            reranked_nn, reranked_sim = self.rerank_individual(idx)
            
            reranked_nns.append(reranked_nn)
            reranked_similarities.append(reranked_sim)

        return reranked_nns, reranked_similarities