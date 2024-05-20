from PIL import ImageFile
from PIL import Image
import open_clip
import argparse
import torch
import clip
import json
import os

from Datastore.retrieval_dataset import RetrievalDataset
from Datastore.retrieval_method import RetrievalMethod
from Reranking.rerank import CaptionReranking
from faiss_search import FaissSearch



ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_CLIP_VERSIONS = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
            

def float_or_list(arg):
    try:
        # Try to convert the input to a single float
        return float(arg)
    except ValueError:
        # If conversion to float fails, try to convert it to a list of floats
        try:
            # Split the input string by commas and convert each part to float
            return [float(x) for x in arg.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Argument must be a float or a comma-separated list of floats")

def load_clip_model(clip_version, device):
    clip_tokenizer = None

    # If CLIP is from openai load using clip library
    if clip_version in DEFAULT_CLIP_VERSIONS:
        clip_model, feature_extractor = clip.load(clip_version, device=device)
    # Otherwise load from open_ai library
    else:
        clip_model, _, feature_extractor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
        clip_tokenizer = open_clip.get_tokenizer(clip_version)

    return clip_model, feature_extractor, clip_tokenizer


def nns_to_caption(nns, img_cap_similarities, captions, xq_image_ids):
    retrieved_captions = {}
    retrieved_similarities = {}
    for nns_list, image_id, img_cap_similarity_list in zip(nns, xq_image_ids, img_cap_similarities):
        retrieved_captions[image_id] = []
        retrieved_similarities[image_id] = []
        for nn, img_cap_similarity in zip(nns_list, img_cap_similarity_list):
            retrieved_captions[image_id].append(captions[nn])
            retrieved_similarities[image_id].append(str(img_cap_similarity))
            
        assert(len(retrieved_captions[image_id]) >= 4)
    return retrieved_captions


def main(args): 
    if "/" in args.encoder_name:
        clip_version = args.encoder_name.replace("/", "-")
    else:
        clip_version = args.encoder_name

    print("-------------------------------------------------------------------------------------------------------------------------\n"
          f"-> Running caption retrieval with clip {clip_version} with {args.k} captions retrieved\n" + \
          f"-> {args.n} neighbours will be retrieved and after filtering {args.target} neighbours will remain\n" + \
          f"-> Reranking is set to {args.diversity} with lambda {args.lambda_param}\n" + \
          f"-> The dataset used is {args.dataset}."
          "-------------------------------------------------------------------------------------------------------------------------")
    
    # Load CLIP model and auxiliaries
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP model...")
    clip_model, feature_extractor, clip_tokenizer = load_clip_model(clip_version, device)

    # Setup Datastore
    datastore = RetrievalDataset(args.coco_path, args.web_path)
    datastore.preprocess()

    batch_embeddings, batch_ids = datastore.encode_datastore(clip_model, clip_tokenizer)

    # Setup Retrieval System
    # TODO Allow retrieval to be done to NoCaps

    retrieval_method = RetrievalMethod(clip_model, feature_extractor, clip_tokenizer, device)
    target_embeddings, target_ids = None, None

    if args.retrieval_method == "text2text":

        if args.dataset == "coco":
            target_embeddings, target_ids = retrieval_method.coco_text_2_text(args.coco_path)
        if args.dataset == "nocaps":
            pass

    elif args.retrieval_method == "img2text":

        if args.dataset == "coco":
            target_embeddings, target_ids = retrieval_method.coco_image_2_text(args.coco_path, args.images_dir)
        if args.dataset == "nocaps":
            pass

    else:
        raise ValueError("Invalid retrieval method")

    search_engine = FaissSearch(datastore, args.n, args.target)
    retrieved_neighbors, similarities = search_engine.retrieve_captions(target_embeddings, target_ids, batch_embeddings, batch_ids)

    # Re-rank
    if args.diversity:
        for lambda_value in args.lambda_params:
            rerank_method = CaptionReranking(retrieved_neighbors, similarities, batch_embeddings, lambda_value)
            retrieved_neighbors, similarities = rerank_method.rerank_all()
        

    # Save captions
    retrieved_captions = nns_to_caption(retrieved_neighbors, similarities, datastore.captions, target_ids)

    json.dump(retrieved_captions, open(args.output_path, 'w'))




   



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caption retrieval')

    parser.add_argument("--images_dir", type=str, default="data/images/", help="Directory where input image features are stored")
    parser.add_argument("--coco_path", type=str, default="", help="Path to COCO data")
    parser.add_argument("--web_path", type=str, default="", help="Path to Web data")

    parser.add_argument("--encoder_name", type=str, default="ViT-L-14", help="Encoder used to retrieve captions and categories")

    parser.add_argument("--retrieval_method", type=str, default="text2text", help="Use text2text retrieval or img2text retrieval")

    parser.add_argument("--n", type=int, default=100, help="Number of neighbours retrieved")
    parser.add_argument("--target", type=int, default=50, help="Total number of captions remaining after filtering")

    parser.add_argument("--diversity", action="store_true", default=False, help="Use maximal marginal relevance to rerank retreived captions")
    parser.add_argument("--lambda_params", type=float_or_list, help="Lambda parameter or parameters used in maximal marginal relevance (pos - diverse, neg - cohesive)")

    parser.add_argument("--dataset", type=str, default="coco", help="Use nocaps data instead of web data")

    parser.add_argument("--output_path", type=str, default="", help="Json file where captions will be saved")

    args = parser.parse_args()

    main(args)