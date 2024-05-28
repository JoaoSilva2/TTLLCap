import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
import clip
import open_clip

from src.utils import load_data_for_inference, prep_strings, postprocess_preds, get_cap_indexes
ImageFile.LOAD_TRUNCATED_IMAGES = True


PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df, checkpoint=None, clip_model=None):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []

    for idx in tqdm(range(0, len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        decoder_input_ids = prep_strings('', tokenizer, is_test=True)
                
        # load image 
        image = Image.open(args.images_dir + file_name).convert("RGB")
        pixel_values = feature_extractor(image).unsqueeze(0).to(args.device)
        encoder_last_hidden_state = clip_model.encode_image(pixel_values).unsqueeze(0).to(args.device)
        encoder_last_hidden_state = torch.nn.functional.normalize(encoder_last_hidden_state, dim=2)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
        #pixel_values = feature_extractor(images, return_tensors="pt").pixel_values

        with torch.no_grad():
            pred = model.generate(encoder_outputs=encoder_outputs, 
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        pred = tokenizer.decode(pred.sequences[0])
 
        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred})

    return out

def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df, checkpoint=None, clip_model=None):
    """RAG models can only be evaluated with a batch of length 1."""

    template_path = "src/template_base_cat.txt" if args.socratic_models else args.template_path
    template = open(template_path).read().strip() + ' '

    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')

    out = []
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        caps = eval_df['caps'][idx]

        if not args.socratic_models:
            decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                                    k=int(args.k), is_test=True)
        else:
            places = eval_df['places'][idx]
            objects = eval_df['objects'][idx]
            decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                                    k=int(args.k), pk=int(args.pk), ok=int(args.ok), is_test=True,
                                                    retrieved_places=places, retrieved_objects=objects)
        # load features
        if args.features_path is not None:
            encoder_last_hidden_state = torch.FloatTensor([features[image_id][()]])

            encoder_last_hidden_state = torch.nn.functional.normalize(encoder_last_hidden_state, dim=2)

            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        else:
            image = Image.open(args.images_dir + file_name).convert("RGB")
            #pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

            pixel_values = feature_extractor(image).unsqueeze(0).to(args.device)
            encoder_last_hidden_state = clip_model.encode_image(pixel_values).unsqueeze(0).to(args.device)
            encoder_last_hidden_state = torch.nn.functional.normalize(encoder_last_hidden_state, dim=2)

            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))

            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)

        seqs = []
        for seq in pred.sequences:
            gen_seq = tokenizer.decode(seq)
            gen_seq = postprocess_preds(gen_seq, tokenizer)
            seqs.append(gen_seq)

        if args.candidate_captions == 1:
            out.append({"image_id": int(image_id), "caption": seqs[0]})
        else:
            out.append({"image_id": int(image_id), "caption": seqs})

    return out

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn, clip_model):
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df, checkpoint=checkpoint_path, clip_model=clip_model)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)



def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.Configs.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.Configs.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.Configs.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src.Configs.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
    
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

def main(args):

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.infer_test or args.disable_rag or args.dataset == "nocaps":
        args.features_path = None
    
    if args.features_path is not None:
        feature_extractor = None
        clip_encoder = None
    else:
        clip_encoder, _, feature_extractor = open_clip.create_model_and_transforms("ViT-L-14", pretrained='laion2B-s32B-b82K', device=args.device)
        #feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    if args.disable_rag:
        args.k=0
        infer_fn = evaluate_norag_model
    else:
        infer_fn = evaluate_rag_model

    if args.infer_test:
        split = 'test'
    else:
        split = 'val'

    if not args.socratic_models:
        args.categories_path = None

    data = load_data_for_inference(args.annotations_path, args.captions_path, args.categories_path, split=split)

    eval_df = pd.DataFrame(data[split])
    args.outfile_name = '{}_{}_preds.json'.format(split, args.dataset)

    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id, 'output_attentions': True,
                              'output_scores': True, 'return_dict_in_generate': True, 'num_return_sequences': args.candidate_captions}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn, clip_encoder)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn, clip_encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--images_dir", type=str, default="data/images/", help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default="features/val.hdf5", help="H5 file with cached input image features")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
        
    parser.add_argument("--model_path", type=str, default=None, help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps/ViT-L-14/retrieved_caps.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")

    # Socratic model extension
    parser.add_argument("--socratic_models", action="store_true", default=False, help="Whether to use socratic models prompts")
    parser.add_argument("--pk", type=int, default=0, help="Number of place categories to be added to the prompt")
    parser.add_argument("--ok", type=int, default=0, help="NUmber of object categories to be added to the prompt")
    parser.add_argument("--categories_path", type=str, default="data/retrieved_cats/ViT-L-14/retrieved_categories.json", help="JSON file with retrieved captions")

    parser.add_argument("--self_att", action="store_true", default=False, help="Whether to analyze self attention scores")
    parser.add_argument("--candidate_captions", type=int, default=1, help="Number of candidate captions to generate")
    parser.add_argument("--rerank_encoder_name", type=str, default="ViT-B/32", help="Encoder name as found of HuggingFace or stored locally")

    parser.add_argument("--dataset", type=str, default="coco", help="Use nocaps data instead of coco data")

    args = parser.parse_args()

    main(args)
   
