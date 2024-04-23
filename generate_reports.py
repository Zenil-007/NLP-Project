import argparse
import os
import re
import json
import random
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteria, StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

def clean_reports(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    parser.add_argument('--image_path', default='', type=str, help='path of the input image')
    parser.add_argument('--generation_prompts', type=str, default='prompts/stage2-generation-prompts.txt', help='path of the generation prompts for the first stage')
    parser.add_argument('--refinement_prompts', type=str, default='prompts/stage2-refinement-prompts.txt', help='path of the refinement prompts for the second stage')
    parser.add_argument('--annotations', type=str, default='', help='path of annotation file, to load in the GTs')
    parser.add_argument('--checkpoint', required=True, help='checkpoint path')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_txt_len', default=160, type=int)
    
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

# load LoRA
peft_config = LoraConfig(inference_mode=False, r=cfg.model_cfg.lora_rank, lora_alpha=cfg.model_cfg.lora_alpha, lora_dropout=cfg.model_cfg.lora_dropout)
peft_model = get_peft_model(model.llama_model, peft_config=peft_config)
# loading normal pytroch checkpoint
if args.checkpoint.endswith('.pth'):
    full_state_dict = torch.load(args.checkpoint, map_location='cpu')
# loading ZeRO checkpoint
elif args.checkpoint.endswith('.pt'):
    full_state_dict = torch.load(args.checkpoint, map_location='cpu')['module']
set_peft_model_state_dict(peft_model, full_state_dict)
peft_model = peft_model.to(device)
print('LLaMA checkpoint loaded.')
# load in the linear projection layer
llama_proj_state_dict = {}
for key, value in full_state_dict.items():
    if 'llama_proj' in key:
        llama_proj_state_dict[key[18:]] = value
model.llama_proj.load_state_dict(llama_proj_state_dict)
print('Linear projection layer loaded.')

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

# ========================================
#             Start Testing
# ========================================


# image_paths = []
# for root, dirs, files in os.walk(args.images):
#     for file in files:
#         image_paths.append(os.path.join(root, file))

# load generation prompts from local path
generation_prompts = []
with open(args.generation_prompts, 'r') as f:
    for line in f.readlines():
        generation_prompts.append(line.strip('\n'))

# load refinement prompts from local path
refinement_prompts = []
with open(args.refinement_prompts, 'r') as f:
    for line in f.readlines():
        refinement_prompts.append(line.strip('\n'))

final_record_message = ''
with torch.no_grad():    
    # TODO: Start the first stage
    # random sample one prompt
    prompt = random.choice(generation_prompts)
    prompt = '###Human: ' + prompt + '###Assistant: '
    
    # encode image
    img_list = []
    raw_image = Image.open(args.image_path).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0).to(device)
    image_emb, _ = model.encode_img(image)
    img_list.append(image_emb)
    
    # wrap image with prompt
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [peft_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    
    # prepare other things before generate
    stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    # generate
    outputs = peft_model.base_model.model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=args.max_txt_len,
            stopping_criteria=stopping_criteria,
            num_beams=args.beam_size,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=args.temperature,)
    
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    generated_text = output_text
    
    # TODO: Start the second stage
    coarse_generated_report = output_token
    coarse_report_embeds = peft_model.base_model.model.model.embed_tokens(coarse_generated_report).expand(image_emb.shape[0], -1, -1)
    atts_report = torch.ones(coarse_report_embeds.size()[:-1], dtype=torch.long).to(device)
    prompt = random.choice(refinement_prompts)
    prompt = '###Human: ' + prompt + '###Assistant: '
            
    # encode image
    img_list = []
    raw_image = Image.open(args.image_path).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0).to(device)
    image_emb, _ = model.encode_img(image)
    img_list.append(image_emb)
    
    # the right implementation
    p_before, p_after_all = prompt.split('<ImageHere>')
    p_mid, p_after = p_after_all.split('<ReportHere>')
    p_before_tokens = model.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=True).to(device).input_ids
    p_mid_tokens = model.llama_tokenizer(p_mid, return_tensors="pt", add_special_tokens=False).to(device).input_ids
    p_after_tokens = model.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device).input_ids

    # embedding
    p_before_embeds = peft_model.base_model.model.model.embed_tokens(p_before_tokens)
    p_mid_embeds = peft_model.base_model.model.model.embed_tokens(p_mid_tokens)
    p_after_embeds = peft_model.base_model.model.model.embed_tokens(p_after_tokens)
    mixed_embs = torch.cat([p_before_embeds, img_list[0], p_mid_embeds, coarse_report_embeds, p_after_embeds], dim=1)
    mixed_embs = torch.cat([p_mid_embeds, coarse_report_embeds, p_after_embeds], dim=1)
    
    # prepare other things before generate
    stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    # generate
    outputs = peft_model.base_model.model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=args.max_txt_len,
            stopping_criteria=stopping_criteria,
            num_beams=args.beam_size,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=args.temperature,)
    
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    refined_text = output_text
    
    print('Generated report:')
    print(refined_text)
