import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torchvision.models as models

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft.utils.save_and_load import set_peft_model_state_dict

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    
        # configurations of the pretraining stage
        is_pretraining_stage=True,
        freeze_llama=True,
        pretraining_ckpt=None,
        
        # configurations for contrastive loss or refinement loss
        use_contrastive_loss=False,
        use_refinement_loss=False,
        triplet_margin=1.0,
        triplet_weight=1.0,
        refinement_loss_weight=1.0,
        generation_prompt_path="",
        refinement_prompt_path="",
        
        # lora configurations
        use_lora=False,
        lora_rank=None,
        lora_alpha=None,
        lora_dropout=None,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.use_lora = use_lora
        peft_config = LoraConfig(inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # use the ResNet101 in R2Gen as visual encoder
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.ln_vision.train = disabled_train
        logging.info("freeze vision encoder")
        
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = disabled_train
        self.query_tokens.requires_grad = False
        logging.info("freeze Qformer")
        
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if freeze_llama:
            print("Freeze LLaMA model.")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            if self.use_lora:
                print("Freeze LLaMA model still, using LoRA to update.")
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                self.peft_model = get_peft_model(self.llama_model, peft_config)
            else:
                print("Do full parameters tuning.")
        if pretraining_ckpt is not None:
            if self.use_lora:
                print("LLaMA LoRA model loaded.")
                full_state_dict = torch.load(pretraining_ckpt, map_location='cpu')['module']
                set_peft_model_state_dict(self.peft_model, full_state_dict)    
        
        print('Loading LLAMA Done')
        
        self.llama_proj = nn.Linear(
            768, self.llama_model.config.hidden_size
        )
        
        if pretraining_ckpt:
            linear_state_dict = {}
            for key, value in full_state_dict.items():
                if 'llama_proj' in key:
                    linear_state_dict[key[18:]] = value
            self.llama_proj.load_state_dict(linear_state_dict)
            print('Linear projection layer loaded.')
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        # TODO: configurations for MIMIC
        self.use_contrastive_loss = use_contrastive_loss
        self.use_refinement_loss = use_refinement_loss
        self.triplet_weight = triplet_weight
        self.refinement_loss_weight = refinement_loss_weight
        if self.use_contrastive_loss:
            self.triplet_criterion = nn.TripletMarginLoss(margin=triplet_margin, p=2)

        # load generation prompt
        if generation_prompt_path:
            with open(generation_prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.generation_prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} generation prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.generation_prompt_list)))
          
        # load refinement prompt  
        if refinement_prompt_path:
            with open(refinement_prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.refinement_prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} refinement prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.refinement_prompt_list)))        
        
        # is it pretraining stage or finetuning stage?
        if is_pretraining_stage:
            pass
        else:
            self.forward = self.generate_then_refine        

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
      
    # TODO: input additional report and the visual image    
    def prompt_wrap_with_report(self, img_embeds, coarse_report_embeds, atts_img, atts_report, prompt):
        if prompt:          
            # split out image
            batch_size = img_embeds.shape[0]
            p_before, p_after_all = prompt.split('<ImageHere>')
            p_mid, p_after = p_after_all.split('<ReportHere>')     # p_mid is </Img><Report>
            
            # tokenization
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            
            p_mid_tokens = self.llama_tokenizer(
                p_mid, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            
            # embedding
            p_before_embeds = self.peft_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_mid_embeds = self.peft_model.base_model.model.model.embed_tokens(p_mid_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.peft_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_mid_embeds, coarse_report_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            wrapped_atts_report = atts_report[:, :1].expand(-1, atts_report.size(1))    
            return wrapped_img_embeds, wrapped_atts_img, wrapped_atts_report
        else:
            return img_embeds, atts_img, atts_report

    def forward(self, samples):
        image = samples["image"]
        
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        # turn pad_token_id into -100
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        
        if self.use_lora:
            bos_embeds = self.peft_model.base_model.model.model.embed_tokens(bos)
        else:
            bos_embeds = self.llama_model.model.embed_tokens(bos)
        
        atts_bos = atts_img[:, :1]

        if self.use_lora:
            to_regress_embeds = self.peft_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        else:
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        
        
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            if self.use_lora:
                outputs = self.peft_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss

        return {"loss": loss}

    def generate_then_refine(self, samples):
        with torch.no_grad():
                prompt = random.choice(self.generation_prompt_list)
                
                # load image
                img_list = []
                image = samples['image']
                image_emb, _ = self.encode_img(image)
                img_list.append(image_emb)
                
                # wrap image with prompt
                prompt_segs = prompt.split('<ImageHere>')
                seg_tokens = [
                    self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=i == 0).to(image_emb.device).input_ids
                    # only add bos to the first seg
                    for i, seg in enumerate(prompt_segs)
                ]
                seg_embs = [self.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
                mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
                
                # batch size should be smaller than 1 during testing LLM
                # chunking the image embedding, where we only need one image as input to generate report
                mixed_embs[1] = torch.chunk(mixed_embs[1], chunks=2, dim=0)[0]
                
                mixed_embs = torch.cat(mixed_embs, dim=1)
                
                # prepare other things before generate
                stop_words_ids = [torch.tensor([835]).to(image_emb.device),
                                    torch.tensor([2277, 29937]).to(image_emb.device)]  # '###' can be encoded in two different ways.
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
                
                # generate
                outputs = self.llama_model.generate(
                        inputs_embeds=mixed_embs,
                        max_new_tokens=160,
                        stopping_criteria=stopping_criteria,
                        num_beams=1,
                        do_sample=True,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        temperature=1,)
                output_token = outputs[0]
                if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                    output_token = output_token[1:]
                if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                    output_token = output_token[1:]
                output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
                output_text = output_text.split('###')[0]  # remove the stop sign '###'
                output_text = output_text.split('Assistant:')[-1].strip()
        
        if self.use_contrastive_loss:
            # TODO: ask LLM to write a radiology report (stage 1)
            image = samples["image"]
            img_embeds, atts_img = self.encode_img(image)
            if hasattr(samples, 'question_split'):  # VQA dataset
                print('VQA Batch')
                vqa_prompt = '###Human: <Img><ImageHere></Img> '
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
            elif self.generation_prompt_list:
                prompt = random.choice(self.generation_prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

            self.llama_tokenizer.padding_side = "right"

            # TODO: process reference report
            all_ref_reports = samples['ref_caption']
            all_unrelated_reports = samples['unlabeled_caption']
            
            ref_inputs_embeds_list = []
            ref_inputs_masks_list = []
            neg_inputs_embeds_list = []
            neg_inputs_masks_list = []
            for i in range(len(all_ref_reports)):
                ref_report = [t + self.end_sym for t in all_ref_reports[i]]
                ref_report_tokens = self.llama_tokenizer(
                    ref_report,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False
                ).to(image.device)
                neg_report = [t + self.end_sym for t in all_unrelated_reports[i]]
                neg_report_tokens = self.llama_tokenizer(
                    neg_report,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False
                ).to(image.device)
                gen_report = output_text
                gen_report_tokens = self.llama_tokenizer(
                    gen_report,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False
                ).to(image.device)
                length = max(gen_report_tokens.input_ids.size(1), ref_report_tokens.input_ids.size(1), neg_report_tokens.input_ids.size(1))
                
                # pad three reports to the same size
                if gen_report_tokens.input_ids.size(1) < length:
                    placeholder = torch.ones(gen_report_tokens.input_ids.size(0), length, dtype=torch.long).to(image.device)
                    placeholder = placeholder * self.llama_tokenizer.pad_token_id
                    placeholder[:, :gen_report_tokens.input_ids.size(1)] = gen_report_tokens.input_ids
                    gen_report_tokens.input_ids = placeholder
                    gen_report_tokens.attention_mask = torch.ones(gen_report_tokens.input_ids.size(0), gen_report_tokens.input_ids.size()[-1], dtype=torch.long).to(image.device)
                    
                if ref_report_tokens.input_ids.size(1) < length:
                    placeholder = torch.ones(ref_report_tokens.input_ids.size(0), length, dtype=torch.long).to(image.device)
                    placeholder = placeholder * self.llama_tokenizer.pad_token_id
                    placeholder[:, :ref_report_tokens.input_ids.size(1)] = ref_report_tokens.input_ids
                    ref_report_tokens.input_ids = placeholder
                    ref_report_tokens.attention_mask = torch.ones(ref_report_tokens.input_ids.size(0), ref_report_tokens.input_ids.size()[-1], dtype=torch.long).to(image.device)
                    
                if neg_report_tokens.input_ids.size(1) < length:
                    placeholder = torch.ones(neg_report_tokens.input_ids.size(0), length, dtype=torch.long).to(image.device)
                    placeholder = placeholder * self.llama_tokenizer.pad_token_id
                    placeholder[:, :neg_report_tokens.input_ids.size(1)] = neg_report_tokens.input_ids
                    neg_report_tokens.input_ids = placeholder
                    neg_report_tokens.attention_mask = torch.ones(neg_report_tokens.input_ids.size(0), neg_report_tokens.input_ids.size()[-1], dtype=torch.long).to(image.device)
                
                reference_report_targets = ref_report_tokens.input_ids.masked_fill(
                    ref_report_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )
                ref_empty_targets = (
                    torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                            dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
                )
                reference_report_targets = torch.cat([ref_empty_targets, reference_report_targets], dim=1)
                batch_size = img_embeds.shape[0]
                bos = torch.ones([batch_size, 1],
                                dtype=ref_report_tokens.input_ids.dtype,
                                device=ref_report_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.peft_model.base_model.model.model.embed_tokens(bos)
                atts_bos = atts_img[:, :1]
                
                # TODO: process gt report before forwarding
                gen_report_targets = gen_report_tokens.input_ids.masked_fill(
                    gen_report_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )
                gt_empty_targets = (
                    torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                            dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
                )
                gen_report_targets = torch.cat([gt_empty_targets, gen_report_targets], dim=1)
                batch_size = img_embeds.shape[0]
                bos = torch.ones([batch_size, 1],
                                dtype=gen_report_tokens.input_ids.dtype,
                                device=gen_report_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.peft_model.base_model.model.model.embed_tokens(bos)
                atts_bos = atts_img[:, :1]
                gen_report_embeds = self.peft_model.base_model.model.model.embed_tokens(gen_report_tokens.input_ids)
                gt_inputs_embeds = torch.cat([bos_embeds, img_embeds, gen_report_embeds], dim=1)
                gt_attention_mask = torch.cat([atts_bos, atts_img, gen_report_tokens.attention_mask], dim=1)

                # TODO: process positive report (reference report) before forwarding
                ref_report_targets = ref_report_tokens.input_ids.masked_fill(
                    ref_report_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )
                ref_empty_targets = (
                    torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                            dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
                )
                ref_report_targets = torch.cat([ref_empty_targets, ref_report_targets], dim=1)
                batch_size = img_embeds.shape[0]
                bos = torch.ones([batch_size, 1],
                                dtype=ref_report_tokens.input_ids.dtype,
                                device=ref_report_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.peft_model.base_model.model.model.embed_tokens(bos)
                atts_bos = atts_img[:, :1]
                ref_report_embeds = self.peft_model.base_model.model.model.embed_tokens(ref_report_tokens.input_ids)
                ref_inputs_embeds = torch.cat([bos_embeds, img_embeds, ref_report_embeds], dim=1)
                ref_attention_mask = torch.cat([atts_bos, atts_img, ref_report_tokens.attention_mask], dim=1)

                # TODO: process negative report (unlabled report) before forwarding
                neg_report_targets = neg_report_tokens.input_ids.masked_fill(
                    neg_report_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )
                neg_empty_targets = (
                    torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                            dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
                )
                neg_report_targets = torch.cat([neg_empty_targets, neg_report_targets], dim=1)
                batch_size = img_embeds.shape[0]
                bos = torch.ones([batch_size, 1],
                                dtype=neg_report_tokens.input_ids.dtype,
                                device=neg_report_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.peft_model.base_model.model.model.embed_tokens(bos)
                atts_bos = atts_img[:, :1]
                neg_report_embeds = self.peft_model.base_model.model.model.embed_tokens(neg_report_tokens.input_ids)
                neg_inputs_embeds = torch.cat([bos_embeds, img_embeds, neg_report_embeds], dim=1)
                neg_attention_mask = torch.cat([atts_bos, atts_img, neg_report_tokens.attention_mask], dim=1)

                # all the processed inputs into the lists
                ref_inputs_embeds_list.append(ref_inputs_embeds)
                ref_inputs_masks_list.append(ref_attention_mask)
                neg_inputs_embeds_list.append(neg_inputs_embeds)
                neg_inputs_masks_list.append(neg_attention_mask)

            # forwarding
            all_losses = []
            for i in range(len(ref_inputs_embeds_list)):
                with self.maybe_autocast():  
                    if i > 0:          
                        outputs_hidden_states = self.peft_model(
                            inputs_embeds=gt_inputs_embeds,
                            attention_mask=gt_attention_mask,
                            return_dict=True,
                        )
                    else:
                        outputs_hidden_states = self.peft_model(
                            inputs_embeds=gt_inputs_embeds.detach(),
                            attention_mask=gt_attention_mask.detach(),
                            return_dict=True,
                        )
                    
                    positive_hidden_states = self.peft_model(
                            inputs_embeds=ref_inputs_embeds_list[i],
                            attention_mask=ref_inputs_masks_list[i],
                            return_dict=True,
                        )
                    
                    # negative
                    negative_hidden_states = self.peft_model(
                        inputs_embeds=neg_inputs_embeds_list[i],
                        attention_mask=neg_inputs_masks_list[i],
                        return_dict=True,
                        )
            
                # compute the triplet contrastive loss
                triplet_loss = self.triplet_criterion(outputs_hidden_states['hidden_states'],
                                                    positive_hidden_states['hidden_states'],
                                                    negative_hidden_states['hidden_states'])
                triplet_loss *= self.triplet_weight
                all_losses.append(triplet_loss)
            i3_loss = 0.0
            for loss in all_losses:
                i3_loss += loss.item() / len(ref_inputs_embeds_list)
                
        if self.use_refinement_loss:
            # TODO: ask LLM to refine the generated report (stage 2)
            image = samples["image"]
            img_embeds, atts_img = self.encode_img(image)
            if hasattr(samples, 'question_split'):  # VQA dataset
                print('VQA Batch')
                vqa_prompt = '###Human: <Img><ImageHere></Img> '
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
            elif self.refinement_prompt_list:
                before_report, after_report = random.choice(self.refinement_prompt_list).split("<ReportHere>")
                prompt = before_report + output_text + after_report
                
                # analysis - use visual embeddings in c2fd
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                    
                atts_report = torch.ones([img_embeds.size(0), output_token.size(-1)], dtype=torch.long).to(img_embeds.device)

            self.llama_tokenizer.padding_side = "right"
            
            # preprocess
            gt_report = [t + self.end_sym for t in samples["text_input"]]
            gt_report_tokens = self.llama_tokenizer(
                gt_report,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)
            gt_report_targets = gt_report_tokens.input_ids.masked_fill(
                gt_report_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            gt_empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            gt_report_targets = torch.cat([gt_empty_targets, gt_report_targets], dim=1)
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=gt_report_tokens.input_ids.dtype,
                            device=gt_report_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.peft_model.model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]
            gt_report_embeds = self.peft_model.model.model.embed_tokens(gt_report_tokens.input_ids)
            gt_inputs_embeds = torch.cat([bos_embeds, img_embeds, gt_report_embeds], dim=1)
            gt_attention_mask = torch.cat([atts_bos, atts_img, gt_report_tokens.attention_mask], dim=1)
            
            # forwarding
            with self.maybe_autocast():
                outputs = self.peft_model(
                    inputs_embeds=gt_inputs_embeds,
                    attention_mask=gt_attention_mask,
                    return_dict=True,
                    labels=gt_report_targets,
                )
            refinement_loss = outputs.loss * self.refinement_loss_weight
        else:
            refinement_loss = 0.0
            
        loss = i3_loss + refinement_loss
        
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        # TODO: tune MiniGPT-4 on MIMIC-CXR with additional objectives
        # pretraining stage
        is_pretraining_stage = cfg.get("is_pretraining_stage", True)
        pretraining_ckpt = cfg.get("pretraining_ckpt", None)
        
        # finetuning stage
        use_contrastive_loss = cfg.get("use_contrastive_loss", False)
        use_refinement_loss = cfg.get("use_refinement_loss", False)
        triplet_margin = cfg.get("triplet_margin", 1.0)
        triplet_weight = cfg.get("triplet_weight", 1.0)
        refinement_loss_weight = cfg.get("refinement_loss_weight", 1.0)
        generation_prompt_path = cfg.get("generation_prompt_path", None)
        refinement_prompt_path = cfg.get("refinement_prompt_path", None)
        freeze_llama = cfg.get("freeze_llama", True)
        
        # configurations for medical encoders
        
        # some choices on PEFT methods
        use_lora = cfg.get("use_lora", False)
        lora_rank = cfg.get("lora_rank", None)
        lora_alpha = cfg.get("lora_alpha", None)
        lora_dropout = cfg.get("lora_dropout", None)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            
            # pretraining stage
            is_pretraining_stage=is_pretraining_stage,
            pretraining_ckpt=pretraining_ckpt,
            
            # contrastive loss and refinement loss
            use_contrastive_loss=use_contrastive_loss,
            use_refinement_loss=use_refinement_loss,
            triplet_margin=triplet_margin,
            triplet_weight=triplet_weight,
            refinement_loss_weight=refinement_loss_weight,
            generation_prompt_path=generation_prompt_path,
            refinement_prompt_path=refinement_prompt_path,
            freeze_llama=freeze_llama,
            
            # lora configurations
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
