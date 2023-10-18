import json
import logging
import re
from collections import defaultdict

import pytorch_lightning as pl
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartConfig

from .constrained_gen import BartConstrainedGen
from .constrained_gen_calibrate import BartConstrainedGenCalibrate
from .utils import load_ontology

logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer, util

sim_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
MAX_LENGTH = 512


class GenIEModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

        # bart-large
        self.config = BartConfig.from_pretrained("facebook/bart-large")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.tokenizer.add_tokens([" <arg>", " <tgr>"])  # argument & trigger
        
        if self.hparams.model == "constrained-gen":
            if self.hparams.calibrate:
                self.model = BartConstrainedGenCalibrate(self.config, self.tokenizer)
                self.model.resize_token_embeddings()
            else:
                self.model = BartConstrainedGen(self.config, self.tokenizer)
                self.model.resize_token_embeddings()
        else:
            raise NotImplementedError
        
        self.pair_constraints = {
            ("Justice.Sentence.Unspecified_JudgeCourt", "Life.Die.Unspecified_Victim"),
            ("Justice.Sentence.Unspecified_Defendant", "Life.Die.Unspecified_Victim"),
            ("Control.ImpedeInterfereWith.Unspecified_Impeder", "Justice.ArrestJailDetain.Unspecified_Jailer"),
            ("Contact.RequestCommand.Unspecified_Recipient", "Justice.ArrestJailDetain.Unspecified_Jailer"),
            ("Life.Injure.Unspecified_Injurer", "Transaction.ExchangeBuySell.Unspecified_Giver"),
            ("Justice.TrialHearing.Unspecified_Defendant", "Transaction.ExchangeBuySell.Unspecified_Giver"),
            ("Justice.TrialHearing.Unspecified_Defendant", "Transaction.ExchangeBuySell.Unspecified_Recipient"),
            ("Justice.Sentence.Unspecified_JudgeCourt", "Life.Die.Unspecified_Victim"),
            ("Justice.ArrestJailDetain.Unspecified_Detainee", "Justice.ArrestJailDetain.Unspecified_Detainee"),
            ("Conflict.Attack.DetonateExplode_Attacker", "Contact.Contact.Broadcast_Communicator"),
            ("Conflict.Attack.Unspecified_Attacker", "Contact.Contact.Broadcast_Communicator"),
            ("Conflict.Attack.DetonateExplode_Attacker", "Contact.ThreatenCoerce.Unspecified_Communicator"),
            ("Conflict.Attack.Unspecified_Attacker", "Contact.ThreatenCoerce.Unspecified_Communicator"),
        }
        self.up_constraints = {
            "Killer_Attacker_Injurer_Damager_Destroyer": "Killer_Attacker_Destroyer_Defendant",
            "JudgeCourt": "JudgeCourt",
        }
        self.up_thresh = 4

        self.ontology_dict = load_ontology(dataset=self.hparams.dataset)
        for key in self.ontology_dict:
            for role in self.ontology_dict[key]["arg_to_prev"]:
                w = self.ontology_dict[key]["arg_to_prev"][role]
                if w == "<s>":
                    self.ontology_dict[key]["arg_to_prev"][role] = [w, 2]
                else:
                    w_list = self.tokenizer.tokenize(w, add_prefix_space=True)
                    self.ontology_dict[key]["arg_to_prev"][role] = \
                        [w, self.tokenizer.encode_plus(w_list, add_special_tokens=True, add_prefix_space=True)["input_ids"][-2]]

        self.memory = {}
        self.memory_down = {}
        self.memory_up_cnt = defaultdict(int)

        with open("preprocessed_simtr_{}/test.jsonl".format(self.hparams.dataset), "r") as f:
            for line in f:
                ex = json.loads(line.strip())
                doc_key = ex["doc_key"]
                # {doc_key: {evt_type: {role: [], role: [], ...}, evt_type: {role: [], ...}}}
                if doc_key not in self.memory:
                    self.memory[doc_key] = {}
                    self.memory_down[doc_key] = {}
                    self.memory_up_cnt[doc_key] = {}
                    for evt_type in self.ontology_dict:
                        self.memory[doc_key][evt_type] = {}
                        self.memory_down[doc_key][evt_type] = {}
                        for role in self.ontology_dict[evt_type]["roles"]:
                            if role not in self.memory[doc_key][evt_type]:
                                self.memory[doc_key][evt_type][role] = []
                                self.memory_down[doc_key][evt_type][role] = []
                    for role_grp_key, role_grp in self.up_constraints.items():
                        if role_grp not in self.memory_up_cnt[doc_key]:
                            self.memory_up_cnt[doc_key][role_grp] = {}  # ent1: #, ent2: #
                            if role_grp_key == "JudgeCourt":
                                ent = "George O'Toole Jr."
                                w_list = self.tokenizer.tokenize("Jr.", add_prefix_space=True)
                                out_id = \
                                    self.tokenizer.encode_plus(w_list, add_special_tokens=True, add_prefix_space=True)[
                                        "input_ids"][1]
                                self.memory_up_cnt[doc_key][role_grp][ent] = [out_id, self.up_thresh]

        self.all_output_templates, self.all_out_template_embs = {}, {}
        self.all_outputs, self.all_assists = {}, {}
        for doc_key in self.memory:
            if doc_key not in self.all_output_templates:
                self.all_output_templates[doc_key] = []
                self.all_out_template_embs[doc_key] = []
                self.all_outputs[doc_key] = []
                self.all_assists[doc_key] = []

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):

        inputs = {
            "input_ids": batch["input_token_ids"],
            "attention_mask": batch["input_attn_mask"],
            "decoder_input_ids": batch["tgt_token_ids"],
            "decoder_attention_mask": batch["tgt_attn_mask"],
            "task": 0
        }

        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)

        log = {
            "train/loss": loss,
        }
        return {
            "loss": loss,
            "log": log
        }

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_token_ids"],
            "attention_mask": batch["input_attn_mask"],
            "decoder_input_ids": batch["tgt_token_ids"],
            "decoder_attention_mask": batch["tgt_attn_mask"],
            "task": 0,
        }
        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            "val/loss": avg_loss,
        }
        return {
            "loss": avg_loss,
            "log": log
        }

    def extract_args_from_template(self, evt_type, pred_template, input_ids=None, token2prob=None):
        if input_ids and token2prob:
            i_ptr, o_ptr = 0, 0
            ents = []
            while i_ptr < len(input_ids) and o_ptr < len(token2prob):
                if input_ids[i_ptr] != token2prob[o_ptr][0]:  # may detect an argument
                    arg_start = o_ptr
                    while (o_ptr < len(token2prob)) and (
                            (i_ptr == len(input_ids) - 1) or token2prob[o_ptr][0] != input_ids[i_ptr + 1]):
                        o_ptr += 1
                    if input_ids[i_ptr] == 50265:  # ensure this is an argument
                        ents.append(token2prob[arg_start])
                    i_ptr += 1
                else:
                    i_ptr += 1
                    o_ptr += 1
        # extract argument text
        template = self.ontology_dict[evt_type]["template"]
        template_words = template.strip().split()
        predicted_words = pred_template.strip().split()
        predicted_args = defaultdict(list)  # each argname may have multiple participants
        t_ptr = 0
        p_ptr = 0
        ptr = 0
        while t_ptr < len(template_words) and p_ptr < len(predicted_words):
            if re.match(r"<(arg\d+)>", template_words[t_ptr]):  # found <argx> in original template
                m = re.match(r"<(arg\d+)>", template_words[t_ptr])
                arg_num = m.group(1)
                try:  # obtain argument name
                    arg_name = self.ontology_dict[evt_type][arg_num]
                except KeyError:
                    print(evt_type)
                    exit()
                if predicted_words[p_ptr] == "<arg>":  # missing argument
                    p_ptr += 1
                    t_ptr += 1
                else:  # find argument span
                    arg_start = p_ptr
                    while (p_ptr < len(predicted_words)) and ((t_ptr == len(template_words) - 1) or (
                            predicted_words[p_ptr] != template_words[t_ptr + 1])):
                        p_ptr += 1
                    arg_text = predicted_words[arg_start:p_ptr]
                    if input_ids and token2prob:
                        if ptr < len(ents):
                            arg_text.extend(ents[ptr])
                            ptr += 1
                        else:
                            arg_text.append(-1)
                    # argument name -> argument span
                    predicted_args[arg_name].append(arg_text)
                    t_ptr += 1
            else:
                t_ptr += 1
                p_ptr += 1

        return predicted_args

    def test_step(self, batch, batch_idx):
        if self.hparams.constrained_decoding:  # decoding based on constraint pairs
            doc_key = batch["doc_key"][0]
            evt_type = batch["event_type"][0]

            # the id of the word before the argument name -> [the ids of the first token of each entity]
            # entity: another entity within a pair constraint
            id_pairs_down = {}
            # the word before the argument name -> [all tokens of each entity]
            id_pairs_down_print = {}
            for role, ents in self.memory_down[doc_key][evt_type].items():  # argument name -> entities
                # the id of the word before the argument name
                in_id = self.ontology_dict[evt_type]["arg_to_prev"][role][-1]
                if ents:
                    down_out_ids = []
                    down_out_ids_print = []
                    for ent in ents[:]:
                        # the id of the first token of the entity
                        down_out_ids.append(ent[-1])
                        # all tokens of the entity
                        down_out_ids_print.append(ent[:-1])
                    id_pairs_down[in_id] = down_out_ids
                    id_pairs_down_print[self.ontology_dict[evt_type]["arg_to_prev"][role][0]] = down_out_ids_print

                    if role == "Participant":  # fix participant exception (2 roles)
                        in_id2 = 19
                        id_pairs_down[in_id2] = down_out_ids

            # the id of the word before the argument name -> [the ids of the first token of each entity]
            # entity: appears more than up_thresh times (in the input)
            id_pairs_up = {}
            for role in self.ontology_dict[evt_type]["roles"]:
                for role_grp_key, role_grp in self.up_constraints.items():
                    if role in role_grp:
                        # the id of the word before the argument name
                        in_id = self.ontology_dict[evt_type]["arg_to_prev"][role][-1]
                        for ent in self.memory_up_cnt[doc_key][role_grp]:
                            # the current threshold >= threshold of the previous step,
                            # and the entity is in the input
                            if self.memory_up_cnt[doc_key][role_grp][ent][-1] >= self.up_thresh and \
                                    self.memory_up_cnt[doc_key][role_grp][ent][0] in batch["input_token_ids"]:
                                if in_id not in id_pairs_up:
                                    id_pairs_up[in_id] = []
                                id_pairs_up[in_id].append(self.memory_up_cnt[doc_key][role_grp][ent][0])

        input_token_ids = batch["input_token_ids"]
        retrieved_template = []
        if self.hparams.retrieval_augmented:
            # find the most similar template
            doc_key = batch["doc_key"][0]
            context_emb = sim_model.encode(batch["context_words"][0], show_progress_bar=False)
            most_sim_out_template = []
            context = batch["context_tokens"][0]
            if len(self.all_out_template_embs[doc_key]) > 0:
                cosine_scores = util.pytorch_cos_sim([context_emb], self.all_out_template_embs[doc_key])
                _, most_sim_idx = torch.max(cosine_scores, dim=-1)
                most_sim_out_template = self.all_output_templates[doc_key][most_sim_idx]
                retrieved_template = self.all_outputs[doc_key][most_sim_idx]

            context = most_sim_out_template + ["</s>"] + context
            input_tokens = self.tokenizer.encode_plus(batch["input_template"][0], context,
                                                      add_special_tokens=True,
                                                      add_prefix_space=True,
                                                      max_length=MAX_LENGTH,
                                                      truncation="only_second",
                                                      padding="max_length")

            input_token_ids = torch.stack([torch.LongTensor(input_tokens["input_ids"])])
            if batch["input_token_ids"].device.type != "cpu":
                input_token_ids = input_token_ids.cuda()

        # input_ids
        evt_type = batch["event_type"][0]
        input_template = self.ontology_dict[evt_type]["template"]
        input_template = re.sub(r"<arg\d>", "<arg>", input_template)
        tokenized_input = []
        for w in input_template.split():
            tokenized_w = self.tokenizer.tokenize(w, add_prefix_space=True)
            tokenized_input.extend(tokenized_w)
        input_ids = self.tokenizer.encode(tokenized_input, add_special_tokens=True, add_prefix_space=True,
                                          max_length=MAX_LENGTH, truncation="only_second")[1:]

        if self.hparams.constrained_decoding:
            generated_output, token2prob = self.model.generate(id_pairs_down, id_pairs_up,
                                                                         input_ids=input_token_ids, do_sample=False,
                                                                         max_length=30, num_return_sequences=1,
                                                                         num_beams=1)
            doc_key = batch["doc_key"][0]
            evt_type = batch["event_type"][0]
            pred_temp = self.tokenizer.decode(generated_output.squeeze(0), skip_special_tokens=True)
            if self.hparams.revised_constraints:
                pred_args = self.extract_args_from_template(evt_type, pred_temp, input_ids, token2prob)
            else:
                pred_args = self.extract_args_from_template(evt_type, pred_temp)
            
            for role in pred_args:
                for ent in pred_args[role]:
                    if self.hparams.revised_constraints:
                        if not ent or ent[-1] == -1:
                            continue
                        self.memory[doc_key][evt_type][role].append(ent[:-1])
                        # lower and upper bounds
                        if 0.5 <= ent[-1] <= 0.8:
                            # down
                            evt_type_role = "_".join([evt_type, role])
                            for pair in self.pair_constraints:
                                if evt_type_role == pair[0]:
                                    evt_type2, role2 = pair[1].split("_")
                                    self.memory_down[doc_key][evt_type2][role2].append(ent[:-1])
                                if evt_type_role == pair[1]:
                                    evt_type2, role2 = pair[0].split("_")
                                    self.memory_down[doc_key][evt_type2][role2].append(ent[:-1])
                            # up
                            for role_grp_key, role_grp in self.up_constraints.items():
                                if role in role_grp_key:
                                    if ent[0] not in self.memory_up_cnt[doc_key][role_grp]:
                                        self.memory_up_cnt[doc_key][role_grp][ent[0]] = [ent[-2], 1]
                                    else:
                                        self.memory_up_cnt[doc_key][role_grp][ent[0]][-1] += 1
                    else:
                        if not ent:
                            continue
                        w_list = self.tokenizer.tokenize(ent[0], add_prefix_space=True)
                        # the id of the first token of the entity
                        out_id = \
                            self.tokenizer.encode_plus(w_list, add_special_tokens=True, add_prefix_space=True)["input_ids"][
                                1]
                        ent.append(out_id)
                        # add into the memory
                        self.memory[doc_key][evt_type][role].append(ent)
                        # down
                        evt_type_role = "_".join([evt_type, role])
                        for pair in self.pair_constraints:
                            if evt_type_role == pair[0]:
                                evt_type2, role2 = pair[1].split("_")
                                self.memory_down[doc_key][evt_type2][role2].append(ent)
                                print(evt_type_role, "_".join([evt_type2, role2]), ent)
                            if evt_type_role == pair[1]:
                                evt_type2, role2 = pair[0].split("_")
                                self.memory_down[doc_key][evt_type2][role2].append(ent)
                                print(evt_type_role, "_".join([evt_type2, role2]), ent)
                        # up
                        for role_grp_key, role_grp in self.up_constraints.items():
                            if role in role_grp_key:
                                if ent[0] not in self.memory_up_cnt[doc_key][role_grp]:
                                    self.memory_up_cnt[doc_key][role_grp][ent[0]] = [out_id, 1]
                                else:
                                    self.memory_up_cnt[doc_key][role_grp][ent[0]][-1] += 1
        else:
            # normal decoding
            generated_output, token2prob = self.model.generate({}, {},
                                                                         input_ids=input_token_ids, do_sample=False,
                                                                         max_length=30, num_return_sequences=1,
                                                                         num_beams=1)

        generated_output = generated_output.reshape(batch["input_token_ids"].size(0), 1, -1)
        doc_key = batch["doc_key"]
        tgt_token_ids = batch["tgt_token_ids"]

        i_ptr, o_ptr = 0, 0
        arg_probs = []
        arg_o_ids, arg_g_ids = [], []
        while i_ptr < len(input_ids) and o_ptr < len(token2prob):
            if input_ids[i_ptr] != token2prob[o_ptr][0]:  # may detect an argument
                arg_start = o_ptr
                while (o_ptr < len(token2prob)) and (
                        (i_ptr == len(input_ids) - 1) or token2prob[o_ptr][0] != input_ids[i_ptr + 1]):
                    o_ptr += 1
                if input_ids[i_ptr] == 50265:  # ensure this is an argument
                    arg_probs.append([token2prob[i][1] for i in range(arg_start, o_ptr)])
                    arg_o_ids.append(token2prob[arg_start][0])
                i_ptr += 1
            else:
                if input_ids[i_ptr] == 50265:
                    arg_probs.append([])
                    arg_o_ids.append(-1)
                i_ptr += 1
                o_ptr += 1

        i_ptr, g_ptr = 0, 0
        gold_ids = tgt_token_ids.squeeze()
        gold_ids = gold_ids[gold_ids != 1].tolist()[1:]
        while i_ptr < len(input_ids) and g_ptr < len(gold_ids):
            if input_ids[i_ptr] != gold_ids[g_ptr]:
                arg_start = g_ptr
                while (g_ptr < len(gold_ids)) and (
                        (i_ptr == len(input_ids) - 1) or gold_ids[g_ptr] != input_ids[i_ptr + 1]):
                    g_ptr += 1
                if input_ids[i_ptr] == 50265:
                    arg_g_ids.append(gold_ids[arg_start])
                i_ptr += 1
            else:
                if input_ids[i_ptr] == 50265:
                    arg_g_ids.append(gold_ids[g_ptr])
                i_ptr += 1
                g_ptr += 1

        length = min(len(arg_probs), len(arg_o_ids), len(arg_g_ids))
        arg_probs = arg_probs[:length]
        arg_o_ids = arg_o_ids[:length]
        arg_g_ids = arg_g_ids[:length]
        confs, accus = [], []
        for l in range(length):
            if arg_o_ids[l] == -1:
                continue
            confs.append(arg_probs[l])
            accus.append(int(arg_o_ids[l] == arg_g_ids[l]))

        if self.hparams.retrieval_augmented:
            output_template = self.tokenizer.decode(generated_output[0][0], skip_special_tokens=True)
            out_template_emb = sim_model.encode(output_template, show_progress_bar=False)
            space_tokenized_template = output_template.split()
            tokenized_output_template = []
            for w in space_tokenized_template:
                tokenized_output_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
            # add to memory
            self.all_output_templates[doc_key[0]].append(tokenized_output_template)
            self.all_out_template_embs[doc_key[0]].append(out_template_emb)
            self.all_outputs[doc_key[0]].append(output_template)

        return doc_key, generated_output, tgt_token_ids, retrieved_template, confs, accus

    def test_epoch_end(self, outputs):
        with open("checkpoints/{}/predictions.jsonl".format(self.hparams.ckpt_name), "w") as writer:
            for tup in outputs:
                pred = {
                    "doc_key": tup[0][0],
                    "predicted": self.tokenizer.decode(tup[1].squeeze(), skip_special_tokens=True),
                    "gold": self.tokenizer.decode(tup[2].squeeze(), skip_special_tokens=True),
                    "retrieved": tup[3],
                    "confs": tup[4],
                    "accus": tup[5]
                }
                writer.write(json.dumps(pred) + "\n")
        return {}

    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info("{} training steps in total.. ".format(t_total))

        # prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler is called only once per epoch by default
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=t_total)
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "name": "linear-schedule",
        }

        return [optimizer, ], [scheduler_dict, ]
