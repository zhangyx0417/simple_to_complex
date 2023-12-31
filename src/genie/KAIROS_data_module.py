import json
import os
import re
from collections import defaultdict

import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from .data import IEDataset, my_collate
from .utils import load_ontology, check_pronoun, clean_mention

MAX_CONTEXT_LENGTH = 400  # measured in words
MAX_LENGTH = 512
MAX_TGT_LENGTH = 70


class KAIROSDataModule(pl.LightningDataModule):
    """
    Dataset processing for KAIROS. Involves chunking for long documents.
    """

    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>', ' <tgr>'])
        self.sim_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    def create_gold_gen(self, ex, ontology_dict, mark_trigger=True, index=0, ent2info=None, use_info=False):
        """
        If there are multiple events per example, use index parameter.

        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found.
        """
        if use_info and ent2info is None:
            raise ValueError('entity to informative mention mapping required.')

        # event type
        evt_type = ex['event_mentions'][index]['event_type']

        # obtain tokenized template of the event
        template = ontology_dict[evt_type]['template']
        input_template = re.sub(r'<arg\d>', '<arg>', template)
        space_tokenized_input_template = input_template.split()
        tokenized_input_template = []
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        # role -> arguments
        role2arg = defaultdict(list)
        for argument in ex['event_mentions'][index]['arguments']:
            role2arg[argument['role']].append(argument)
        role2arg = dict(role2arg)

        # create output template
        arg_idx2text = defaultdict(list)
        for role in role2arg.keys():
            if role not in ontology_dict[evt_type]:  # annotation error
                continue
            for i, argument in enumerate(role2arg[role]):
                use_arg = True
                if use_info:
                    ent_id = argument['entity_id']
                    if ent_id in ent2info:
                        # clean up a mention by removing 'a', 'an', 'the' prefixes
                        arg_text = clean_mention(ent2info[ent_id])
                        if check_pronoun(arg_text):  # skip pronouns
                            use_arg = False
                    else: 
                        arg_text = argument['text']
                else:
                    arg_text = argument['text']

                if i < len(ontology_dict[evt_type][role]):  # enough slots to fill in
                    arg_idx = ontology_dict[evt_type][role][i]
                else:  # multiple participants for the same role
                    arg_idx = ontology_dict[evt_type][role][-1]

                if use_arg:
                    arg_idx2text[arg_idx].append(arg_text)

        for arg_idx, text_list in arg_idx2text.items():
            text = ' and '.join(text_list)
            template = re.sub('<{}>'.format(arg_idx), text, template)

        trigger = ex['event_mentions'][index]['trigger']
        offset = 0
        context_words = ex['tokens']
        center_sent = trigger['sent_idx']
        if len(context_words) > MAX_CONTEXT_LENGTH:
            cur_len = len(ex['sentences'][center_sent][0])
            context_words = [tup[0] for tup in ex['sentences'][center_sent][0]]
            if cur_len > MAX_CONTEXT_LENGTH:  # a sentence is very long
                trigger_start = trigger['start']
                # [trigger_start - MAX_CONTEXT_LENGTH // 2, trigger_start + MAX_CONTEXT_LENGTH // 2]
                start_idx = max(0, trigger_start - MAX_CONTEXT_LENGTH // 2)
                end_idx = min(len(context_words), trigger_start + MAX_CONTEXT_LENGTH // 2)
                context_words = context_words[start_idx: end_idx]
                offset = start_idx
            else: # expand current sentence with left and right sentences
                left = center_sent - 1
                right = center_sent + 1
                total_sents = len(ex['sentences'])
                prev_len = 0
                # loop util no new tokens added to context words
                while cur_len > prev_len:
                    prev_len = cur_len
                    if left >= 0:  # left expansion
                        left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                        if cur_len + len(left_sent_tokens) <= MAX_CONTEXT_LENGTH:
                            context_words = left_sent_tokens + context_words
                            left -= 1
                            cur_len += len(left_sent_tokens)
                    if right < total_sents:  # right expansion
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= MAX_CONTEXT_LENGTH:
                            context_words = context_words + right_sent_tokens
                            right += 1
                            cur_len += len(right_sent_tokens)
                # update trigger offset 
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left + 1)])

        assert (len(context_words) <= MAX_CONTEXT_LENGTH)
        # update trigger span
        trigger['start'] = trigger['start'] - offset
        trigger['end'] = trigger['end'] - offset
        # tokenize context words and mark trigger
        if mark_trigger:
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger['start']]), add_prefix_space=True)
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger['start']: trigger['end']]),
                                          add_prefix_space=True)
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger['end']:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)

        # obtain output template and tokenized template
        output_template = re.sub(r'<arg\d>', '<arg>', template)
        space_tokenized_template = output_template.split()
        tokenized_template = []
        for w in space_tokenized_template:
            tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        # calculate S-BERT embeddings of context words and output template
        context_emb = self.sim_model.encode(' '.join(context_words), show_progress_bar=False)
        out_template_emb = self.sim_model.encode(' '.join(space_tokenized_template), show_progress_bar=False)
        return tokenized_input_template, tokenized_template, context, context_emb, out_template_emb, ' '.join(
            context_words)

    def prepare_data(self):
        if self.hparams.retrieval_augmented:  # train with most similar template as additional input
            data_dir = 'preprocessed_simtr_{}'.format(self.hparams.dataset)
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            ontology_dict = load_ontology(self.hparams.dataset)
            max_tokens = 0
            max_tgt = 0
            for split, f in [('train', self.hparams.train_file), ('val', self.hparams.val_file), ('test', self.hparams.test_file)]:
                coref_split = 'dev' if split == 'val' else split
                coref_reader = open(os.path.join(self.hparams.coref_dir, '{}.jsonlines'.format(coref_split)))
                with open(f, 'r') as reader, open(os.path.join(data_dir, '{}.jsonl'.format(split)), 'w') as writer:
                    # original data file, co-reference file
                    for line, coref_line in zip(reader, coref_reader):
                        ex = json.loads(line.strip())
                        corefs = json.loads(coref_line.strip())
                        assert (ex['doc_id'] == corefs['doc_key'])
                        # mapping from entity id to informative mention
                        ent2info = {}
                        for cidx, cluster in enumerate(corefs['clusters']):
                            for eid in cluster: 
                                ent2info[eid] = corefs['informative_mentions'][cidx]

                        all_output_templates, all_out_template_embs = [], []
                        for i in range(len(ex['event_mentions'])):
                            # skip events with no arguments
                            if split == 'train' and len(ex['event_mentions'][i]['arguments']) == 0:
                                continue
                            
                            evt_type = ex['event_mentions'][i]['event_type']
                            
                            # skip events with unknown type
                            if evt_type not in ontology_dict:
                                continue

                            input_template, output_template, context, context_emb, out_template_emb, context_words = \
                                self.create_gold_gen(ex, ontology_dict, self.hparams.mark_trigger, index=i,
                                                     ent2info=ent2info, use_info=self.hparams.use_info)

                            # select the most similar output template
                            most_sim_out_template = []
                            if len(all_out_template_embs) > 0:
                                cosine_scores = util.pytorch_cos_sim([context_emb], all_out_template_embs)
                                most_sim_idx = torch.argmax(cosine_scores, dim=-1)
                                most_sim_out_template = all_output_templates[most_sim_idx]
                            all_output_templates.append(output_template)
                            all_out_template_embs.append(out_template_emb)

                            max_tokens = max(len(context) + len(input_template) + 2, max_tokens)
                            max_tgt = max(len(output_template) + 1, max_tgt)
                            assert (len(output_template) < MAX_TGT_LENGTH)

                            if (split == 'train' or split == 'val') and self.hparams.retrieval_augmented:
                                # input for the generation model
                                context = most_sim_out_template + ['</s>'] + context
                            input_tokens = self.tokenizer.encode_plus(input_template, context,
                                                                      add_special_tokens=True,
                                                                      add_prefix_space=True,
                                                                      max_length=MAX_LENGTH,
                                                                      truncation='only_second',
                                                                      padding='max_length')
                            tgt_tokens = self.tokenizer.encode_plus(output_template,
                                                                    add_special_tokens=True,
                                                                    add_prefix_space=True,
                                                                    max_length=MAX_TGT_LENGTH,
                                                                    truncation=True,
                                                                    padding='max_length')
                            processed_ex = {
                                'event_idx': i,
                                'doc_key': ex['doc_id'],
                                'event_type': evt_type,
                                'input_token_ids': input_tokens['input_ids'],
                                'input_attn_mask': input_tokens['attention_mask'],
                                'tgt_token_ids': tgt_tokens['input_ids'],
                                'tgt_attn_mask': tgt_tokens['attention_mask'],
                                'input_template': input_template,
                                'context_tokens': context,
                                'context_words': context_words
                            }
                            writer.write(json.dumps(processed_ex) + '\n')

            print('longest context:{}'.format(max_tokens))
            print('longest target {}'.format(max_tgt))

    def train_dataloader(self):
        if self.hparams.retrieval_augmented:
            dataset = IEDataset("preprocessed_simtr_{}/train.jsonl".format(self.hparams.dataset))
        else:
            dataset = IEDataset('preprocessed_{}/train.jsonl'.format(self.hparams.dataset))

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, collate_fn=my_collate,
                                batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self):
        if self.hparams.retrieval_augmented:
            dataset = IEDataset("preprocessed_simtr_{}/val.jsonl".format(self.hparams.dataset))
        else:
            dataset = IEDataset('preprocessed_{}/val.jsonl'.format(self.hparams.dataset))

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        if self.hparams.ordering:
            dataset = IEDataset('preprocessed_simtr_KAIROS/ordered_test.jsonl')
        else:
            if self.hparams.retrieval_augmented:
                dataset = IEDataset("preprocessed_simtr_{}/test.jsonl".format(self.hparams.dataset))
            else:
                dataset = IEDataset('preprocessed_{}/test.jsonl'.format(self.hparams.dataset))

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=64, collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader