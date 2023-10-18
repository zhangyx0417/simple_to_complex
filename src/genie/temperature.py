import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
tokenizer.add_tokens([" <arg>", " <tgr>"])

MAX_LENGTH = 512


def temperature_scaling(model, valid_loader, gpus):
    """
    Tune the tempearature of the model (using the validation set).
    """
    gpus = [str(i) for i in gpus]
    gpus = "cuda:" + ",".join(gpus)
    torch.cuda.set_device(gpus)
    
    ece_criterion = Evaluate().cuda()
    model.eval()
    model = model.cuda()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for ex in tqdm(valid_loader):
            inputs = {
                "input_ids": ex["input_token_ids"].cuda(),
                "attention_mask": ex["input_attn_mask"].cuda(),
                "decoder_input_ids": ex["tgt_token_ids"].cuda(),
                "decoder_attention_mask": ex["tgt_attn_mask"].cuda(),
                "task": 0,
            }
            logits, label = model(inputs)
            # output ids
            output_ids = torch.argmax(logits, dim=1)
            output_idx = torch.nonzero(output_ids == 2)
            if output_idx.size(0) > 1:
                output_idx = output_idx[0].item()
            else:
                output_idx = output_idx.item()
            output_ids = output_ids[:output_idx + 1]
            # gold ids
            gold_ids = label[label != -100]
            # input_ids
            input_ids = tokenizer.encode(ex["input_template"][0], add_special_tokens=True, add_prefix_space=True,
                                         max_length=MAX_LENGTH, truncation="only_second")[1:]
            i_ptr, o_ptr = 0, 0
            o_arg_indices = []
            while i_ptr < len(input_ids) and o_ptr < len(output_ids):
                if input_ids[i_ptr] != output_ids[o_ptr]:  # may detect an argument
                    if input_ids[i_ptr] == 50265:
                        o_arg_indices.append(o_ptr)
                    while (o_ptr < len(output_ids)) and (
                            (i_ptr == len(input_ids) - 1) or output_ids[o_ptr] != input_ids[i_ptr + 1]):
                        o_ptr += 1
                    i_ptr += 1
                else:
                    if output_ids[o_ptr] == 50265:
                        o_arg_indices.append(-1)  # mask <arg> in the output sequence
                    i_ptr += 1
                    o_ptr += 1
            i_ptr, g_ptr = 0, 0
            g_arg_indices = []
            while i_ptr < len(input_ids) and g_ptr < len(gold_ids):
                if input_ids[i_ptr] != gold_ids[g_ptr]:
                    if input_ids[i_ptr] == 50265:
                        g_arg_indices.append(g_ptr)
                    while (g_ptr < len(gold_ids)) and (
                            (i_ptr == len(input_ids) - 1) or gold_ids[g_ptr] != input_ids[i_ptr + 1]):
                        g_ptr += 1
                    i_ptr += 1
                else:
                    if gold_ids[g_ptr] == 50265:
                        g_arg_indices.append(g_ptr)
                    i_ptr += 1
                    g_ptr += 1
            length = min(len(o_arg_indices), len(g_arg_indices))
            o_arg_indices = o_arg_indices[:length]
            g_arg_indices = g_arg_indices[:length]
            for o_arg_idx, g_arg_idx in zip(o_arg_indices, g_arg_indices):
                if o_arg_idx == -1:  # only consider non-empty arguments in the output
                    continue
                all_logits.append(logits[o_arg_idx])
                all_labels.append(label[g_arg_idx])

    all_logits = torch.stack(all_logits).cuda()
    all_labels = torch.stack(all_labels).cuda()

    temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))
    best_ece = float("inf")
    best_T = -1
    for T in temp_values:
        ece = ece_criterion(all_logits / T, all_labels)
        if ece < best_ece:
            print({"T": T, "ece": ece.item()})
            best_ece = ece
            best_T = T
    print("Best temperature: " + str(best_T))
    np.save("best_temp.npy", best_T)


class Evaluate(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(Evaluate, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece