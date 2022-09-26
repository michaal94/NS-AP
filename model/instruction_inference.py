import abc
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from .rnn_blocks import BaselineLSTMDecoder, BaselineLSTMEncoder
import utils.tokenisation as token_utils


class InstructionInferenceModel:
    @abc.abstractmethod
    def get_program(self, instruction_dict):
        '''
        Abstract class for the implementation of instruction -> symbolic program inference
        Expected input: dict: {'instruction': [...], ...}
        Expected output: list: [symbolic_function1, ...] 
        '''
        pass

class InstructionGTLoader(InstructionInferenceModel):
    def get_program(self, instruction_dict):
        assert 'program' in instruction_dict, "Provide GT program"
        program_dict_list = instruction_dict['program']
        program_list = []
        for prog in program_dict_list:
            prog_name = prog['type']
            if prog['input_value'] is not None:
                prog_name = f"{prog_name}[{prog['input_value']}]"
            program_list.append(prog_name)
        return program_list


class BaselineSeq2SeqTrainer(InstructionInferenceModel):
    def __init__(self, model_params, loss_params, optimiser_params, scheduler_params) -> None:
        self.model = BaselineSeq2SeqModel(model_params)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = 0.005
        weight_decay = 0
        if 'lr' in optimiser_params:
            lr = scheduler_params['lr']
        if 'weight_decay' in optimiser_params:
            weight_decay = scheduler_params['weight_decay']
        self.optimiser = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

        self.epoch = 0
        self.new_epoch_trigger = False
        if 'starting_epoch' in scheduler_params:
            self.epoch = scheduler_params['starting_epoch']
        scheduler_step = 3
        if 'scheduler_step' in scheduler_params:
            scheduler_step = scheduler_params['scheduler_step']
        gamma = 0.9
        if 'gamma' in scheduler_params:
            gamma = scheduler_params['gamma']

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=scheduler_step,
            gamma=gamma
        )

        self.loss_function = nn.CrossEntropyLoss()

        self.program_idx_to_token = None

    def train_step(self, print_debug=None):
        assert self.data is not None, "Set input first"
        self.set_train()

        if self.new_epoch_trigger:
            self.lr_scheduler.step()
            self.new_epoch_trigger = False

        instructions, programs = self.data
        instructions = instructions.to(self.device)
        programs = programs.to(self.device)

        preds = self.model(instructions, programs)
        # Cut last one (that would be GT at size +1)
        preds = preds[:, :-1, :].transpose(1, 2)
        # Cut first one
        programs = programs[:, 1:]

        loss = self.loss_function(preds, programs)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item(), None

    def validate(self, loader):
        stats = {
            'w_previous_token_wo_null': [],
            'wo_previous_token_wo_null': []
        }
        null_token = self.model.vocab['program_token_to_idx']['<NULL>']
        start_token = self.model.vocab['program_token_to_idx']['<START>']
        end_token = self.model.vocab['program_token_to_idx']['<END>']
        with torch.no_grad():
            for instructions, programs in loader:
                instructions = instructions.to(self.device)
                programs = programs.to(self.device)

                # w prev token
                preds = self.model(instructions, programs)
                preds = preds[:, :-1, :]
                _, preds = torch.max(preds, 2)
                programs = programs[:, 1:]
                non_null_mask = programs != null_token
                non_null_lengths = non_null_mask.sum(dim=1)
                correct_tokens = preds == programs
                correct_non_null_tokens = torch.logical_and(correct_tokens, non_null_mask)
                correct_sum = correct_non_null_tokens.sum(dim=1)
                acc = correct_sum / non_null_lengths
                stats['w_previous_token_wo_null'].extend(acc.tolist())

                # wo prev token
                start_token_tensor = torch.ones(
                    (instructions.shape[0], 1), dtype=torch.long
                ) * start_token
                start_token_tensor = start_token_tensor.to(self.device)
                preds = self.model.sample(instructions, start_token_tensor, end_token)
                preds = preds[:, :programs.shape[1]]
                correct_tokens = preds == programs
                correct_non_null_tokens = torch.logical_and(correct_tokens, non_null_mask)
                correct_sum = correct_non_null_tokens.sum(dim=1)
                acc = correct_sum / non_null_lengths
                stats['wo_previous_token_wo_null'].extend(acc.tolist())

        stats['w_previous_token_wo_null'] = sum(stats['w_previous_token_wo_null']) / len(stats['w_previous_token_wo_null'])
        stats['wo_previous_token_wo_null'] = sum(stats['wo_previous_token_wo_null']) / len(stats['wo_previous_token_wo_null'])

        return stats

    def get_program(self, instruction_dict):
        if self.program_idx_to_token is None:
            self.program_idx_to_token = {v: k for k, v in self.model.vocab['program_token_to_idx'].items()}

        instruction_tokens = token_utils.tokenise(
            instruction_dict['instruction'],
            punct_to_keep=[';', ','],
            punct_to_remove=['?', '.']
        )
        instruction_encoded = token_utils.encode(
            instruction_tokens,
            self.model.vocab['instruction_token_to_idx']
        )

        instruction_tensor = torch.tensor(instruction_encoded).unsqueeze(0).to(self.device)
        start_token_tensor = torch.tensor(self.model.vocab['program_token_to_idx']['<START>']).unsqueeze(0).unsqueeze(0).to(self.device)

        preds = self.model.sample(instruction_tensor, start_token_tensor, self.model.vocab['program_token_to_idx']['<END>'])
        match_matrix = (preds == self.model.vocab['program_token_to_idx']['<END>'])
        idx = match_matrix.nonzero()
        length = idx.item()
        preds = preds[0:length]

        pred_decode = [self.program_idx_to_token[idx] for idx in preds.tolist()]
        
        return pred_decode

    def save_checkpoint(self, path, epoch=None, num_iter=None):
        checkpoint = {
            'state_dict': self.model.cpu().state_dict(),
            'epoch': epoch,
            'num_iter': num_iter
        }
        #TODO it saves in validation
        if self.model.training:
            checkpoint['optim_state_dict'] = self.optimiser.state_dict()
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, path)
        self.model = self.model.to(self.device)

    def load_checkpoint(self, path, zero_train=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optimiser is not None:
            if self.model.training and not zero_train:
                self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        if self.lr_scheduler is not None:
            if self.model.training and not zero_train:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] if not zero_train else None
        if epoch is not None:
            self.epoch = epoch
        num_iter = checkpoint['num_iter'] if not zero_train else None
        return epoch, num_iter

    def set_train(self):
        if not self.model.training:
            self.model.train()

    def set_test(self):
        if self.model.training:
            self.model.eval()

    def set_input(self, data):
        self.data = data

    def unset_input(self):
        self.data = None

    def set_epoch_number(self, epoch):
        if self.epoch != epoch:
            self.new_epoch_trigger = True
            self.epoch = epoch

    def __str__(self):
        return "Baseline Instructions Seq2Seq"


class BaselineSeq2SeqModel(nn.Module):
    def __init__(self, params={}) -> None:
        super().__init__()
        assert 'vocab_path' in params
        with open(params['vocab_path'], 'r') as f:
            self.vocab = json.load(f)
        
        params['encoder_vocab_size'] = len(self.vocab['instruction_token_to_idx'])
        params['decoder_vocab_size'] = len(self.vocab['instruction_token_to_idx'])
        params['start_token'] = self.vocab['instruction_token_to_idx']['<START>']
        params['end_token'] = self.vocab['instruction_token_to_idx']['<END>']

        self.encoder = BaselineLSTMEncoder(params)
        self.decoder = BaselineLSTMDecoder(params)

    def forward(self, inp, out):
        encoder_outputs, encoder_hidden = self.encoder(inp)
        decoder_outputs, decoder_hidden = self.decoder(out, encoder_outputs, encoder_hidden)

        return decoder_outputs

    def sample(self, inp, prog_start, end_tok):
        encoder_outputs, encoder_hidden = self.encoder(inp)
        token_pred = self.decoder.sample(
            prog_start, encoder_outputs, encoder_hidden, end_tok)

        return token_pred