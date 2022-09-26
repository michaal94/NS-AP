import os
import json
import h5py
import torch
import numpy as np
from dataset.ns_ap import GeneralNSAP
import utils.tokenisation as token_utils
import utils.utils as utils

class InstructionsNSAP(GeneralNSAP):
    def __init__(self, params={}) -> None:
        super().__init__(params)

        preprocess = False
        if 'preprocess' in params:
            preprocess = params['preprocess']
        
        if not preprocess:
            for path in self.subtask_split_paths:
                if not os.path.exists(os.path.join(path, 'instructions.h5')):
                    print("Please run with preprocess=True for the first time")
                if not os.path.exists(os.path.join(path, 'vocab.json')):
                    print("Please run with preprocess=True for the first time")
        else:
            self.preprocess_instructions()

        self.instructions = []
        self.programs = []
        self.image_idxs = []
        for path in self.subtask_split_paths:
            instructions_subtask_h5 = h5py.File(os.path.join(path, 'instructions.h5'), 'r')
            instructions_subtask = torch.tensor(
                np.asarray(
                    instructions_subtask_h5['instructions'],
                    dtype=np.int64
                ),
                dtype=torch.long
            )
            programs_subtask = torch.tensor(
                np.asarray(
                    instructions_subtask_h5['programs'],
                    dtype=np.int64
                ),
                dtype=torch.long
            )
            image_idxs_subtask = torch.tensor(
                np.asarray(
                    instructions_subtask_h5['image_idxs'],
                    dtype=np.int64
                ),
                dtype=torch.long
            )
            self.instructions.append(instructions_subtask) 
            self.programs.append(programs_subtask) 
            self.image_idxs.append(image_idxs_subtask)
        
        self.instructions = torch.cat(self.instructions, dim=0)
        self.programs = torch.cat(self.programs, dim=0)
        self.image_idxs = torch.cat(self.image_idxs)

        _, sort_indices = torch.sort(self.image_idxs)
        self.image_idxs = self.image_idxs[sort_indices]
        self.instructions = self.instructions[sort_indices, :]
        self.programs = self.programs[sort_indices, :]

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        program = self.programs[idx]

        return instruction, program

    def __len__(self):
        return len(self.instructions)

    def __str__(self):
        return "NS-AP instructions"

    def preprocess_instructions(self):
        instructions_list = []
        instructions_per_subtask = {}
        for path in self.subtask_split_paths:
            with open(os.path.join(path, 'instructions.json'), 'r') as f:
                subtask_instructions = json.load(f)['instructions']
            instructions_per_subtask[path] = subtask_instructions
            instructions_list.extend(subtask_instructions)
        # instructions_list = sorted(instructions_list, key=lambda x: x['image_index'])
        # print(instructions_list[0:5])
        save = False
        if self.split != 'train':
            if os.path.exists(os.path.join(self.subtask_split_paths[0], '../train/vocab.json')):
                with open(os.path.join(self.subtask_split_paths[0], '../train/vocab.json'), 'r') as f:
                    vocab = json.load(f)
            else:
                print("Run on train first")
                exit()
        else:
            vocab = None
            save = True
        print('Building vocabulary')
        instruction_token_to_idx = token_utils.build_vocab(
            (instr['instruction'] for instr in instructions_list),
            punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )
        # print(instruction_token_to_idx)
        all_program_strs = []
        for instr in instructions_list:
            if 'program' not in instr: continue
            program_str = token_utils.program_to_str(instr['program'])
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = token_utils.build_vocab(
            all_program_strs, delim='::'
        )
        # print(program_token_to_idx)
        if vocab is not None:
            for token in instruction_token_to_idx:
                if token not in vocab['instruction_token_to_idx']:
                    vocab[token] = len(vocab['instruction_token_to_idx'])
                    print(f"New token added: {token}")
                    save = True
            for token in program_token_to_idx:
                if token not in vocab['program_token_to_idx']:
                    vocab[token] = len(vocab['program_token_to_idx'])
                    print(f"New token added: {token}")
                    save = True
        else:
            vocab = {
                "instruction_token_to_idx": instruction_token_to_idx,
                "program_token_to_idx": program_token_to_idx
            }

        print("Saving vocab")
        if save:
            for path in self.subtask_split_paths:
                with open(os.path.join(path, '../train/vocab.json'), 'w') as f:
                    json.dump(vocab, f, indent=4)
        for path in self.subtask_split_paths:
            with open(os.path.join(path, 'vocab.json'), 'w') as f:
                json.dump(vocab, f, indent=4)

        print("Encoding data")
        instructions_encoded = {}
        programs_encoded = {}
        image_idxs = {}

        for path, instr_list in instructions_per_subtask.items():
            instructions_encoded[path] = []
            programs_encoded[path] = []
            image_idxs[path] = []
            for instr in instr_list:
                instruction = instr['instruction']

                image_idxs[path].append(instr['image_index'])
                
                instruction_tokens = token_utils.tokenise(
                    instruction,
                    punct_to_keep=[';', ','],
                    punct_to_remove=['?', '.']
                )
                instruction_encoded = token_utils.encode(
                    instruction_tokens,
                    vocab['instruction_token_to_idx']
                )
                instructions_encoded[path].append(instruction_encoded)

                program = instr['program']
                program_str = token_utils.program_to_str(program)
                program_tokens = token_utils.tokenise(program_str, delim='::')
                program_encoded = token_utils.encode(
                    program_tokens,
                    vocab['program_token_to_idx']
                )
                programs_encoded[path].append(program_encoded)

        # Pad encoded instructions and programs
        max_instruction_length = max([max(len(x) for x in instructions_encoded[path]) for path in self.subtask_split_paths])
        for path in self.subtask_split_paths:
            for ie in instructions_encoded[path]:
                while len(ie) < max_instruction_length:
                    ie.append(vocab['instruction_token_to_idx']['<NULL>'])

        max_program_length = max([max(len(x) for x in programs_encoded[path]) for path in self.subtask_split_paths])
        for path in self.subtask_split_paths:
            for pe in programs_encoded[path]:
                while len(pe) < max_program_length:
                    pe.append(vocab['program_token_to_idx']['<NULL>'])

        # Create h5 file
        print('Writing output')
        tot_number = 0
        for path in self.subtask_split_paths:
            instructions_encoded_np = np.asarray(instructions_encoded[path], dtype=np.int32)
            programs_encoded_np = np.asarray(programs_encoded[path], dtype=np.int32)
            tot_number += instructions_encoded_np.shape[0] 
            with h5py.File(os.path.join(path, 'instructions.h5'), 'w') as f:
                f.create_dataset('instructions', data=instructions_encoded_np)
                f.create_dataset('image_idxs', data=np.asarray(image_idxs[path]))
                f.create_dataset('programs', data=programs_encoded_np)
        print(f'Processed instructions: {tot_number}')