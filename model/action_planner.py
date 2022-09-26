import abc
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from dataset.ns_ap import GOAL_VOCAB, ACTION_VOCAB


class ActionPlanInference:
    @abc.abstractmethod
    def get_action_sequence(self, goal, scene_state):
        '''
        Abstract class for the implementation of instruction -> symbolic program inference
        Expected input: dict: {'instruction': [...], ...}
        Expected output: list: [symbolic_function1, ...] 
        '''
        pass

class ActionGTPlanner(ActionPlanInference):
    def get_action_sequence(self, goal, scene_state):
        task = goal['task']
        target = goal['target']
        action_list = []
        subseq = self._get_subsequence(goal, target[0], scene_state)
        action_list = action_list + subseq
        return action_list

    def _get_subsequence(self, goal, target, scene_state):
        if goal['task'] == 'measure_weight' or goal['task'] == 'pick_up':
            target_idx = target[0]
            target_state = scene_state[target_idx]
            in_hand = None
            for idx, obj in enumerate(scene_state):
                if obj['in_hand']:
                    in_hand = idx
            if in_hand is not None:
                if in_hand != target_idx:
                    if scene_state[in_hand]['raised']:
                        action_list = [
                            ('put_down', in_hand),
                            ('release', None),
                            ('move', target_idx),
                            ('approach_grasp', target_idx),
                            ('grasp', target_idx),
                            ('pick_up', target_idx)
                        ]
                    else:
                        action_list = [
                            ('release', None),
                            ('move', target_idx),
                            ('approach_grasp', target_idx),
                            ('grasp', target_idx),
                            ('pick_up', target_idx)
                        ]
                else:
                    if target_state['raised']:
                        action_list = []
                    else:
                        action_list = [('pick_up', target_idx)]
            elif target_state['approached']:
                action_list = [
                    ('grasp', target_idx),
                    ('pick_up', target_idx)
                ]
            elif target_state['gripper_over']:
                action_list = [
                    ('approach_grasp', target_idx),
                    ('grasp', target_idx),
                    ('pick_up', target_idx)
                ]
            else:
                action_list = [
                    ('move', target_idx),
                    ('approach_grasp', target_idx),
                    ('grasp', target_idx),
                    ('pick_up', target_idx)
                ]
            return action_list
        elif goal['task'] == 'stack':
            in_hand = None
            for idx, obj in enumerate(scene_state):
                if obj['in_hand']:
                    in_hand = idx
            action_list = []
            for i in reversed(range(len(target) - 1)):
                top = target[i]
                bottom = target[i + 1]
                pos_top = scene_state[top]['pos']
                pos_bot = scene_state[bottom]['pos']
                if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) < 0.1:
                    if (pos_top[2] - pos_bot[2]) > 0.001:
                        if in_hand is None:
                            continue
                        else:
                            if in_hand != top:
                                continue
                            top_obj_bot = np.amin(scene_state[top]['bbox'][:, 2])
                            bot_obj_top = np.amax(scene_state[bottom]['bbox'][:, 2])
                            # if scene_state[top]['raised']:
                            if top_obj_bot - bot_obj_top > 0.05:
                                action_list = [
                                    ('put_down', in_hand),
                                    ('release', None)
                                ]
                            else:
                                action_list = [
                                    ('release', None)
                                ]
                            return action_list
                    else:
                        return []
                else:
                    if in_hand is not None:
                        if in_hand != top:
                            if scene_state[in_hand]['raised']:
                                action_list = [
                                    ('put_down', in_hand),
                                    ('release', None),
                                    ('move', top),
                                    ('approach_grasp', top),
                                    ('grasp', top),
                                    ('pick_up', top),
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None),
                                ]
                            else:
                                action_list = [
                                    ('release', None),
                                    ('move', top),
                                    ('approach_grasp', top),
                                    ('grasp', top),
                                    ('pick_up', top),
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None),
                                ]
                        else:
                            if scene_state[top]['raised']:
                                action_list = [
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None)
                                ]
                            else:
                                action_list = [
                                    ('pick_up', top),
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None)
                                ]
                    elif scene_state[top]['approached']:
                        action_list = [
                            ('grasp', top),
                            ('pick_up', top),
                            ('move', bottom),
                            ('put_down', top),
                            ('release', None)
                        ]
                    elif scene_state[top]['gripper_over']:
                        action_list = [
                            ('approach_grasp', top),
                            ('grasp', top),
                            ('pick_up', top),
                            ('move', bottom),
                            ('put_down', top),
                            ('release', None)
                        ]
                    else:
                        action_list = [
                            ('move', top),
                            ('approach_grasp', top),
                            ('grasp', top),
                            ('pick_up', top),
                            ('move', bottom),
                            ('put_down', top),
                            ('release', None)
                        ]
                    return action_list
            return action_list
        elif 'move' in goal['task']:
            side = goal['task'].split('[')[1].strip(']')
            in_hand = None
            for idx, obj in enumerate(scene_state):
                if obj['in_hand']:
                    in_hand = idx
            action_list = []
            if in_hand is not None:
                if in_hand not in target:
                    if scene_state[in_hand]['raised']:
                        action_list += [
                            ('put_down', in_hand),
                            ('release', None)
                        ]
                    else:
                        action_list += [
                            ('release', None)
                        ]
                else:
                    new_target_list = []
                    new_target_list.append(in_hand)
                    for idx in target:
                        if idx != in_hand:
                            new_target_list.append(idx)
                    target = new_target_list
    
            for idx in target:
                y_pos = scene_state[idx]['pos'][1]
                if (side == 'right' and y_pos > 0) or (side == 'left' and y_pos < 0):
                    if idx == in_hand:
                        if scene_state[in_hand]['raised']:
                            action_list += [
                                ('put_down', in_hand),
                                ('release', None)
                            ]
                        else:
                            action_list += [
                                ('release', None)
                            ]
                    else:
                        continue
                else:
                    if idx == in_hand:
                        if scene_state[idx]['raised']:
                            action_list += [
                                ('move', side),
                                ('put_down', in_hand),
                                ('release', None)
                            ]
                        else:
                            action_list += [
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', in_hand),
                                ('release', None)
                            ]
                    else:
                        if scene_state[idx]['approached']:
                            action_list += [   
                                ('grasp', idx),
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', idx),
                                ('release', None)
                            ]
                        elif scene_state[idx]['gripper_over']:
                            action_list += [   
                                ('approach_grasp', idx),
                                ('grasp', idx),
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', idx),
                                ('release', None)
                            ]
                        else:
                            action_list += [   
                                ('move', idx),
                                ('approach_grasp', idx),
                                ('grasp', idx),
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', idx),
                                ('release', None)
                            ]
            return action_list
    
        else:
            return []


class BaselineActionTrainer(ActionPlanInference):
    def __init__(self, model_params, loss_params, optimiser_params, scheduler_params) -> None:
        self.model = BaselineActionModel(model_params)
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
        scheduler_step = 5
        if 'scheduler_step' in scheduler_params:
            scheduler_step = scheduler_params['scheduler_step']
        gamma = 0.95
        if 'gamma' in scheduler_params:
            gamma = scheduler_params['gamma']

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=scheduler_step,
            gamma=gamma
        )
        
        self.loss_function = self.loss_fn
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.token_to_action = None

    def loss_fn(self, pred, target):
        pred_action = pred[0]
        pred_target = pred[1]
        target_action = target[0]
        target_target = target[1]
        # targeted_idxs = torch.nonzero(target_target > -1).squeeze()
        # pred_target = pred_target[targeted_idxs, :]
        # target_target = target_target[targeted_idxs]
        loss_action = self.ce_loss(pred_action, target_action)
        loss_target =  self.ce_loss(pred_target, target_target)
        loss = loss_action + loss_target
        return loss

    def train_step(self, print_debug=None):
        assert self.data is not None, "Set input first"
        self.set_train()

        if self.new_epoch_trigger:
            self.lr_scheduler.step()
            self.new_epoch_trigger = False

        objs, goal_task, (action_task, action_target), length = self.data
        objs = objs.to(self.device)
        goal_task = goal_task.to(self.device)
        action_task = action_task.to(self.device)
        action_target = action_target.to(self.device)
        length = length.to(self.device)
        # target = [t.to(self.device) for t in target]

        preds = self.model(goal_task, objs, length)

        loss = self.loss_function(preds, (action_task, action_target))

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item(), None

    def validate(self, loader):
        stats = {
            'avg_acc': None,
            'action_acc': 0,
            'target_acc': 0
        }
        total_target = 0
        self.set_test()
        with torch.no_grad():
            for objs, goal_task, (action_task, action_target), length in tqdm(loader):
                objs = objs.to(self.device)
                goal_task = goal_task.to(self.device)
                action_task = action_task.to(self.device)
                action_target = action_target.to(self.device)
                length = length.to(self.device)

                pred_actions, pred_targets = self.model(goal_task, objs, length)
                # print(pred_actions.shape)
                for i in range(pred_actions.shape[0]):
                    # print(pred_actions[i, :])
                    # print(torch.max(pred_actions[i, :], 0))
                    _, pred_action_token = torch.max(pred_actions[i, :], 0)
                    correct_action = (pred_action_token == action_task[i])
                    stats['action_acc'] += int(correct_action.item())
                    # print(pred_action_token, action_task[i])
                    # print(pred_targets_token, action_target[i])
                    # exit()
                    if action_target[i] > -1:
                        _, pred_targets_token = torch.max(pred_targets[i, :], 0)
                        correct_target = (pred_targets_token == action_target[i])
                        stats['target_acc'] += int(correct_target.item())
                        total_target += 1
                    # if not correct_action:
                    #     print(pred_action_token, action_task[i])

            stats['action_acc'] /= len(loader.dataset)
            stats['target_acc'] /= total_target
            stats['avg_acc'] = (stats['action_acc'] + stats['target_acc']) / 2

        return stats

    def get_action_sequence(self, goal, scene_state):
        self.set_test()
        if self.token_to_action is None:
            self.token_to_goal = {v: k for k, v in GOAL_VOCAB.items()}
            self.token_to_action = {v: k for k, v in ACTION_VOCAB.items()}
        with torch.no_grad():
            goal_task = torch.tensor(GOAL_VOCAB[goal['task']]).unsqueeze(0).to(self.device)
            object_vecs = []
            targets = goal['target'][0]
            primary_target = None
            secondary_target = None
            if len(targets) > 0:
                primary_target = targets[0]
            if len(targets) > 1:
                secondary_target = targets[1]
            for obj in scene_state:
                object_vecs.append(
                    [
                        int(obj['in_hand']),
                        int(obj['raised']),
                        int(obj['approached']),
                        int(obj['gripper_over']),
                        int(obj['weight'] is not None),
                        0,
                        0
                    ]
                )
            if primary_target is not None:
                object_vecs[primary_target][5] = 1
            if secondary_target is not None:
                object_vecs[secondary_target][6] = 1
            for i, obj in enumerate(object_vecs):
                object_vecs[i] = torch.tensor(obj, dtype=torch.int)
            length = len(object_vecs)
            if length < 5:
                object_vecs.append(-torch.ones(7, dtype=torch.int))
            objs = torch.stack(object_vecs, dim=0).unsqueeze(0).to(self.device)
            length = [length]
            pred_actions, pred_targets = self.model(goal_task, objs, length)
            _, pred_action_token = torch.max(pred_actions[0, :], 0)
            pred_action_name = self.token_to_action[pred_action_token.item()]
            _, pred_targets_token = torch.max(pred_targets[0, :], 0)
            if pred_action_name == 'move_right':
                pred = [('move', 'right')]
            elif pred_action_name == 'move_left':
                pred = [('move', 'left')]
            else:
                pred = [(pred_action_name, pred_targets_token.item())]
        return pred

    def save_checkpoint(self, path, epoch=None, num_iter=None):
        checkpoint = {
            'state_dict': self.model.cpu().state_dict(),
            'epoch': epoch,
            'num_iter': num_iter
        }
        #TODO it saves in validation
        if self.optimiser is not None:
            checkpoint['optim_state_dict'] = self.optimiser.state_dict()
        if self.lr_scheduler is not None:
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
        return "Baseline Acion Planning Transformer"


class BaselineActionModel(nn.Module):
    def __init__(self, params={}) -> None:
        super().__init__()
        self.goal_embedding = nn.Embedding(5, 32)
        self.state_embedding = nn.Linear(7, 32)
        transformer_enc_layer = nn.TransformerEncoderLayer(
            32, 2, dim_feedforward=64, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_enc_layer, 2
        )
        self.task_predictor = nn.Linear(32, 8)
        self.sqrt_dim = torch.sqrt(torch.tensor(32))

    def forward(self, goal, scene, lengths):
        # print(goal.shape)
        goal = self.goal_embedding(goal)
        # print(goal.shape)
        # print(scene.shape)
        scene = self.state_embedding(scene.float())
        # print(scene.shape)
        transformer_input = torch.cat(
            (
                goal.unsqueeze(1),
                scene
            ), dim=1
        )
        src_key_mask = torch.zeros((scene.shape[0], scene.shape[1] + 1), dtype=torch.bool)
        # print(src_key_mask)
        # print(lengths)
        if scene.shape[0] == 1:
            if lengths[0] == 4:
                src_key_mask[0, -1] = True
        else:
            src_key_mask[torch.nonzero(lengths == 4), -1] = True
        # print(src_key_mask)
        # exit()
        src_key_mask = src_key_mask.to(transformer_input.device)
        encoded = self.transformer_encoder(transformer_input, src_key_padding_mask=src_key_mask)
        # print(encoded[:, -1, :])
        action_task = self.task_predictor(encoded[:, 0, :])
        targets = encoded[:, 1:, :]
        target_mask = ~src_key_mask[:, 1:]
        target_sum = torch.sum(target_mask.unsqueeze(2) * targets, dim=1)

        score = torch.bmm(target_sum.unsqueeze(1), targets.transpose(1, 2)).squeeze(1) / self.sqrt_dim
        score.masked_fill_(src_key_mask[:, 1:], -float('Inf'))

        return action_task, score