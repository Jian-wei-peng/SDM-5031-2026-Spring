from dataclasses import dataclass

import torch

from TSProblemDef import augment_xy_data_by_8_fold, get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class TSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.problems = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.edge_weight_type = None
        self.original_node_xy_lib = None

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.problems = get_random_problems(batch_size, self.problem_size)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
            else:
                raise NotImplementedError

        device = self.problems.device
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(
            self.batch_size, self.pomo_size
        )

    def reset(self):
        device = self.problems.device
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0),
            dtype=torch.long,
            device=device,
        )

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, self.problem_size),
            device=device,
        )

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, lib_mode: bool = False):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')

        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance(lib_mode=lib_mode)
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self, lib_mode: bool = False):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(
            self.batch_size, -1, self.problem_size, 2
        )

        if lib_mode and self.original_node_xy_lib is not None:
            base = self.original_node_xy_lib
            if base.dim() == 3 and base.size(0) == 1 and self.batch_size != 1:
                base = base.expand(self.batch_size, -1, -1)
            seq_expanded = base[:, None, :, :].expand(
                self.batch_size, self.pomo_size, self.problem_size, 2
            )
        else:
            seq_expanded = self.problems[:, None, :, :].expand(
                self.batch_size, self.pomo_size, self.problem_size, 2
            )

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths_raw = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        if lib_mode:
            ewt = self.edge_weight_type or 'EUC_2D'
            if ewt == 'CEIL_2D':
                segment_lengths = torch.ceil(segment_lengths_raw)
            elif ewt == 'EUC_2D':
                segment_lengths = torch.floor(segment_lengths_raw + 0.5)
            else:
                segment_lengths = segment_lengths_raw
        else:
            segment_lengths = segment_lengths_raw

        travel_distances = segment_lengths.sum(2)
        return travel_distances
