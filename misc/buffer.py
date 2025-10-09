import os
from collections import namedtuple
from typing import Any, Dict, List, Union

import torch
from accelerate import Accelerator
from diffusers.callbacks import PipelineCallback


class ReplayBuffer(PipelineCallback):
    def __init__(
        self,
        size: int,
        cpu_offload: bool = True,
        time_key: str = "t",
    ):
        self.size = size
        self.cpu_offload = cpu_offload
        self.time_key = time_key

        self.is_time_dependents: Dict[str, bool] = dict()
        # Replay buffer.
        self.buffers: Dict[str, torch.Tensor] = dict()
        # Temporary storage for the current trajectory.
        self.staging: Dict[str, Union[List, torch.Tensor]] = dict()

    def push(self, key: str, data: torch.Tensor, is_time_dependent: bool = False):
        data = data.detach().clone()
        if self.cpu_offload:
            data = data.cpu()
        # Initialize staging.
        if key not in self.staging:
            if is_time_dependent:
                self.staging[key] = []
            else:
                self.staging[key] = None
        # Check time-dependency consistency.
        if self.is_time_dependents.get(key, is_time_dependent) != is_time_dependent:
            raise ValueError(
                f"Buffer {key} is initialized with time_dependent = {self.is_time_dependents[key]} "
                f"but received time_dependent = {is_time_dependent}")
        self.is_time_dependents[key] = is_time_dependent
        # Append to staging.
        if is_time_dependent:
            self.staging[key].append(data)
        else:
            self.staging[key] = data

    def commit(self):
        T = len(self.staging[self.time_key])
        B = None
        # Merge time-dependent stagings.
        for key in self.staging.keys():
            if key != self.time_key:
                if self.is_time_dependents[key]:
                    self.staging[key] = torch.stack(self.staging[key], dim=1)[:, :T]
                    if self.staging[key].size(1) != T:
                        raise ValueError(
                            f"Time dimension mismatch for key {key}: {self.staging[key].size(1)} != {T}")
                if B is not None and B != self.staging[key].size(0):
                    raise ValueError(
                        f"Batch size mismatch for key {key}: {B} != {self.staging[key].size(0)}")
                B = self.staging[key].size(0)
        self.staging[self.time_key] = torch.stack(self.staging[self.time_key])
        if len(self.staging[self.time_key].shape) == 1:
            assert self.staging[self.time_key][:-1].greater(self.staging[self.time_key][1:]).all().item(), f"Sanity check failed for times: {self.staging[self.time_key]}"
            # Expand times to batch dimension.
            self.staging[self.time_key] = self.staging[self.time_key].unsqueeze(0).expand(B, T)
        else:
            assert self.staging[self.time_key].shape == (T, B), f"Shape mismatch for times: {self.staging[self.time_key].shape} != ({T}, {B})"
            assert self.staging[self.time_key][:-1].greater(self.staging[self.time_key][1:]).all().item(), f"Sanity check failed for times: {self.staging[self.time_key]}"
            # Transpose times to batch dimension.
            self.staging[self.time_key] = self.staging[self.time_key].t()

        # Gather and store buffers.
        for key in self.staging.keys():
            if key not in self.buffers:
                self.buffers[key] = self.staging[key]
            else:
                self.buffers[key] = torch.cat([self.buffers[key], self.staging[key]], dim=0)

        # Clear staging.
        self.staging = dict()

        # Trim buffers.
        self.trim(self.size)

    def trim(self, size: int):
        for key in self.buffers:
            self.buffers[key] = self.buffers[key][-size:]

    def sample(self, batch_size: int, device: torch.device = None):
        # Sanity check.
        N = None
        T = None
        for key in self.buffers:
            if N is not None and N != len(self.buffers[key]):
                raise ValueError(f"Buffer size mismatch for key {key}: {N} != {self.buffers[key].size(0)}")
            N = self.buffers[key].size(0)
            if self.is_time_dependents[key]:
                if T is not None and T != self.buffers[key].size(1):
                    raise ValueError(f"Time dimension mismatch for key {key}: {T} != {self.buffers[key].size(1)}")
                T = self.buffers[key].size(1)

        # Sample indices.
        if batch_size <= N:
            n_indices = torch.randperm(N)[:batch_size]
        else:
            n_indices = torch.randint(0, N, (batch_size,))
        t_indices = torch.randint(0, T, (batch_size,))

        # Gather samples.
        dict_output = dict()
        for key in self.buffers:
            if self.is_time_dependents[key]:
                dict_output[key] = self.buffers[key][n_indices, t_indices].clone()
            else:
                dict_output[key] = self.buffers[key][n_indices].clone()
            if device is not None:
                dict_output[key] = dict_output[key].to(device)

        # Named tuple output.
        names = list(dict_output.keys())
        NamedOutput = namedtuple("NamedOutput", names)
        buffer_output = NamedOutput(**dict_output)

        return buffer_output

    def __len__(self):
        if self.buffers:
            size = len(next(iter(self.buffers.values())))
        else:
            size = 0
        return size

    def callback_fn(
        self,
        pipeline,
        step_index,
        timesteps,
        callback_kwargs
    ) -> Dict[str, Any]:
        """Override the callback function to store the trajectory."""
        self.push(self.time_key, timesteps, is_time_dependent=True)
        self.push("xt", callback_kwargs["latents"], is_time_dependent=True)
        self.push("eps", callback_kwargs["noise_pred"], is_time_dependent=True)

        # return empty dictionary to avoid modifying callback_kwargs
        return {}

    @property
    def tensor_inputs(self) -> List[str]:
        """Override the tensor_inputs. to include latents and noise_pred."""
        return ["latents", "noise_pred"]

    def save(self, path: str, accelerator: Accelerator = None):
        if os.path.isdir(path):
            path = os.path.join(path, "replay_buffer.pth")

        if accelerator is not None:
            buffers = dict()
            for key in self.buffers:
                buffers = accelerator.gather(self.buffers[key].cuda()).cpu()
        else:
            buffers = self.buffers

        if accelerator is None or accelerator.is_main_process:
            torch.save({
                "version": "0.0.1",
                "buffers": buffers,
                "is_time_dependents": self.is_time_dependents,
                "size": self.size,
                "cpu_offload": self.cpu_offload,
                "time_key": self.time_key,
            }, path)

    @classmethod
    def load(cls, path: str, accelerator: Accelerator = None) -> "ReplayBuffer":
        if os.path.isdir(path):
            path = os.path.join(path, "replay_buffer.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        state = torch.load(path)

        if accelerator is not None:
            buffers = dict()
            for key in state["buffers"]:
                split_size = state["buffers"][key].size(0) // accelerator.num_processes
                splits = state["buffers"][key].split(split_size)
                buffers[key] = splits[accelerator.process_index]
                if not state["cpu_offload"]:
                    buffers[key] = buffers[key].cuda()
            state["buffers"] = buffers

        replay_buffer = cls(
            size=state["size"],
            cpu_offload=state["cpu_offload"],
            time_key=state["time_key"])
        replay_buffer.is_time_dependents = state["is_time_dependents"]
        replay_buffer.buffers = state["buffers"]

        return replay_buffer
