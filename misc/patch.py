from typing import List, Optional, Union

import diffusers
import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput


class StableDiffusionPipeline(diffusers.StableDiffusionPipeline):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "noise_pred"
    ]


class StableDiffusionXLPipeline(diffusers.StableDiffusionXLPipeline):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
        "noise_pred",
    ]


class SetSampleWiseTimestepsMixin:
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        """Accept data-dependent scheduling."""
        if isinstance(timesteps, torch.Tensor) and len(timesteps.shape) > 1:
            # Sanity checks
            if len(timesteps.shape) != 2:
                raise ValueError("`timesteps` must be a 2D tensor for data-dependent scheduling.")
            if timesteps.dtype != torch.long:
                raise ValueError("`timesteps` must be a tensor of dtype `torch.long`.")
            if not timesteps.lt(self.config.num_train_timesteps).all():
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )
            if not timesteps.ge(0).all():
                raise ValueError("`timesteps` must be non-negative.")
            if not (timesteps[:-1] > timesteps[1:]).all():
                raise ValueError("`custom_timesteps` must be in descending order.")

            # The initialization of some constants might depend on the number
            # shape of the `timesteps` tensor, so we need to call the base
            # class implementation first.
            super().set_timesteps(
                num_inference_steps=num_inference_steps, device=device, timesteps=timesteps[:, 0])

            # Overwrite the `timesteps` attribute so that pipeline can access.
            # The correct `timesteps` tensor for each sample will be set
            # dynamically in the `step` method.
            self.timesteps = timesteps.to(device)
        else:
            # Fallback to the base class implementation
            super().set_timesteps(
                num_inference_steps=num_inference_steps, device=device, timesteps=timesteps)


class StepWithSampleWiseTimestepsMixin:
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        **kwargs
    ):
        if not isinstance(timestep, (int, torch.Tensor)):
            raise ValueError("`timestep` must be an integer or a tensor.")
        if isinstance(timestep, torch.Tensor) and len(timestep.shape) not in [0, 1]:
            raise ValueError("`timestep` must be a scalar or a 1D tensor.")
        if isinstance(timestep, int) or len(timestep.shape) == 0:
            # Single timestep, fallback to the base class implementation
            return super().step(model_output, timestep, sample, **kwargs)
        if timestep.shape[0] != sample.shape[0]:
            raise ValueError("The batch size of `timestep` and `sample` must be the same. {timestep.shape[0]} != {sample.shape[0]}")

        return_dict = kwargs.get("return_dict", True)
        kwargs["return_dict"] = False

        B = timestep.shape[0]
        # Backup the timesteps
        timesteps = self.timesteps
        latent_list = []
        for i in range(B):
            # Set the state for the i-th sample
            self.timesteps = timesteps[:, i]
            # Step for the i-th sample
            latent = super().step(
                model_output=model_output[i],
                timestep=timestep[i],
                sample=sample[i],
                **kwargs)[0]
            latent_list.append(latent)
        # Restore the timesteps
        self.timesteps = timesteps

        prev_sample = torch.stack(latent_list, dim=0)
        if not return_dict:
            return (prev_sample,)
        else:
            return SchedulerOutput(prev_sample=prev_sample)


class DDIMScheduler(
    SetSampleWiseTimestepsMixin,
    StepWithSampleWiseTimestepsMixin,
    diffusers.DDIMScheduler,
):
    pass


class DDPMScheduler(
    SetSampleWiseTimestepsMixin,
    StepWithSampleWiseTimestepsMixin,
    diffusers.DDPMScheduler,
):
    pass


class DPMSolverMultistepScheduler(
    SetSampleWiseTimestepsMixin,
    diffusers.DPMSolverMultistepScheduler,
):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        """Accept data-dependent scheduling."""
        super().set_timesteps(
            num_inference_steps=num_inference_steps, device=device, timesteps=timesteps)
        # Initialize the state for each sample
        if isinstance(timesteps, torch.Tensor) and len(timesteps.shape) > 1:
            self.model_outputs_list = []
            self.lower_order_nums_list = []
            self._step_index_list = []
            self._begin_index_list = []

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        **kwargs
    ):
        if not isinstance(timestep, (int, torch.Tensor)):
            raise ValueError("`timestep` must be an integer or a tensor.")
        if isinstance(timestep, torch.Tensor) and len(timestep.shape) not in [0, 1]:
            raise ValueError("`timestep` must be a scalar or a 1D tensor.")
        if isinstance(timestep, int) or len(timestep.shape) == 0:
            # Single timestep, fallback to the base class implementation
            return super().step(model_output, timestep, sample, **kwargs)

        return_dict = kwargs.get("return_dict", True)
        kwargs["return_dict"] = False

        B = timestep.shape[0]
        if B != sample.shape[0]:
            raise ValueError(
                f"The batch size of `timestep` and `sample` must be the same. "
                f"{B} != {sample.shape[0]}")

        # Backup the timesteps
        timesteps = self.timesteps
        latent_list = []
        for i in range(B):
            if i >= len(self.model_outputs_list):
                self.model_outputs_list.append([None] * self.config.solver_order)
                self.lower_order_nums_list.append(0)
                self._step_index_list.append(None)
                self._begin_index_list.append(None)
            # Set the state for the i-th sample
            self.timesteps = timesteps[:, i]
            self.model_outputs = self.model_outputs_list[i]
            self.lower_order_nums = self.lower_order_nums_list[i]
            self._step_index = self._step_index_list[i]
            self._begin_index = self._begin_index_list[i]
            # Step for the i-th sample
            latent = super().step(
                model_output=model_output[i],
                timestep=timestep[i],
                sample=sample[i],
                **kwargs)[0]
            latent_list.append(latent)
            # Store the state for the i-th sample
            self.lower_order_nums_list[i] = self.lower_order_nums
            self._step_index_list[i] = self._step_index
            self._begin_index_list[i] = self._begin_index
        # Restore the timesteps
        self.timesteps = timesteps

        prev_sample = torch.stack(latent_list, dim=0)
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
