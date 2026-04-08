import logging
import os
import sys
from datetime import timedelta
from typing import Dict, List, Tuple

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

_POSSIBLE_QWEN_UTIL_ROOTS = (
    "Qwen2.5-VL",
    "Qwen2-VL",
)
for _root in _POSSIBLE_QWEN_UTIL_ROOTS:
    if os.path.isdir(_root):
        sys.path.append(_root)
        utils_path = os.path.join(_root, "qwen-vl-utils", "src")
        if os.path.isdir(utils_path):
            sys.path.append(utils_path)

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "qwen_vl_utils is required for qwen25vl. Ensure the Qwen2.5-VL utilities "
        "are available in the repo or on PYTHONPATH."
    ) from exc

eval_logger = logging.getLogger("eval_logger")

DEFAULT_GEN_KWARGS: Dict[str, float | int | bool] = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
}

ALLOWED_GEN_KWARGS = set(DEFAULT_GEN_KWARGS.keys())


@register_model("qwen25vl")
class Qwen25VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if device_map == "auto":
            model_kwargs["device_map"] = device_map

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.path,
            **model_kwargs,
        ).eval()
        if device_map != "auto":
            self._model.to(torch.device(device))

        self._processor = AutoProcessor.from_pretrained(
            self.path, trust_remote_code=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=True
        )

        batch_size = int(batch_size)
        assert batch_size == 1, (
            f"Batch size should be 1 for Qwen25VL, but got {batch_size}."
        )
        self.batch_size_per_gpu = batch_size

        self._config = None

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in {
                DistributedType.DEEPSPEED,
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            }:
                raise ValueError(
                    "Unsupported distributed type provided. Only FSDP, Deepspeed, and DDP are supported."
                )
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                eval_logger.info(
                    "Detected DistributedType.DEEPSPEED. Ensure `accelerate config` uses zero stage 0 for evaluation."
                )

            if accelerator.distributed_type in {
                DistributedType.FSDP,
                DistributedType.DEEPSPEED,
            }:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(
                    self._model, evaluation_mode=True
                )

            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            eval_logger.info(
                f"Using {accelerator.num_processes} devices with data parallelism"
            )
        else:
            if device_map == "auto":
                eval_logger.info("Using device_map='auto' for Qwen2.5-VL inference")
                self._device = torch.device(device)
            else:
                self._device = torch.device(device)
                self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.device_map = device_map

        self.modality = modality
        self.max_frames_num = max_frames_num

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if visuals == [None]:
                visuals = []
            if self.modality == "image":
                raise NotImplementedError(
                    "Image inference for Qwen25VL is not supported yet."
                )
            if self.modality == "video":
                if not visuals:
                    raise ValueError("No video inputs found for Qwen25VL request.")
                if len(visuals) != 1:
                    raise AssertionError(
                        f"Only one video is supported per request, but got {len(visuals)} videos."
                    )
                video_path = visuals[0]
                video_entry = {
                    "type": "video",
                    "video": f"{video_path}",
                }
                if self.max_frames_num:
                    video_entry["nframes"] = self.max_frames_num

                messages = []
                system_prompt = getattr(self, "system_prompt", "")
                if system_prompt:
                    messages.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": f"{system_prompt}"}],
                        }
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            video_entry,
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                )
                if "until" in gen_kwargs:
                    gen_kwargs.pop("until")

                for key, value in DEFAULT_GEN_KWARGS.items():
                    if key not in gen_kwargs:
                        gen_kwargs[key] = value

                drop_keys = [key for key in gen_kwargs if key not in ALLOWED_GEN_KWARGS]
                for key in drop_keys:
                    gen_kwargs.pop(key)

                chat_prompt = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                processor_kwargs = {
                    "text": [chat_prompt],
                    "return_tensors": "pt",
                    "padding": True,
                }
                if image_inputs is not None and len(image_inputs) > 0:
                    processor_kwargs["images"] = image_inputs
                if video_inputs is not None and len(video_inputs) > 0:
                    processor_kwargs["videos"] = video_inputs

                model_inputs = self._processor(**processor_kwargs)
                if "input_ids" not in model_inputs:
                    raise ValueError(
                        "Processor did not return input_ids for Qwen25VL request."
                    )
                input_length = model_inputs["input_ids"].shape[-1]

                model_dtype = next(self.model.parameters()).dtype
                for key, value in model_inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.is_floating_point():
                            model_inputs[key] = value.to(self.device, dtype=model_dtype)
                        else:
                            model_inputs[key] = value.to(self.device)

                with torch.inference_mode():
                    generated = self.model.generate(**model_inputs, **gen_kwargs)

                new_tokens = generated[:, input_length:]
                output_text = self._tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )[0].strip()
            else:
                raise NotImplementedError
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for Qwen25VL.")
