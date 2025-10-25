import logging
import os
import sys
from typing import List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from transformers import AutoProcessor, AutoTokenizer

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


@register_model("qwen25vl")
class Qwen25VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: str = "1",
        max_frames_num: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self._model = LLM(
            self.path,
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLELISM", 1)),
            max_model_len=65536,
            rope_scaling={
                "type": "mrope",
                "rope_type": "mrope",
                "mrope_section": [16, 24, 24],
            },
        )
        self._processor = AutoProcessor.from_pretrained(self.path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=True
        )

        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

        batch_size = int(batch_size)
        assert batch_size == 1, (
            f"Batch size should be 1 for Qwen25VL, but got {batch_size}."
        )
        self.batch_size_per_gpu = batch_size

        self._config = None

        self._device = device
        self._rank = 0
        self._world_size = 1

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
            if self.modality == "image":
                raise NotImplementedError(
                    "Image inference for Qwen25VL is not supported yet."
                )
            if self.modality == "video":
                if not visuals:
                    raise ValueError("No video inputs found for Qwen25VL request.")
                # Qwen2.5-VL supports streaming multiple videos in a single turn, so
                # we pack every available clip as a dedicated video content block.
                video_contents = []
                for video_path in visuals:
                    video_entry = {
                        "type": "video",
                        "video": f"{video_path}",
                    }
                    if self.max_frames_num:
                        video_entry["nframes"] = self.max_frames_num
                    video_contents.append(video_entry)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            *video_contents,
                            {"type": "text", "text": f"{contexts}"},
                        ],
                    }
                ]
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                _, video_inputs = process_vision_info(messages)
                generated_ids = self._model.generate(
                    {
                        "prompt": text,
                        "multi_modal_data": {"video": video_inputs},
                    },
                    sampling_params=self.sampling_params,
                )
                output_text = generated_ids[0].outputs[0].text
            else:
                raise NotImplementedError
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for Qwen25VL.")
