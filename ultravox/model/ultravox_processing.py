from typing import Optional, Union

import numpy as np
import torch
import transformers

from .ultravox_adapter import UltravoxAdapter

# TODO: update the comments to reflect the actual implementation
class UltravoxProcessor(transformers.ProcessorMixin):
    """
    Constructs an Ultravox processor which wraps an audio processor and a tokenizer into a single processor.

    Args:
        audio_processor: The audio processor for the audio encoder.
        tokenizer: The tokenizer for the language model.
    """

    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = (
        "Wav2Vec2Processor",
        "SeamlessM4TFeatureExtractor",
        "WhisperProcessor",
    )
    tokenizer_class = (
        "PreTrainedTokenizer",
        "PreTrainedTokenizerFast",
    )

    tokenizer: transformers.PreTrainedTokenizerBase
    audio_processor: transformers.ProcessorMixin

    def __init__(
        self,
        audio_processor=None,
        tokenizer=None,
        audio_padding: str = "longest",
        encoder_ds_factor: int = 320,
        adapter: UltravoxAdapter = None,
        audio_placeholder: str = "<|audio|>",
    ):
        """
        Args:
            audio_processor: The audio processor for the audio encoder.
            tokenizer: The tokenizer for the language model.
            audio_padding: The padding strategy for the audio encoder.
            encoder_ds_factor: The downsample factor of the audio encoder.
            stack_factor: The factor by which the audio encoder output is stacked in the multimodal projector.
            audio_placeholder: The placeholder for the audio in the text.
        """
        self.audio_padding = audio_padding
        self.encoder_ds_factor = encoder_ds_factor
        self.adapter = adapter
        self.audio_placeholder = audio_placeholder
        self.audio_token_replacement = tokenizer.eos_token
        assert (
            self.audio_token_replacement is not None
        ), "The tokenizer has no EOS token. Cannot recover."
        super().__init__(audio_processor=audio_processor, tokenizer=tokenizer)

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transcript: Optional[str] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[
            Union[str, transformers.TensorType]
        ] = transformers.TensorType.PYTORCH,
        **kwargs,
    ) -> transformers.BatchFeature:
        """
        Main method to prepare for the model one text sequence and audio. This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio`, `sampling_rate` and `kwargs` arguments to
        audio processor's [`~Wav2Vec2Processor.__call__`] if `audio` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence to be encoded. Sequence can be a string or (pretokenized string).
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio to be prepared. Audio can be NumPy array or PyTorch tensor. In case of a
                NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T the
                sample length of the audio.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the input audio. We expect 16kHz audio. Don't change this value unless you know what
                you are doing.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_values** -- Processed audio values to be fed to a model. Returned when `audio` is not `None`.
            - **audio_token_len** -- Predicted number of audio frames: this value is guaranteed to be a close upper bound.
              Returned when `audio` is not `None`.
            - **audio_token_start_idx** -- The index in the tokenized text where the audio starts. Returned when `audio` is not `None`.
        """
        # TODO: Add support for multiple audio and text inputs.
        data = {}
        audio_token_len = 0
        if audio is not None and len(audio) > 0:
            if not self.adapter:
                raise ValueError("Adapter must be provided for determing audio_token_len.")

            if self.audio_padding == "max_length":
                # 30 seconds is the expected length for Whisper
                assert sampling_rate is not None, "Sampling rate must be provided."
                audio_len = 30 * sampling_rate
            else:
                audio_len = audio.shape[-1]
            
            # num_encoder_frames is needed for the Stacking adapter to determine audio_token_len, both in training and inference.
            # It's guaranteed that the number of frames is less than or equal to this amount.
            # For Whisper this is exact AFAICT, but for Wav2Vec2 it's an upper bound.
            # Currently, StackAudioFrames makes sure an over-estimation won't cause issues by padding the audio embeddings.
            num_encoder_frames = int(round(audio_len / self.encoder_ds_factor + 1e-4))
            # num_text_tokens is needed for the CFormer adapter to determine audio_token_len in training mode.
            # In inference mode, the inferred transcript length during forward pass is used to determine audio_token_len.
            if transcript:
                num_text_tokens = len(self.tokenizer.encode(transcript, add_special_tokens=False))
            else:
                num_text_tokens = 0
            # compute the audio_token_len based on the model's adapter
            audio_token_len = self.adapter.get_audio_token_len(num_encoder_frames, num_text_tokens)
            data["audio_token_len"] = [audio_token_len]

            # Main audio processing. The processor is model-specific.
            x = self.audio_processor(
                audio,
                sampling_rate=sampling_rate,
                padding="longest",
                max_length=audio_len,
                **kwargs,
            )
            if "input_features" in x:
                data["audio_values"] = x.input_features
            else:
                data["audio_values"] = x.input_values

        if text is not None:
            assert isinstance(
                text, str
            ), "Text must be a string. Batch mode not supported yet."
            if self.audio_placeholder in text:
                if "audio_token_len" not in data:
                    raise ValueError(
                        f"audio must be provided when using audio placeholder ({self.audio_placeholder}) in text."
                    )

                start_idx = len(
                    self.tokenizer.encode(
                        text[: text.index(self.audio_placeholder)],
                        add_special_tokens=False,
                    )
                )
                data["audio_token_start_idx"] = [start_idx]

                # Replace the audio placeholder with the audio token.
                #   e.g. "Transcribe\n<|audio|>" -> "Transcribe </s></s></s></s></s></s></s></s>"
                #        where the number of </s> is the number of audio frames.
                text = text.replace(
                    self.audio_placeholder,
                    self.audio_token_replacement * audio_token_len,
                )

            # Special tokens like BOS should already have been added by the caller.
            data.update(self.tokenizer([text], add_special_tokens=False, **kwargs))

        return transformers.BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(set(tokenizer_input_names + audio_processor_input_names))
