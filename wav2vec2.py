from __future__ import print_function 
from __future__ import division

import logging
from typing import List, Tuple, Union

import torch
import torch.optim as optim
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class W2VModelForClassification():
    def __init__(self,logger: Union[logging.Logger, None] = None) -> None:
        super().__init__(logger=logger)
        self.feature_extract = False #  When False, we finetune the whole model, when True we only update the classifier

    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            model.freeze_base_model()

    def observe_parameter(self, model):
        params_to_update = model.parameters()
        # print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        return params_to_update

    def build_model(self, num_classes, pretrained_model) -> torch.nn.Module:
        num_classes = num_classes
        config = Wav2Vec2Config.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            use_weighted_layer_sum =True,
            vocab_size=100
        )
        setattr(config, 'pooling_mode', 'mean')
        model = Wav2Vec2.from_pretrained(pretrained_model, config=config)
        model.set_feature_extractor(pretrained_model)
        return model

class Wav2Vec2(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def set_feature_extractor(self, name):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)

    def pad(self, input_values):
        feat = self.feature_extractor.pad([self.feature_extractor(input_value, sampling_rate=16000, padding=False, max_length=320_000, truncation=True, return_attention_mask=True, return_tensors="pt") for input_value in input_values]).to(self.device)
        input_values = feat['input_values'][:, 0, :]
        attention_mask = feat['attention_mask'][:, 0, :]
        return input_values, attention_mask

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        

        input_values, attention_mask = self.pad(input_values)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        return self.classifier(pooled_output)
