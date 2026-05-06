# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ascend NPU adaptation patch for Ernie MTP speculative decoding.

Monkey-patches ErnieMultiTokenPredictorLayer to ensure compatibility with
Ascend NPU when running speculative decoding via --speculative-config
with ernie_mtp method.
"""

import torch
import vllm
from vllm.model_executor.models.ernie_mtp import (
    ErnieMultiTokenPredictorLayer,
)


class AscendErnieMultiTokenPredictorLayer(ErnieMultiTokenPredictorLayer):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # Use torch.where for NPU-compatible masked fill (safer than
        # in-place indexing on Ascend).
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)

        inputs_embeds = self.mtp_emb_norm(inputs_embeds)
        previous_hidden_states = self.mtp_hidden_norm(previous_hidden_states)

        hidden_states = self.mtp_linear_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states

        return hidden_states


# Apply monkey-patches for Ascend NPU compatibility
vllm.model_executor.models.ernie_mtp.ErnieMultiTokenPredictorLayer = AscendErnieMultiTokenPredictorLayer
