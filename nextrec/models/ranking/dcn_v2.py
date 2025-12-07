"""
Date: create on 09/11/2025
"""

import torch
import torch.nn as nn


class CrossNetV2(nn.Module):
    """Vector-wise cross network proposed in DCN V2 (Wang et al., 2021)."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_dim, input_dim, bias=False)
                for _ in range(num_layers)
            ]
        )
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)]
        )

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x = x0 * self.w[i](x) + self.b[i] + x
        return x


class CrossNetMix(nn.Module):
    """Mixture of low-rank cross experts from DCN V2 (Wang et al., 2021)."""

    def __init__(self, input_dim, num_layers=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        # U: (input_dim, low_rank)
        self.u_list = torch.nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(num_experts, input_dim, low_rank)
                    )
                )
                for i in range(self.num_layers)
            ]
        )
        # V: (input_dim, low_rank)
        self.v_list = torch.nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(num_experts, input_dim, low_rank)
                    )
                )
                for i in range(self.num_layers)
            ]
        )
        # C: (low_rank, low_rank)
        self.c_list = torch.nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))
                )
                for i in range(self.num_layers)
            ]
        )
        self.gating = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=False) for i in range(self.num_experts)]
        )

        self.bias = torch.nn.ParameterList(
            [
                nn.Parameter(nn.init.zeros_(torch.empty(input_dim, 1)))
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x):
        x_0 = x.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(
                    self.v_list[i][expert_id].t(), x_l
                )  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.c_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(
                    self.u_list[i][expert_id], v_x
                )  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(
                output_of_experts, 2
            )  # (bs, in_features, num_experts)
            gating_score_experts = torch.stack(
                gating_score_experts, 1
            )  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l
