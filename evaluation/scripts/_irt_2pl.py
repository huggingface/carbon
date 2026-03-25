# MIT License
#
# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from the MIT-licensed py-irt trainer and AllenAI's fluid-benchmarking
# 2PL fitting script, reduced to the minimal pieces needed for this repo.

import json
from dataclasses import dataclass
from pathlib import Path

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO


@dataclass
class IrtDataset:
    item_id_to_ix: dict[str, int]
    ix_to_item_id: dict[int, str]
    subject_id_to_ix: dict[str, int]
    ix_to_subject_id: dict[int, str]
    observation_subjects: list[int]
    observation_items: list[int]
    observations: list[float]

    @classmethod
    def from_jsonlines(cls, path: str | Path) -> "IrtDataset":
        path = Path(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No JSONL rows found in {path}")

        item_ids: list[str] = []
        item_id_to_ix: dict[str, int] = {}
        subject_ids: list[str] = []
        subject_id_to_ix: dict[str, int] = {}
        observation_subjects: list[int] = []
        observation_items: list[int] = []
        observations: list[float] = []

        for row in rows:
            subject_id = row["subject_id"]
            if subject_id not in subject_id_to_ix:
                subject_id_to_ix[subject_id] = len(subject_ids)
                subject_ids.append(subject_id)
            responses = row["responses"]
            for item_id in responses:
                if item_id not in item_id_to_ix:
                    item_id_to_ix[item_id] = len(item_ids)
                    item_ids.append(item_id)

        for row in rows:
            subject_ix = subject_id_to_ix[row["subject_id"]]
            for item_id, response in row["responses"].items():
                observation_subjects.append(subject_ix)
                observation_items.append(item_id_to_ix[item_id])
                observations.append(float(response))

        return cls(
            item_id_to_ix=item_id_to_ix,
            ix_to_item_id={ix: item_id for item_id, ix in item_id_to_ix.items()},
            subject_id_to_ix=subject_id_to_ix,
            ix_to_subject_id={ix: subject_id for subject_id, ix in subject_id_to_ix.items()},
            observation_subjects=observation_subjects,
            observation_items=observation_items,
            observations=observations,
        )


class TwoParamLog:
    def __init__(self, *, priors: str, num_items: int, num_subjects: int, device: str) -> None:
        if priors not in {"vague", "hierarchical"}:
            raise ValueError("priors must be 'vague' or 'hierarchical'")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if num_items <= 0 or num_subjects <= 0:
            raise ValueError("num_items and num_subjects must be positive")
        self.priors = priors
        self.num_items = num_items
        self.num_subjects = num_subjects
        self.device = device

    def model_vague(self, subjects, items, obs):
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample(
                "theta",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )

        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample(
                "b",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.1, device=self.device),
                ),
            )
            slope = pyro.sample(
                "a",
                dist.LogNormal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.1, device=self.device),
                ),
            )

        with pyro.plate("observe_data", obs.size(0), device=self.device):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=slope[items] * (ability[subjects] - diff[items])),
                obs=obs,
            )

    def guide_vague(self, subjects, items, obs):
        m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_subjects, device=self.device))
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.empty(self.num_items, device=self.device).fill_(1.0e1),
            constraint=constraints.positive,
        )
        m_a_param = pyro.param("loc_slope", torch.zeros(self.num_items, device=self.device))
        s_a_param = pyro.param(
            "scale_slope",
            torch.empty(self.num_items, device=self.device).fill_(1.0e-6),
            constraint=constraints.positive,
        )

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))
            pyro.sample("a", dist.LogNormal(m_a_param, s_a_param))

    def model_hierarchical(self, subjects, items, obs):
        mu_b = pyro.sample(
            "mu_b",
            dist.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_b = pyro.sample(
            "u_b",
            dist.Gamma(torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)),
        )
        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)),
        )
        mu_a = pyro.sample(
            "mu_a",
            dist.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0e6, device=self.device),
            ),
        )
        u_a = pyro.sample(
            "u_a",
            dist.Gamma(torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)),
        )
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))
        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))
            slope = pyro.sample("a", dist.LogNormal(mu_a.clamp(-5, 5), (1.0 / u_a).clamp(max=2.0)))
        with pyro.plate("observe_data", obs.size(0)):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=slope[items] * (ability[subjects] - diff[items])),
                obs=obs,
            )

    def guide_hierarchical(self, subjects, items, obs):
        loc_mu_b_param = pyro.param("loc_mu_b", torch.tensor(0.0, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b",
            torch.tensor(1.0e1, device=self.device),
            constraint=constraints.positive,
        )
        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.tensor(0.0, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.tensor(1.0e1, device=self.device),
            constraint=constraints.positive,
        )
        loc_mu_a_param = pyro.param("loc_mu_a", torch.tensor(0.0, device=self.device))
        scale_mu_a_param = pyro.param(
            "scale_mu_a",
            torch.tensor(1.0e1, device=self.device),
            constraint=constraints.positive,
        )
        alpha_b_param = pyro.param(
            "alpha_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_b_param = pyro.param(
            "beta_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        alpha_theta_param = pyro.param(
            "alpha_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_theta_param = pyro.param(
            "beta_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        alpha_a_param = pyro.param(
            "alpha_a", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_a_param = pyro.param(
            "beta_a", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_subjects, device=self.device))
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        m_a_param = pyro.param("loc_slope", torch.zeros(self.num_items, device=self.device))
        s_a_param = pyro.param(
            "scale_slope",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))
        pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))
        pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))
        pyro.sample("mu_a", dist.Normal(loc_mu_a_param, scale_mu_a_param))
        pyro.sample("u_a", dist.Gamma(alpha_a_param, beta_a_param))

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))
            pyro.sample("a", dist.LogNormal(m_a_param, s_a_param))

    def get_model(self):
        if self.priors == "vague":
            return self.model_vague
        return self.model_hierarchical

    def get_guide(self):
        if self.priors == "vague":
            return self.guide_vague
        return self.guide_hierarchical

    def export(self) -> dict[str, list[float]]:
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_slope").data.exp().tolist(),
        }


class TwoParamIrtTrainer:
    def __init__(
        self,
        dataset: IrtDataset,
        *,
        priors: str = "hierarchical",
        lr: float = 0.1,
        lr_decay: float = 0.9999,
    ) -> None:
        self.dataset = dataset
        self.priors = priors
        self.lr = lr
        self.lr_decay = lr_decay
        self.irt_model: TwoParamLog | None = None
        self.best_params: dict | None = None
        self.last_params: dict | None = None

    def train(
        self,
        *,
        epochs: int,
        device: str = "cpu",
        log_every: int = 100,
    ) -> dict:
        self.irt_model = TwoParamLog(
            priors=self.priors,
            num_items=len(self.dataset.ix_to_item_id),
            num_subjects=len(self.dataset.ix_to_subject_id),
            device=device,
        )
        pyro.clear_param_store()
        pyro_model = self.irt_model.get_model()
        pyro_guide = self.irt_model.get_guide()
        scheduler = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self.lr},
                "gamma": self.lr_decay,
            }
        )
        svi = SVI(pyro_model, pyro_guide, scheduler, loss=Trace_ELBO())

        torch_device = torch.device(device)
        subjects = torch.tensor(self.dataset.observation_subjects, dtype=torch.long, device=torch_device)
        items = torch.tensor(self.dataset.observation_items, dtype=torch.long, device=torch_device)
        responses = torch.tensor(self.dataset.observations, dtype=torch.float, device=torch_device)

        _ = pyro_model(subjects, items, responses)
        _ = pyro_guide(subjects, items, responses)

        best_loss = float("inf")
        current_lr = self.lr
        for epoch in range(epochs):
            loss = float(svi.step(subjects, items, responses))
            if loss < best_loss:
                best_loss = loss
                self.best_params = self.export()
            scheduler.step()
            current_lr *= self.lr_decay
            if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0 or epoch + 1 == epochs):
                print(
                    f"epoch={epoch + 1} loss={loss:.4f} best_loss={best_loss:.4f} lr={current_lr:.6f}"
                )

        self.last_params = self.export()
        if self.best_params is None:
            self.best_params = self.last_params
        return self.best_params

    def export(self) -> dict:
        if self.irt_model is None:
            raise RuntimeError("Call train() before export().")
        result = self.irt_model.export()
        result["item_ids"] = self.dataset.ix_to_item_id
        result["subject_ids"] = self.dataset.ix_to_subject_id
        return result
