import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from .vit import VisionTransformer
from typing import List, Optional
from torch import Tensor, tensor
import torch.nn.functional as F
import math

# Our method!


class MatrixPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(prompt_param)

        # e prompt init
        # for each layer register a prompt pool/key/alpha
        # number of prompts: pool_size*p_length*len(e_layers)/100*8*5
        for e in self.e_layers:
            e_l: int = self.e_p_length
            # [pool_size, p_length, p_dim]:[100, 8, d]
            p: Parameter = tensor_prompt(
                self.e_pool_size, e_l, emb_d, ortho=True
            )
            # [pool_size, k_dim]
            k: Parameter = tensor_prompt(
                self.e_pool_size, self.key_d, ortho=True
            )
            # [pool_size, k_dim]
            a: Parameter = tensor_prompt(
                self.e_pool_size, self.key_d, ortho=True
            )
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)
            setattr(self, f"e_a_{e}", a)

        g_cls = tensor_prompt(self.g_pool_size, self.emb_d, ortho=True)
        setattr(self, "g_cls", g_cls)
        if self.ortho_mu > 0:
            print("|-" * 100 + "|")
            print("ortho penalty is on")

        self.nb_pt = int(self.e_pool_size / (self.n_tasks))
        self.task_start = int(self.task_count * self.nb_pt)
        self.task_end = int((self.task_count + 1) * self.nb_pt)

    def _init_smart(self, prompt_param) -> None:
        """initiate basic parameters: int | float | List"""
        # prompt basic param
        # number of prompt groups in the pool
        self.e_pool_size = int(prompt_param[0])
        # number of prompts in each prompt group
        self.e_p_length = int(prompt_param[1])
        self.e_layers: list[int] = [0, 1, 2, 3, 4]
        # strength of ortho penalty
        self.ortho_mu: float = prompt_param[2]

        self.g_pool_size = int(prompt_param[0])
        self.nb_pt = int(self.g_pool_size / (self.n_tasks))

    def compute_sim(self, hints, pmts):
        """
        Args:
            hints: [batch_size, nb_pmt, dim]
            pmts: [nb_pmt, pmt_len, dim]

        Returns:
            sim: [batch_size, nb_pmt]
        """
        proj_mat = torch.einsum("pij,pjk -> pik", pmts.permute(0, 2, 1), pmts)
        hint_proj = torch.einsum("bpd,pdd->bpd", hints, proj_mat)
        hint_proj = F.normalize(hint_proj, dim=-1, p=2)
        hints = F.normalize(hints, dim=-1, p=2)
        sim = torch.einsum("bpd,bpd->bp", hint_proj, hints)
        return sim

    def process_task_count(self) -> None:
        self.task_count += 1
        self.task_start = int(self.task_count * self.nb_pt)
        self.task_end = int((self.task_count + 1) * self.nb_pt)
        # for idx in range(self.task_start, self.task_end):
        #     *_, Vh0 = torch.linalg.svd(self.e, full_matrices=False)
        #     self.Vh.append()

    def forward(
        self,
        x_querry: Tensor,
        idx: int,
        x_block: Tensor,
        train=False,
        task_id=None,
    ):
        assert self.task_count == task_id, "task id does not match"
        # e prompts
        e_valid = False
        int_e_key = None
        int_e_value = None
        if idx in self.e_layers:
            e_valid = True

            g_p = getattr(self, "g_cls")
            e_a = getattr(self, f"e_a_{idx}")
            e_key = getattr(self, f"e_k_{idx}")
            e_prompt = getattr(self, f"e_p_{idx}")
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            task_start = int(self.task_count * nb_pt)
            task_end = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            if train:
                if self.task_count > 0:
                    g_p = torch.cat(
                        (
                            g_p[:task_start].detach().clone(),
                            g_p[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_key = torch.cat(
                        (
                            e_key[:task_start].detach().clone(),
                            e_key[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_a = torch.cat(
                        (
                            e_a[:task_start].detach().clone(),
                            e_a[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_prompt = torch.cat(
                        (
                            e_prompt[:task_start].detach().clone(),
                            e_prompt[task_start:task_end],
                        ),
                        dim=0,
                    )
                else:
                    g_p = g_p[task_start:task_end]
                    e_key = e_key[task_start:task_end]
                    e_a = e_a[task_start:task_end]
                    e_prompt = e_prompt[task_start:task_end]
            else:
                g_p = g_p[0:task_end]
                e_key = e_key[0:task_end]
                e_a = e_a[0:task_end]
                e_prompt = e_prompt[0:task_end]
            x_querry = x_querry[:, -task_end:, :]
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum("bkd,kd->bkd", x_querry, e_a)

            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            # e_key_nrom = nn.functional.normalize(e_key, dim=1)
            # qry_norm = nn.functional.normalize(a_querry, dim=2)
            # sim = torch.einsum("bkd,kd->bk", qry_norm, e_key_nrom)

            sim = self.compute_sim(a_querry, e_prompt)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # integrated_prompt: [bt_size, p_length, emb_d]
            int_pmt = torch.einsum("bk,kld->bld", sim, e_prompt)

            # select prompts
            i = int(self.e_p_length / 2)
            int_e_key = int_pmt[:, :i, :]
            int_e_value = int_pmt[:, i:, :]

            # ortho penalty
            """
            100 x 8 x 768
            compute ortho penalty for dim=1
            """
            # loss = 0
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(g_p)
                loss += ortho_penalty(e_key)
                loss += ortho_penalty(e_a)
                loss += ortho_penalty(e_prompt.flatten(start_dim=1, end_dim=2))
                # TODO-ablation: orthogonal loss 1) formation, 2) with/without
                # loss_cls_p = 0
                # for cls_idx in range(len(p)):
                #     p_cls = p[cls_idx]
                #     loss_cls_p += ortho_penalty(p_cls)
                # loss_cls_p = loss_cls_p / len(p)
                # loss += loss_cls_p
                loss = loss * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [int_e_key, int_e_value]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


class IntPrompt(MatrixPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
        self.topk = 10

    def rewrite_sim(self, sim, train=False, targets: Optional[Tensor] = None):
        # return sim, 0
        """
        训练，随机
        Args:
            sim: [B, task_count * 10]
            targets: [B,]
        """
        topk = self.topk
        device = sim.device
        alpha = torch.randn(1)[0]

        sim_zero = torch.zeros_like(sim).to(device)
        if train and targets is not None:
            tar_vals = torch.gather(sim, dim=1, index=targets[:, None])
            sim_zero.scatter_(dim=1, index=targets[:, None], src=tar_vals)
        if train and alpha > 0.7:
            topk_idxs = (
                torch.arange(self.task_start, self.task_end)
                .unsqueeze(0)
                .repeat(sim.size(0), 1)
                .to(device)
            )
        else:
            _, topk_idxs = torch.topk(sim.abs(), topk, dim=1)
        topk_vals = torch.gather(sim, dim=1, index=topk_idxs)
        sim_re = sim_zero.scatter(dim=1, index=topk_idxs, src=topk_vals)

        pull_loss = (sim.abs().sum() - sim_zero.abs().sum()) / (
            sim.size(0) * sim.size(1) - 1
        )
        loss_sim = (sim.size(0) - sim_zero.abs().sum()) / sim.size(0)
        return sim_re, (loss_sim + pull_loss)

    # def rewrite_sim(self, sim, train=False):
    #     """
    #     训练，随机
    #     Args:
    #         sim: [B, task_count * 10]
    #     """
    #     topk = self.topk
    #     nb_pt = int(self.e_pool_size / (self.n_tasks))
    #     device = sim.device
    #     task_start = int(self.task_count * nb_pt)
    #     alpha = torch.rand(1)[0]
    #     if train and alpha > 0.7:
    #         sim_zeroed = sim
    #         sim_zeroed[:, :task_start] = 0.0
    #     else:
    #         topk_vals, topk_idxs = torch.topk(sim, topk, dim=1)
    #         sim_zeroed = torch.zeros_like(sim).to(device)
    #         sim_zeroed.scatter_(dim=1, index=topk_idxs, src=topk_vals)

    #     _, topk_mat = torch.topk(sim_zeroed, 5, dim=1)
    #     loss_sim = (1 - sim_zeroed[:, topk_mat]).mean() * 0.2
    #     return sim_zeroed, loss_sim

    def forward(
        self,
        x_querry: Tensor,
        idx: int,
        x_block: Tensor,
        train=False,
        task_id=None,
        targets=None,
    ):
        assert self.task_count == task_id, "task id does not match"
        # e prompts
        e_valid = False
        int_e_key = None
        int_e_value = None
        if idx in self.e_layers:
            e_valid = True

            g_p = getattr(self, "g_cls")
            e_a = getattr(self, f"e_a_{idx}")
            e_key = getattr(self, f"e_k_{idx}")
            e_prompt = getattr(self, f"e_p_{idx}")
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            task_start = int(self.task_count * nb_pt)
            task_end = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            "get prompts: g_p, e_key, e_a, e_prompt"
            if train:
                if self.task_count > 0:
                    g_p = torch.cat(
                        (
                            g_p[:task_start].detach().clone(),
                            g_p[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_key = torch.cat(
                        (
                            e_key[:task_start].detach().clone(),
                            e_key[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_a = torch.cat(
                        (
                            e_a[:task_start].detach().clone(),
                            e_a[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_prompt = torch.cat(
                        (
                            e_prompt[:task_start].detach().clone(),
                            e_prompt[task_start:task_end],
                        ),
                        dim=0,
                    )
                else:
                    g_p = g_p[task_start:task_end]
                    e_key = e_key[task_start:task_end]
                    e_a = e_a[task_start:task_end]
                    e_prompt = e_prompt[task_start:task_end]
            else:
                g_p = g_p[0:task_end]
                e_key = e_key[0:task_end]
                e_a = e_a[0:task_end]
                e_prompt = e_prompt[0:task_end]

            # with attention and cosine sim
            x_querry = x_querry[:, -task_end:, :]
            # x_querry = x_querry[:, 0:1, :].repeat(1, self.task_end, 1)
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # a_querry = torch.einsum("bkd,kd->bkd", x_querry, e_a)
            a_querry = x_querry
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            e_key_nrom = nn.functional.normalize(e_key, dim=1)
            qry_norm = nn.functional.normalize(a_querry, dim=2)
            # [B, (task_count + 1) * nb_pt]
            sim = torch.einsum("bkd,kd->bk", qry_norm, e_key_nrom)
            sim, loss_sim = self.rewrite_sim(sim, train=train, targets=targets)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # integrated_prompt: [bt_size, p_length, emb_d]
            int_pmt = torch.einsum("bk,kld->bld", sim, e_prompt)
            # select prompts
            i = int(self.e_p_length / 2)
            int_e_key = int_pmt[:, :i, :]
            int_e_value = int_pmt[:, i:, :]

            # ortho penalty
            """
            100 x 8 x 768
            compute ortho penalty for dim=1
            """
            loss = 0
            if train and self.ortho_mu > 0:
                loss += ortho_penalty(g_p)
                loss += ortho_penalty(e_key)
                loss += ortho_penalty(e_a)
                loss += ortho_penalty(e_prompt.flatten(start_dim=1, end_dim=2))
                # loss_cls_p = 0
                # for cls_idx in range(len(p)):
                #     p_cls = p[cls_idx]
                #     loss_cls_p += ortho_penalty(p_cls)
                # loss_cls_p = loss_cls_p / len(p)
                # loss += loss_cls_p
                loss = loss * self.ortho_mu
                loss += loss_sim
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [int_e_key, int_e_value]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


class ProjPrompt(MatrixPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
        self.topk = 10
        self.Vh = []
        self.nb_pt = int(self.e_pool_size / (self.n_tasks))
        self.task_start = int(self.task_count * self.nb_pt)
        self.task_end = int((self.task_count + 1) * self.nb_pt)

    def rewrite_sim(self, sim, train=False, targets: Optional[Tensor] = None):
        """
        训练，随机
        Args:
            sim: [B, task_count * 10]
            targets: [B,]
        """
        topk = self.topk
        device = sim.device
        alpha = torch.randn(1)[0]

        sim_zero = torch.zeros_like(sim).to(device)
        if train:
            if targets is not None:
                tar_vals = torch.gather(sim, dim=1, index=targets[:, None])
                sim_zero.scatter_(dim=1, index=targets[:, None], src=tar_vals)

        if train and alpha > 0.7:
            topk_idxs = (
                torch.arange(self.task_start, self.task_end)
                .unsqueeze(0)
                .repeat(sim.size(0), 1)
                .to(device)
            )
        elif train:
            _, topk_idxs = torch.topk(sim.abs(), topk, dim=1)
        else:
            _, topk_idxs = torch.topk(sim.abs(), topk, dim=1)
        topk_vals = torch.gather(sim, dim=1, index=topk_idxs)
        sim_re = sim_zero.scatter(dim=1, index=topk_idxs, src=topk_vals)

        pull_loss = (sim.abs().sum() - sim_zero.abs().sum()) / (
            sim.size(0) * sim.size(1) - 1
        )
        loss_sim = (sim.size(0) - sim_zero.abs().sum()) / sim.size(0)
        return sim_re, (loss_sim + pull_loss * 2)

    def compute_sim(self, hints, pmts):
        """
        Args:
            hints: [batch_size, nb_pmt, dim]
            pmts: [nb_pmt, pmt_len, dim]

        Returns:
            sim: [batch_size, nb_pmt]
        """
        Vh = []
        for idx in range(self.task_end):
            *_, Vh0 = torch.linalg.svd(
                pmts[idx].detach().clone(), full_matrices=False
            )
            Vh.append(Vh0)
        Vh = torch.stack(Vh, dim=0)
        proj_mat = torch.einsum("pij,pjk -> pik", Vh.permute(0, 2, 1), Vh)
        hint_proj = torch.einsum("bpd,pdd->bpd", hints, proj_mat)
        hint_proj = F.normalize(hint_proj, dim=-1, p=2)
        hints = F.normalize(hints, dim=-1, p=2)
        sim = torch.einsum("bpd,bpd->bp", hint_proj, hints) * 50
        sim = torch.sigmoid(sim)
        return sim

    def forward(
        self,
        x_querry: Tensor,
        idx: int,
        x_block: Tensor,
        train=False,
        task_id=None,
        targets=None,
    ):
        assert self.task_count == task_id, "task id does not match"
        # e prompts
        e_valid = False
        int_e_key = None
        int_e_value = None
        if idx in self.e_layers:
            e_valid = True

            g_p = getattr(self, "g_cls")
            e_a = getattr(self, f"e_a_{idx}")
            e_key = getattr(self, f"e_k_{idx}")
            e_prompt = getattr(self, f"e_p_{idx}")
            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            "get prompts: g_p, e_key, e_a, e_prompt"
            if train:
                if self.task_count > 0:
                    g_p = torch.cat(
                        (
                            g_p[: self.task_start].detach().clone(),
                            g_p[self.task_start : self.task_end],
                        ),
                        dim=0,
                    )
                    e_key = torch.cat(
                        (
                            e_key[: self.task_start].detach().clone(),
                            e_key[self.task_start : self.task_end],
                        ),
                        dim=0,
                    )
                    e_a = torch.cat(
                        (
                            e_a[: self.task_start].detach().clone(),
                            e_a[self.task_start : self.task_end],
                        ),
                        dim=0,
                    )
                    e_prompt = torch.cat(
                        (
                            e_prompt[: self.task_start].detach().clone(),
                            e_prompt[self.task_start : self.task_end],
                        ),
                        dim=0,
                    )
                else:
                    g_p = g_p[self.task_start : self.task_end]
                    e_key = e_key[self.task_start : self.task_end]
                    e_a = e_a[self.task_start : self.task_end]
                    e_prompt = e_prompt[self.task_start : self.task_end]
            else:
                g_p = g_p[0 : self.task_end]
                e_key = e_key[0 : self.task_end]
                e_a = e_a[0 : self.task_end]
                e_prompt = e_prompt[0 : self.task_end]

            # with attention and cosine sim
            # x_querry = x_querry[:, -self.task_end :, :]
            x_querry = x_querry[:, 0:1, :].repeat(1, self.task_end, 1)

            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # a_querry = torch.einsum("bkd,kd->bkd", x_querry, e_a)
            a_querry = x_querry

            # [B, (task_count + 1) * nb_pt] compute similarity with keys
            sim = self.compute_sim(
                a_querry, e_prompt[:, : int(self.e_p_length / 2), :]
            )
            sim, loss_sim = self.rewrite_sim(sim, train=train, targets=targets)
            int_pmt = torch.einsum("bk,kld->bld", sim, e_prompt)
            # select prompts
            i = int(self.e_p_length / 2)
            int_e_key = int_pmt[:, :i, :]
            int_e_value = int_pmt[:, i:, :]

            # # [B, topk]
            # bt_size = x_querry.size(0)
            # _, topk_idxs = torch.topk(sim.abs(), self.topk, dim=1)
            # # [B, (task_count + 1) * nb_pt, pmt_length, emb_d]
            # e_prompt_sim = e_prompt[None, :] * sim[:, :, None, None]
            # # [B, topk, pmt_length, emb_d]
            # e_prompt_sim = e_prompt_sim.gather(
            #     dim=1,
            #     index=topk_idxs[:, :, None, None].expand(
            #         -1, -1, *e_prompt_sim.shape[-2:]
            #     ),
            # )
            # int_pmt = e_prompt_sim.reshape(
            #     bt_size, self.topk, self.e_p_length, -1
            # )

            # p_len = int(self.e_p_length / 2)
            # int_e_key = int_pmt[:, :, :p_len, :]
            # int_e_value = int_pmt[:, :, p_len:, :]
            # int_e_key = int_e_key.reshape(bt_size, -1, int_e_key.size(-1))
            # int_e_value = int_e_value.reshape(
            #     bt_size, -1, int_e_value.size(-1)
            # )

            # ortho penalty
            """
            100 x 8 x 768
            compute ortho penalty for dim=1
            """
            loss = 0
            if train and self.ortho_mu > 0:
                loss += ortho_penalty(g_p)
                # loss += ortho_penalty(e_key)
                loss += ortho_penalty(e_a)
                # loss += ortho_penalty(e_prompt.flatten(start_dim=1, end_dim=2))
                # loss_cls_p = 0
                # for cls_idx in range(len(e_prompt)):
                #     p_cls = e_prompt[cls_idx]
                #     loss_cls_p += ortho_penalty(p_cls)
                # loss_cls_p = loss_cls_p / len(e_prompt)
                # loss += loss_cls_p
                loss = loss * self.ortho_mu
                loss += loss_sim
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [int_e_key, int_e_value]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def process_task_count(self) -> None:
        self.task_count += 1
        self.task_start = int(self.task_count * self.nb_pt)
        self.task_end = int((self.task_count + 1) * self.nb_pt)
        # for idx in range(self.task_start, self.task_end):
        #     *_, Vh0 = torch.linalg.svd(self.e, full_matrices=False)
        #     self.Vh.append()


class Catprompt(MatrixPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
        self.alpha = 10
        self.topk = 10

    def compute_sim(self, hints):
        # [B, 1, emb_d]
        q = hints[:, 0:1, :]
        # [B, int((self.task_count + 1) * self.nb_pt), emb_d]
        cls_hints = hints[:, -self.task_end :, :]
        q_p = F.normalize(q, p=2, dim=-1)
        cls_hints_p = F.normalize(cls_hints, p=2, dim=-1)
        # [B, int((self.task_count + 1) * self.nb_pt)]
        sim = torch.einsum("bkd, bld -> bkl", q_p, cls_hints_p)
        # sim = torch.sigmoid(sim * self.alpha)
        return sim.squeeze()

    def rewrite_sim(self, sim, train=False, targets=None):
        """
        训练，随机
        Args:
            sim: [B, task_count * 10]
        """
        topk = self.topk
        device = sim.device
        alpha = torch.randn(1)[0]
        cls_loss = 0.0

        sim_zero = torch.zeros_like(sim).to(device)
        if train and targets is not None:
            tar_vals = torch.gather(sim, dim=1, index=targets[:, None])
            sim_zero.scatter_(dim=1, index=targets[:, None], src=tar_vals)
            cls_loss = nn.CrossEntropyLoss()(sim, targets) / 5.0
        if train and alpha > 0.7:
            topk_idxs = (
                torch.arange(self.task_start, self.task_end)
                .unsqueeze(0)
                .repeat(sim.size(0), 1)
                .to(device)
            )
        else:
            _, topk_idxs = torch.topk(sim.abs(), topk, dim=1)
        topk_vals = torch.gather(sim, dim=1, index=topk_idxs)
        sim_re = sim_zero.scatter(dim=1, index=topk_idxs, src=topk_vals)
        return sim_re, cls_loss

        # pull_loss = (sim.abs().sum() - sim_zero.abs().sum()) / (
        #     sim.size(0) * sim.size(1) - 1
        # )
        # loss_sim = (sim.size(0) - sim_zero.abs().sum()) / sim.size(0)
        # return sim_re, (loss_sim + pull_loss)

    def forward(
        self,
        x_query: Tensor,
        idx: int,
        x_block: Tensor,
        train=False,
        task_id=None,
        targets=None,
    ):
        assert self.task_count == task_id, "task id does not match"
        # e prompts
        e_valid = False
        int_e_key = None
        int_e_value = None
        if idx in self.e_layers:
            e_valid = True

            g_p = getattr(self, "g_cls")
            e_a = getattr(self, f"e_a_{idx}")
            e_key = getattr(self, f"e_k_{idx}")
            e_prompt = getattr(self, f"e_p_{idx}")
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            task_start = int(self.task_count * nb_pt)
            task_end = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            "get prompts: g_p, e_key, e_a, e_prompt"
            if train:
                if self.task_count > 0:
                    g_p = torch.cat(
                        (
                            g_p[:task_start].detach().clone(),
                            g_p[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_key = torch.cat(
                        (
                            e_key[:task_start].detach().clone(),
                            e_key[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_a = torch.cat(
                        (
                            e_a[:task_start].detach().clone(),
                            e_a[task_start:task_end],
                        ),
                        dim=0,
                    )
                    e_prompt = torch.cat(
                        (
                            e_prompt[:task_start].detach().clone(),
                            e_prompt[task_start:task_end],
                        ),
                        dim=0,
                    )
                else:
                    g_p = g_p[task_start:task_end]
                    e_key = e_key[task_start:task_end]
                    e_a = e_a[task_start:task_end]
                    e_prompt = e_prompt[task_start:task_end]
            else:
                g_p = g_p[0:task_end]
                e_key = e_key[0:task_end]
                e_a = e_a[0:task_end]
                e_prompt = e_prompt[0:task_end]

            # # with attention and cosine sim
            # querry = x_query[:, -task_end:, :]
            # # x_querry = x_querry[:, 0:1, :].repeat(1, self.task_end, 1)
            # # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # a_querry = torch.einsum("bkd,kd->bkd", x_querry, e_a)
            # a_querry = querry
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            # e_key_nrom = nn.functional.normalize(e_key, dim=1)
            # qry_norm = nn.functional.normalize(a_querry, dim=2)
            # # [B, (task_count + 1) * nb_pt]
            # sim = torch.einsum("bkd,kd->bk", qry_norm, e_key_nrom)

            sim = self.compute_sim(x_query)
            sim, loss_sim = self.rewrite_sim(sim, train=train, targets=targets)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # integrated_prompt: [bt_size, p_length, emb_d]
            int_pmt = torch.einsum("bk,kld->bld", sim, e_prompt)
            # select prompts
            i = int(self.e_p_length / 2)
            int_e_key = int_pmt[:, :i, :]
            int_e_value = int_pmt[:, i:, :]

            # ortho penalty
            """
            100 x 8 x 768
            compute ortho penalty for dim=1
            """
            loss = 0
            if train and self.ortho_mu > 0:
                loss += ortho_penalty(g_p)
                loss += ortho_penalty(e_key)
                loss += ortho_penalty(e_a)
                loss += ortho_penalty(e_prompt.flatten(start_dim=1, end_dim=2))
                # loss_cls_p = 0
                # for cls_idx in range(len(p)):
                #     p_cls = p[cls_idx]
                #     loss_cls_p += ortho_penalty(p_cls)
                # loss_cls_p = loss_cls_p / len(p)
                # loss += loss_cls_p
                loss = loss * self.ortho_mu
                loss += loss_sim
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [int_e_key, int_e_value]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t: Tensor) -> Tensor:
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean() * 1e-6


def tensor_prompt(a, b, c=None, ortho=False) -> Parameter:
    if c is None:
        p = torch.nn.parameter.Parameter(
            torch.FloatTensor(a, b), requires_grad=True
        )
    else:
        p = torch.nn.parameter.Parameter(
            torch.FloatTensor(a * b, c), requires_grad=True
        )
    if ortho:
        nn.init.orthogonal_(p)
        if c is not None:
            p.data = p.data.reshape(a, b, c)
    else:
        nn.init.uniform_(p)
    return p


class TopDownZoo(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        prompt_flag: str = "None",
        prompt_param: List[float] = [],
    ):
        """ """
        super(TopDownZoo, self).__init__()
        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag: str = prompt_flag
        self.task_id = 0

        # get feature encoder
        zoo_model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ckpt_layer=0,
            drop_path_rate=0,
        )
        from timm.models import vit_base_patch16_224

        load_dict = vit_base_patch16_224(pretrained=True).state_dict()
        del load_dict["head.weight"]
        del load_dict["head.bias"]
        zoo_model.load_state_dict(load_dict)

        # classifier
        # self.last = nn.Linear(768, num_classes)
        self.last = CosineLinear(768, num_classes, sigma=True)

        # create prompting module
        if self.prompt_flag == "matrix":
            self.prompt = MatrixPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "int":
            self.prompt = IntPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "cat":
            self.prompt = Catprompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "proj":
            self.prompt = ProjPrompt(768, prompt_param[0], prompt_param[1])
        else:
            raise NotImplementedError
        print("Prompt strategy: ", self.prompt_flag)

        # feature encoder changes if transformer vs resnet
        self.feat: VisionTransformer = zoo_model

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False, targets=None):
        assert self.task_id is not None, "task_id is None"

        prompt_loss = tensor(0.0)
        cls_hint = tensor(0.0)
        glob_x = None
        B = x.size(0)
        if self.prompt is not None:
            "initialize class hint"
            glob_start = (self.task_id) * self.prompt.nb_pt
            glob_end = (self.task_id + 1) * self.prompt.nb_pt
            g_cls = getattr(self.prompt, "g_cls")
            glob_x = torch.cat(
                (
                    g_cls[:glob_start, :].detach().clone(),
                    g_cls[glob_start:glob_end, :],
                ),
                dim=0,
            )
            glob_x = glob_x[None, :].expand(B, -1, -1)
            assert glob_x.size(1) == glob_end, "glob_x size error"

            with torch.no_grad():
                o, _ = self.feat(x, glob_x=glob_x)
                # q = o[:, 0, :]
                # q = o[:, -glob_end:, :]
            out, prompt_loss = self.feat(
                x,
                prompt=self.prompt,
                q=o,
                glob_x=glob_x,
                task_id=self.task_id,
                train=train,
                targets=targets,
            )
            cls_hint = out[:, -glob_end:, :]
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)

        "return cfg"
        if self.prompt is not None and train and glob_x is not None:
            return (out, cls_hint), prompt_loss
        elif self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

    def get_query(self, x, apy_glob=True, task_id=None):
        assert self.task_id is not None, "task_id is None"
        task_id = self.task_id if task_id is None else task_id
        glob_x = None
        B = x.size(0)
        if self.prompt is not None:
            "initialize class hint"
            glob_start = (task_id) * self.prompt.nb_pt
            glob_end = (task_id + 1) * self.prompt.nb_pt
            g_cls = getattr(self.prompt, "g_cls")
            glob_x = torch.cat(
                (
                    g_cls[:glob_start, :].detach().clone(),
                    g_cls[glob_start:glob_end, :],
                ),
                dim=0,
            )
            glob_x = glob_x[None, :].expand(B, -1, -1)
            assert glob_x.size(1) == glob_end, "glob_x size error"

            with torch.no_grad():
                o, _ = self.feat(x, glob_x=glob_x)

            return o


def vit_pt_td(
    out_dim, block_division=None, prompt_flag="None", prompt_param=[]
):
    return TopDownZoo(
        num_classes=out_dim, prompt_flag=prompt_flag, prompt_param=prompt_param
    )


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(10)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        if self.sigma is not None:
            out = self.sigma * out
        return out
