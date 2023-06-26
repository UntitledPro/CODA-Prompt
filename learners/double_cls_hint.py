from __future__ import print_function
import torch
import models
from .default import NormalNN
from torch.nn import functional as F
from utils.schedulers import CosineSchedule


class VHint(NormalNN):
    """add global class hints, update by targets
    model output: (logits, glob_hints), prompt_loss

    """

    def __init__(self, learner_config):
        self.prompt_param = learner_config["prompt_param"]
        super(VHint, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        """compute three kind of loss:
        - hint_emb loss
        - contrastive loss between hint_emb and cls_hint,
            with mask generated by targets
        - ce loss with heuristic
        """
        # logits
        B = inputs.size(0)
        o, prompt_loss = self.model(inputs, train=True, targets=targets)
        logits, cls_hint = o

        # hint loss
        # collect embeddings for input image based on targets
        hint_emb = torch.gather(
            input=cls_hint,
            dim=1,
            index=targets[:, None, None].repeat(1, 1, cls_hint.size(-1)),
        ).squeeze()
        hint_logits = self.model.module.last(hint_emb)

        # contrastive loss
        # [B, emb_d]
        hint_emb = F.normalize(hint_emb, p=2, dim=-1)
        # [B, nb_cls, emb_d]
        cls_hint = F.normalize(cls_hint, p=2, dim=-1)
        # compute sim_matrix: [B, nb_cls]
        # sim_matrix[i,j]=sim(hint_emb[i], cls_hint[i][j])
        sim_mat = torch.einsum("bd,bcd->bc", hint_emb, cls_hint)
        # mask: [B,] generated by targets
        mask = (
            torch.zeros_like(sim_mat).scatter_(1, targets[:, None], 1).bool()
        )
        sim_mat = sim_mat[~mask].view(B, -1)
        loss_sim = (torch.nn.LeakyReLU()(sim_mat)).abs().mean()

        # ce with heuristic
        # logits = torch.cat([logits, hint_logits], dim=0)
        # targets = torch.cat([targets, targets], dim=0)
        hint_logits = hint_logits[:, : self.valid_out_dim]
        hint_logits[:, : self.last_valid_out_dim] = -float("inf")
        logits = logits[:, : self.valid_out_dim]
        logits[:, : self.last_valid_out_dim] = -float("inf")
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        hint_loss = self.criterion(hint_logits, targets.long(), dw_cls)

        # ce loss
        # print('prompt loss: ', prompt_loss.sum())
        total_loss = (
            total_loss + prompt_loss.mean() + loss_sim / 2 + hint_loss
        )

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self):
        # parse optimizer args
        # Multi-GPU
        if len(self.config["gpuid"]) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(
                self.model.module.last.parameters()
            )
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(
                self.model.last.parameters()
            )
        print("*****************************************")
        optimizer_arg = {
            "params": params_to_opt,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        if self.config["optimizer"] in ["SGD", "RMSprop"]:
            optimizer_arg["momentum"] = self.config["momentum"]
        elif self.config["optimizer"] in ["Rprop"]:
            optimizer_arg.pop("weight_decay")
        elif self.config["optimizer"] == "amsgrad":
            optimizer_arg["amsgrad"] = True
            self.config["optimizer"] = "Adam"
        elif self.config["optimizer"] == "Adam":
            optimizer_arg["betas"] = (self.config["momentum"], 0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](
            **optimizer_arg
        )

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(
                self.optimizer, K=self.schedule[-1]
            )
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config["gpuid"]) > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=self.config["gpuid"],
                output_device=self.config["gpuid"][0],
            )
        return self


class IntHint(VHint):
    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="int",
            prompt_param=self.prompt_param,
        )
        return model


class MatrixHint(VHint):
    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="matrix",
            prompt_param=self.prompt_param,
        )
        return model


class CatHint(VHint):
    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="cat",
            prompt_param=self.prompt_param,
        )
        return model


class ProjHint(VHint):
    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="proj",
            prompt_param=self.prompt_param,
        )
        return model
