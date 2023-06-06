import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from .vit import VisionTransformer
from typing import List
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
                self.e_pool_size, e_l, emb_d, ortho=True)
            # [pool_size, k_dim]
            k: Parameter = tensor_prompt(
                self.e_pool_size, self.key_d, ortho=True)
            # [pool_size, k_dim]
            a: Parameter = tensor_prompt(
                self.e_pool_size, self.key_d, ortho=True)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

        g_cls = tensor_prompt(self.g_pool_size, self.emb_d, ortho=True)
        setattr(self, 'g_cls', g_cls)
        if self.ortho_mu > 0:
            print('|-' * 100 + '|')
            print('ortho penalty is on')

    def _init_smart(self, prompt_param) -> None:
        '''initiate basic parameters: int | float | List'''
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

    def process_task_count(self) -> None:
        self.task_count += 1

    def forward(self, x_querry: Tensor,
                idx: int, x_block: Tensor,
                train=False, task_id=None):
        assert self.task_count == task_id, 'task id does not match'
        # e prompts
        e_valid = False
        int_e_key = None
        int_e_value = None
        if idx in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            g_p = getattr(self, 'g_cls')
            e_a = getattr(self, f'e_a_{idx}')
            e_key = getattr(self, f'e_k_{idx}')
            e_prompt = getattr(self, f'e_p_{idx}')
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            task_start = int(self.task_count * nb_pt)
            task_end = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            if train:
                if self.task_count > 0:
                    g_p = torch.cat(
                        (g_p[:task_start].detach().clone(),
                         g_p[task_start:task_end]),
                        dim=0)
                    e_key = torch.cat(
                        (e_key[:task_start].detach().clone(),
                         e_key[task_start:task_end]),
                        dim=0)
                    e_a = torch.cat(
                        (e_a[:task_start].detach().clone(),
                         e_a[task_start:task_end]),
                        dim=0)
                    e_prompt = torch.cat(
                        (e_prompt[:task_start].detach().clone(),
                         e_prompt[task_start:task_end]),
                        dim=0)
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
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, e_a)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            e_key_nrom = nn.functional.normalize(e_key, dim=1)
            qry_norm = nn.functional.normalize(a_querry, dim=2)
            sim = torch.einsum('bkd,kd->bk', qry_norm, e_key_nrom)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # integrated_prompt: [bt_size, p_length, emb_d]
            int_pmt = torch.einsum('bk,kld->bld', sim, e_prompt)

            # select prompts
            i = int(self.e_p_length / 2)
            int_e_key = int_pmt[:, :i, :]
            int_e_value = int_pmt[:, i:, :]

            # ortho penalty
            '''
            100 x 8 x 768
            compute ortho penalty for dim=1
            '''
            # loss = 0
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(g_p)
                loss += ortho_penalty(e_key)
                loss += ortho_penalty(e_a)
                loss += ortho_penalty(e_prompt.
                                      flatten(start_dim=1, end_dim=2))
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

    def rewrite_sim(self, sim, train=False):
        """
        训练，随机
        Args:
            sim: [B, task_count * 10]
        """
        topk = self.topk
        nb_pt = int(self.e_pool_size / (self.n_tasks))
        device = sim.device
        task_start = int(self.task_count * nb_pt)
        alpha = torch.rand(1)[0]
        if train and alpha > 0.7:
            sim_zeroed = sim
            sim_zeroed[:, :task_start] = 0.
        else:
            topk_vals, topk_idxs = torch.topk(sim, topk, dim=1)
            sim_zeroed = torch.zeros_like(sim).to(device)
            sim_zeroed.scatter_(dim=1, index=topk_idxs, src=topk_vals)

        _, topk_mat = torch.topk(sim_zeroed, 3, dim=1)
        loss_sim = (1 - sim_zeroed[:, topk_mat]).mean() * 0.2
        return sim_zeroed, loss_sim

    def forward(self, x_querry: Tensor,
                idx: int, x_block: Tensor,
                train=False, task_id=None):
        assert self.task_count == task_id, 'task id does not match'
        # e prompts
        e_valid = False
        int_e_key = None
        int_e_value = None
        if idx in self.e_layers:
            e_valid = True

            g_p = getattr(self, 'g_cls')
            e_a = getattr(self, f'e_a_{idx}')
            e_key = getattr(self, f'e_k_{idx}')
            e_prompt = getattr(self, f'e_p_{idx}')
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            task_start = int(self.task_count * nb_pt)
            task_end = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            'get prompts: g_p, e_key, e_a, e_prompt'
            if train:
                if self.task_count > 0:
                    g_p = torch.cat(
                        (g_p[:task_start].detach().clone(),
                         g_p[task_start:task_end]),
                        dim=0)
                    e_key = torch.cat(
                        (e_key[:task_start].detach().clone(),
                         e_key[task_start:task_end]),
                        dim=0)
                    e_a = torch.cat(
                        (e_a[:task_start].detach().clone(),
                         e_a[task_start:task_end]),
                        dim=0)
                    e_prompt = torch.cat(
                        (e_prompt[:task_start].detach().clone(),
                         e_prompt[task_start:task_end]),
                        dim=0)
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
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, e_a)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            e_key_nrom = nn.functional.normalize(e_key, dim=1)
            qry_norm = nn.functional.normalize(a_querry, dim=2)
            # [B, (task_count + 1) * nb_pt]
            sim = torch.einsum('bkd,kd->bk', qry_norm, e_key_nrom)
            sim, loss_sim = self.rewrite_sim(sim, train=train)
            # # TODO-ablation: using softmax for attention, damage accuracy
            # aq_k = nn.Softmax(dim=1)(aq_k * self.key_d ** -0.5)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # integrated_prompt: [bt_size, p_length, emb_d]
            int_pmt = torch.einsum('bk,kld->bld', sim, e_prompt)

            # select prompts
            i = int(self.e_p_length / 2)
            int_e_key = int_pmt[:, :i, :]
            int_e_value = int_pmt[:, i:, :]

            # ortho penalty
            '''
            100 x 8 x 768
            compute ortho penalty for dim=1
            '''
            loss = 0
            if train and self.ortho_mu > 0:
                loss += ortho_penalty(g_p)
                loss += ortho_penalty(e_key)
                loss += ortho_penalty(e_a)
                loss += ortho_penalty(e_prompt.
                                      flatten(start_dim=1, end_dim=2))
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

            loss += loss_sim
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
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6


def tensor_prompt(a, b, c=None, ortho=False) -> Parameter:
    if c is None:
        p = torch.nn.parameter.Parameter(
            torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.parameter.Parameter(
            torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class TopDownZoo(nn.Module):
    def __init__(self, num_classes: int = 10,
                 prompt_flag: str = 'None',
                 prompt_param: List[float] = []):
        """
        """
        super(TopDownZoo, self).__init__()
        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag: str = prompt_flag
        self.task_id = 0

        # get feature encoder
        zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                      num_heads=12, ckpt_layer=0,
                                      drop_path_rate=0
                                      )
        from timm.models import vit_base_patch16_224
        load_dict = vit_base_patch16_224(pretrained=True).state_dict()
        del load_dict['head.weight']
        del load_dict['head.bias']
        zoo_model.load_state_dict(load_dict)

        # classifier
        # self.last = nn.Linear(768, num_classes)
        self.last = CosineLinear(768, num_classes, sigma=True)

        # create prompting module
        if self.prompt_flag == 'matrix':
            self.prompt = MatrixPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'int':
            self.prompt = IntPrompt(768, prompt_param[0], prompt_param[1])
        else:
            raise NotImplementedError
        print('Prompt strategy: ', self.prompt_flag)

        # feature encoder changes if transformer vs resnet
        self.feat: VisionTransformer = zoo_model

    def cls_hint_loss(self, cls_hint: Tensor):
        '''
        Args:
            cls_hint:   [B, (self.task_id + 1) * self.prompt.nb_pt, emb_d]
                        ex. [32, 10 * task_id, 768]
        '''
        # start index of the task
        task_s = (self.task_id) * self.prompt.nb_pt
        # end index of the task
        task_e = (self.task_id + 1) * self.prompt.nb_pt
        cls_hint_task = cls_hint
        # [(self.task_id + 1) * self.prompt.nb_pt,]
        hint_labels = torch.arange(
            cls_hint_task.size(1)).to(cls_hint_task.device)
        # [B, (self.task_id + 1) * self.prompt.nb_pt,]
        hint_labels = hint_labels[None, :].expand(cls_hint_task.size(0), -1)
        # [B * (self.task_id + 1) * self.prompt.nb_pt,]
        hint_labels = hint_labels.reshape(-1)
        # [B * (self.task_id + 1) * self.prompt.nb_pt, num_classes]
        hint_logits = self.last(
            cls_hint_task.reshape(-1, cls_hint_task.size(-1)))

        # detach 0:task_s, set task_e:end to -inf, and keep task_s:task_e
        # TODO: detach or not detach?
        # hint_logits = torch.cat(
        #     (hint_logits[:, :task_s].detach().clone(),
        #      hint_logits[:, task_s:]),
        #     dim=1
        # )
        hint_logits = torch.cat(
            (hint_logits[:, :task_s],
             hint_logits[:, task_s:]),
            dim=1
        )
        hint_logits[:, task_e:] = -torch.inf

        critn = nn.CrossEntropyLoss()
        hint_loss = critn(hint_logits, hint_labels)
        return hint_loss

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        assert self.task_id is not None, 'task_id is None'

        prompt_loss = tensor(0.)
        cls_hint = tensor(0.)
        B = x.size(0)
        if self.prompt is not None:
            'initialize class hint'
            glob_start = (self.task_id) * self.prompt.nb_pt
            glob_end = (self.task_id + 1) * self.prompt.nb_pt
            g_cls = getattr(self.prompt, 'g_cls')
            glob_x = torch.cat(
                (g_cls[:glob_start, :].detach().clone(),
                 g_cls[glob_start:glob_end, :]),
                dim=0
            )
            glob_x = glob_x[None, :].expand(B, -1, -1)
            assert glob_x.size(1) == glob_end, 'glob_x size error'

            with torch.no_grad():
                q, _ = self.feat(x, glob_x=glob_x)
                q = q[:, 0, :]
            out, prompt_loss = self.feat(
                x, prompt=self.prompt, q=q,
                glob_x=glob_x,
                train=train, task_id=self.task_id)
            cls_hint = out[:, -glob_end:, :]
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)

        'return cfg'
        if self.prompt is not None and train:
            return (out,), prompt_loss
        elif self.prompt is not None and train and glob_x is not None:
            return (out, cls_hint), prompt_loss
        else:
            return (out,)


def vit_pt_td(out_dim, block_division=None, prompt_flag='None', prompt_param=None):
    return TopDownZoo(num_classes=out_dim, prompt_flag=prompt_flag, prompt_param=prompt_param)


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(10)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1),
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out
