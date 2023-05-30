from ctypes import Union
from cv2 import Mat
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from .vit import VisionTransformer
from typing import Dict, List, Union, Any
from torch import Tensor, tensor
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

    def rand_topk(self, similarity: Tensor):
        return similarity.topk(self.nb_pt, dim=1)

    def forward(self, x_querry: Tensor, idx: int,
                x_block: Tensor, train=False, task_id=None):
        assert self.task_count == task_id, 'task id does not match'
        # e prompts
        e_valid = False
        if idx in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            g_p: Parameter = getattr(self, 'g_cls')
            K: Parameter = getattr(self, f'e_k_{idx}')
            A: Parameter = getattr(self, f'e_a_{idx}')
            p: Parameter = getattr(self, f'e_p_{idx}')
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * nb_pt)
            f = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            # prompt: [n_task * 10, p_length, emb_d]
            if train:
                if self.task_count > 0:
                    g_p = torch.cat((g_p[:s].detach().clone(), g_p[s:f]), dim=0)
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    g_p = g_p[s:f]
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                g_p = g_p[0:f]
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # # TODO-ablation: using softmax for attention, damage accuracy
            # aq_k = nn.Softmax(dim=1)(aq_k * self.key_d ** -0.5)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            # [bt_size, p_length, emb_d]
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            '''
            100 x 8 x 768
            compute ortho penalty for dim=1
            '''
            # loss = 0
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(g_p)
                loss += ortho_penalty(K)
                loss += ortho_penalty(A)
                loss += ortho_penalty(p.flatten(start_dim=1, end_dim=2))
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
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

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

    def _init_smart(self, emb_d, prompt_param) -> None:
        '''initiate basic parameters: int | float | List'''
        # prompt basic param
        # number of prompt groups in the pool
        self.e_pool_size = int(prompt_param[0])
        # number of prompts in each prompt group
        self.e_p_length = int(prompt_param[1])
        self.e_layers: list[int] = [0, 1, 2, 3, 4]

        # strength of ortho penalty
        self.ortho_mu: float = prompt_param[2]

    def process_task_count(self) -> None:
        self.task_count += 1

    def forward(self, x_querry: Tensor, idx: int,
                x_block: Tensor, train=False, task_id=None):

        # e prompts
        e_valid = False
        if idx in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K: Parameter = getattr(self, f'e_k_{idx}')
            A: Parameter = getattr(self, f'e_a_{idx}')
            p: Parameter = getattr(self, f'e_p_{idx}')
            # the number of prompts per task
            nb_pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * nb_pt)
            f = int((self.task_count + 1) * nb_pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # weighted sum
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K)
                loss += ortho_penalty(A)
                loss += ortho_penalty(p.flatten(start_dim=1, end_dim=2))
                loss = loss * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t: Tensor) -> Tensor:
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }


class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(prompt_param)

        # g prompt init [0, 1]
        # number of prompts: 6
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d, ortho=True)
            setattr(self, f'g_p_{g}', p)

        # e prompt init [2, 3, 4]
        # for each layer register a prompt pool/key
        # number of prompts: 10 * 10 = 100
        for e in self.e_layers:
            # [pool_size, p_length, p_dim]:[10, 20, d]
            p = tensor_prompt(self.e_pool_size, self.e_p_length,
                              emb_d, ortho=True)
            # [pool_size, k_dim]
            k = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, layer_idx, x_block, train=False, task_id=None):
        # e prompts
        e_valid = False
        if layer_idx in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            # [pool_size, p_length, emb_d]
            K = getattr(self, f'e_k_{layer_idx}')  # 0 based indexing here
            p = getattr(self, f'e_p_{layer_idx}')  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)

            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    # loss = (1.0 - cos_sim[:, task_id]).sum()
                    loss = (1.0 - cos_sim[:, task_id]).mean()
                    # [B, p_length, emb_d]
                    P_ = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    # loss = (1.0 - cos_sim[:, k_idx]).sum()
                    loss = (1.0 - cos_sim[:, k_idx]).mean()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]

            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))

        # g prompts
        g_valid = False
        if layer_idx in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f'g_p_{layer_idx}')  # 0 based indexing here
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            loss = loss * 0.1
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }


class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0, 1, 2, 3, 4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

# note - ortho init has not been found to help l2p/dual prompt


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


class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, prompt_flag=False,
                 prompt_param: List[float] = []):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag: bool = prompt_flag
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
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'matrix':
            self.prompt = MatrixPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        print('Prompt strategy: ', self.prompt_flag)

        # feature encoder changes if transformer vs resnet
        self.feat: VisionTransformer = zoo_model

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        prompt_loss = 0
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:, 0, :]
            out, prompt_loss = self.feat(
                x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class TopDownZoo(nn.Module):
    def __init__(self, num_classes=10, prompt_flag=False,
                 prompt_param: List[float] = []):
        super(TopDownZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag: bool = prompt_flag
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
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        assert self.prompt_flag == 'matrix', 'only support matrix prompt'
        print('Prompt strategy: ', self.prompt_flag)
        self.prompt = MatrixPrompt(768, prompt_param[0], prompt_param[1])

        # feature encoder changes if transformer vs resnet
        self.feat: VisionTransformer = zoo_model

    def cls_hint_loss(self, cls_hint: Tensor):
        '''
        Args:
            cls_hint:   [B, (self.task_id + 1) * self.prompt.nb_pt, emb_d]
                        ex. [32, 10 * task_id, 768]
        '''
        # [(self.task_id + 1) * self.prompt.nb_pt,]
        hint_labels = torch.arange(cls_hint.size(1)).to(cls_hint.device)
        # [B, (self.task_id + 1) * self.prompt.nb_pt,]
        hint_labels = hint_labels[None, :].expand(cls_hint.size(0), -1)
        # [B * (self.task_id + 1) * self.prompt.nb_pt,]
        hint_labels = hint_labels.reshape(-1)
        # [B * (self.task_id + 1) * self.prompt.nb_pt, num_classes]
        hint_logits = self.last(cls_hint.reshape(-1, cls_hint.size(-1)))

        # start index of the task
        task_s = (self.task_id) * self.prompt.nb_pt
        # end index of the task
        task_e = (self.task_id + 1) * self.prompt.nb_pt
        # detach 0:task_s, set task_e:end to -inf, and keep task_s:task_e
        hint_logits = torch.cat(
            (hint_logits[:, :task_s].detach().clone(),
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

        prompt_loss = tensor(0)
        cls_hint = tensor(0)
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
            cls_hint = out[:, -self.prompt.nb_pt:, :]
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            hint_loss = self.cls_hint_loss(cls_hint)
            prompt_loss += hint_loss
            return out, prompt_loss
        else:
            return out


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None):
    return ResNetZoo(num_classes=out_dim, prompt_flag=prompt_flag, prompt_param=prompt_param)


def vit_pt_td(out_dim, block_division=None, prompt_flag='None', prompt_param=None):
    return TopDownZoo(num_classes=out_dim, prompt_flag=prompt_flag, prompt_param=prompt_param)
