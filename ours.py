import pdb
import os.path as osp
import pickle
import shutil
import tempfile
import datetime

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.runner import build_optimizer, build_runner
import cv2

from IPython import embed
from mmseg.ops import resize


import numpy as np
import kornia
import torch
import random
import torch.nn as nn

IMG_MEAN = np.array((104.00698793, 116.66876762,
                    122.67891434), dtype=np.float32)


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def single_gpu_our(model,
                   data_loader,
                   show=False,
                   out_dir=None,
                   efficient_test=False,
                   anchor=None,
                   ema_model=None,
                   anchor_model=None,
                   learning_rate=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:  
        list: The prediction results.
    """
    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./cotta/"+str(datetime.datetime.now())
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            print(name)
        else:
            param.requires_grad = False
    # Batchsize=1 now, was 8 during cityscapes training
    optimizer = torch.optim.Adam(param_list, lr=learning_rate, betas=(0.9, 0.999))
    selected_params = []# same for each domain
    for i, data in enumerate(data_loader):
        model.eval()
        ema_model.eval()
        anchor_model.eval()
        with torch.no_grad():
            result, probs, preds = ema_model(return_loss=False, **data)
            _, probs_, _ = anchor_model(return_loss=False, **data)
            probs_np = np.array(probs_) # 14 1 1080 1920
            confidence = probs_np[4][0]
            average_probs = np.mean(probs_np, axis=0)

            uncertainty = 0
            for j in range(4, 10):
                if j % 2 == 0:
                    uncertainty += probs_np[j][0]
            uncertainty /= 3
            assert(uncertainty.shape == confidence.shape)

            mask = (confidence > 0.69).astype(np.int64)
            result = [(mask*preds[4][0] + (1.-mask)
                       * result[0]).astype(np.int64)]
            weight = 1.
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)
        if isinstance(result, list):
            if len(data['img']) == 14:
                img_id = 4  # The default size without flip
            else:
                img_id = 0
            loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas']
                                 [img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            mask = (uncertainty < 0.85).astype(np.int64)
            mask=torch.from_numpy(mask).cuda()
            masked_loss=mask*loss["decode.loss_seg"][0]
            mask2 = (uncertainty > 0.99).astype(np.int64)
            mask2=torch.from_numpy(mask2).cuda()
            masked_loss2=mask2*loss["decode.loss_seg"][0]
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        masked_consistency_loss = torch.mean(weight * masked_loss)
        masked_consistency_loss.backward(retain_graph=True)
        if i==0:
            selected_params=[]
            selected_layers=[]
        if i<100:
            grad_sum_per_layer = {}  

            for name, param in model.named_parameters():
                if param.grad is not None:  
                    assert(name not in grad_sum_per_layer)
                    grad_sum_per_layer[name] = param.grad.abs().sum().item()

            sorted_layers = sorted(grad_sum_per_layer.keys(), key=lambda x: grad_sum_per_layer[x], reverse=True)
            k_percent = 0.001
            num_layers_to_select = int(len(sorted_layers) * k_percent)
            for layer_name in sorted_layers[:num_layers_to_select]:
                already=False
                for selected in selected_layers:
                    if layer_name==selected:
                        already=True
                        break
                if already==False:
                    selected_params.extend([param for name, param in model.named_parameters() if layer_name in name])
                    selected_layers.append(layer_name)

        optimizer.zero_grad()
        consistency_loss = torch.mean(weight * masked_loss2)
        consistency_loss.backward()
        if i<100:
            grad_sum_per_layer = {}  

            for name, param in model.named_parameters():
                if param.grad is not None:  
                    layer_name = name
                    assert(layer_name not in grad_sum_per_layer)
                    grad_sum_per_layer[layer_name] = param.grad.abs().sum().item()

            sorted_layers = sorted(grad_sum_per_layer.keys(), key=lambda x: grad_sum_per_layer[x], reverse=True)

            k_percent = 0.001
            num_layers_to_select = int(len(sorted_layers) * k_percent)
            
            for layer_name in sorted_layers[:num_layers_to_select]:
                already=False
                for selected in selected_layers:
                    if layer_name==selected:
                        already=True
                        break
                if already==False:
                    selected_params.extend([param for name, param in model.named_parameters() if layer_name in name])
                    selected_layers.append(layer_name)

        optimizer = torch.optim.Adam(selected_params, lr=learning_rate, betas=(0.9, 0.999))

        optimizer.step()
        optimizer.zero_grad()
        ema_model = update_ema_variables(
            ema_model=ema_model, model=model, alpha_teacher=0.999)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_tent(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.requires_grad and ("norm" in name or "bn" in name):
                param_list.append(param)
                print(name)
            else:
                param.requires_grad = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            loss = model.forward(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=torch.from_numpy(
                result[0]).cuda().unsqueeze(0).unsqueeze(0))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            loss = model(
                return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=result)
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        torch.mean(loss["decode.loss_seg"]).backward()

        grad_parameters = [param for param in param_list if param.grad is not None]

        sorted_parameters = sorted(
            grad_parameters, key=lambda param: param.grad.norm() if param.grad is not None else 0, reverse=True)
        k_percentage = 0.1  
        num_selected = int(len(sorted_parameters) * k_percentage)
        selected_parameters = sorted_parameters[:num_selected]
        optimizer = torch.optim.Adam(
            selected_parameters, lr=3e-4, betas=(0.9, 0.999))

        optimizer.step()
        optimizer.zero_grad()

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    out_dir = "./baseline/"+str(datetime.datetime.now())
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
