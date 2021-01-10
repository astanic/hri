import os
import sys
import copy
import time
import datetime
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

torch.backends.cudnn.benchmark = True

from utils import setup_experiment_dir, copy_git_src_files_to_logdir, \
    add_sacred_log, print_model_size, plot_batch_graphs, render_traj, \
    concat_pred_imgs_and_save, render_traj_and_save
from dataset import d, load_data
from model import m, Model, compute_loss

from sacred import Experiment

ex = Experiment("hri", ingredients=[m, d])


@ex.config
def config():
    results_dir = 'results'  # Top directory for all experimental results.
    exp_dir = os.path.join(results_dir, datetime.datetime.now().isoformat())

    resume_dir = ''  # Checkpoint dir with the pretrained model
    evaluate_only = False

    n_epochs = 50  # Number of epochs to train.
    es_patience = 100  # Stop training if no valid loss improvement

    lr = 0.0005  # Initial learning rate.
    lr_decay_epochs = 200  # After how many steps to decay LR.
    lr_decay_rate = 0.5  # LR decay factor.

    render_traj_test = False  # Flag to render predicted trajectories in test
    overlay_alpha = 0.9
    overlay_eps = 0.1
    overlay_n = 20

    rollout_seq_len = 20

@ex.capture
def log_results(result_dict, epoch, prefix, _run):
    add_sacred_log("%s.epoch" % prefix, int(epoch), _run)
    for key, value in result_dict.items():
        if not isinstance(value[0], np.ndarray):
            # list of scalar values
            add_sacred_log("%s.%s" % (prefix, key), float(np.mean(value)), _run)
        else:
            # list of arrays (for example prediction mse for t=1,2,..,20 steps)
            value = np.array(value)
            add_sacred_log("%s.%s" % (prefix, key),
                           np.mean(value, axis=0).astype(float), _run)

@ex.capture
def train(exp_dir, model, dataloaders, meta_data,
          opt_all, sch_all, opt_img, sch_img, opt_dyn, sch_dyn,
          n_epochs, es_patience, rollout_seq_len,
          overlay_alpha, overlay_eps, overlay_n, _config):

    best_model = copy.deepcopy(model.state_dict())
    val_best_score = np.inf
    val_best_results = None
    val_best_epoch = -1

    valid_pred_imgs_dir = os.path.join(exp_dir, 'valid_pred_imgs')
    valid_pred_imgs_dir_wo_objs = os.path.join(exp_dir, 'valid_pred_imgs_wo_objs')
    valid_pred_imgs_dir_overlay = os.path.join(exp_dir, 'valid_pred_imgs_overlay')
    os.makedirs(valid_pred_imgs_dir, exist_ok=True)
    os.makedirs(valid_pred_imgs_dir_wo_objs, exist_ok=True)
    os.makedirs(valid_pred_imgs_dir_overlay, exist_ok=True)

    do_early_stopping = False
    for epoch in range(n_epochs):
        if do_early_stopping:
            break

        for phase in dataloaders.keys():
            first_batch = True
            epoch_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            results = {
                'loss': [],
                'traj_nll': [],
                'traj_nll_partial': [],
                'sf_traj_nll': [],
                'traj_mse': [],
                # 'traj_pred_mse': [],  # this we don't need in train and valid
                'kl_edge': [],
                'kl_obj': [],
                'edge_acc': [],
                'edge_acc_sparse': [],
                'graph_acc': [],
                'imgs_nll': [],
                'imgs_mse': [],
                'imgs_pred_mse': [],
                'imgs_pred_nll': [],
            }

            model.update_loss_coefficients(epoch)

            for i_batch, batch in enumerate(dataloaders[phase]):
                for key in batch.keys():
                    batch[key] = batch[key].cuda()
                batch['imgs'] = batch['imgs'].float() / 255.0
                batch['imgs'] = batch['imgs'].transpose(-1, -3)
                if opt_img is not None:
                    # Case where we deal only with trajectories
                    opt_img.zero_grad()
                opt_all.zero_grad()

                if phase == 'valid' and first_batch:
                    model.first_batch = True

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward pass
                    outputs = model(batch, epoch)

                    # Calculate the loss
                    loss, loss_report = compute_loss(
                        batch, outputs, meta_data, model)

                    for key in results.keys():
                        results[key].append(loss_report[key])

                    # Backprop
                    if phase == 'train':
                        loss.backward()
                        if model.params_to_optimize(n_epochs) == 'all':
                            opt_all.step()
                        elif model.params_to_optimize(n_epochs) == 'img':
                            opt_img.step()
                        elif model.params_to_optimize(n_epochs) == 'dyn':
                            opt_dyn.step()

                if phase == 'valid' and first_batch:
                    if outputs['imgs_pred'] is not None:
                        # Write the predicted images
                        img_dir = os.path.join(valid_pred_imgs_dir, str(epoch))
                        os.makedirs(img_dir)
                        img_dir_wo = os.path.join(valid_pred_imgs_dir_wo_objs, str(epoch))
                        img_dir_ov = os.path.join(valid_pred_imgs_dir_overlay, str(epoch))
                        os.makedirs(img_dir_wo)
                        os.makedirs(img_dir_ov)

                        concat_pred_imgs_and_save(
                            batch['imgs'][:, 1:, :, :, :], outputs['imgs_pred'],
                            outputs['imgs_pred_objs'],
                            img_dir, rollout_seq_len)
                        concat_pred_imgs_and_save(
                            batch['imgs'][:, 1:, :, :, :], outputs['imgs_pred'],
                            outputs['imgs_pred_objs'],
                            img_dir_wo, rollout_seq_len, draw_objs=False)
                        concat_pred_imgs_and_save(
                            batch['imgs'][:, 1:, :, :, :], outputs['imgs_pred'],
                            outputs['imgs_pred_objs'],
                            img_dir_ov, rollout_seq_len,
                            draw_objs=False, overlay_rollouts=True,
                            overlay_alpha=overlay_alpha, overlay_eps=overlay_eps,
                            overlay_n=overlay_n)

                    first_batch = False
                    model.first_batch = False

            if phase == 'train':
                if model.params_to_optimize(n_epochs) == 'all':
                    sch_all.step()
                elif model.params_to_optimize(n_epochs) == 'img':
                    sch_img.step()
                elif model.params_to_optimize(n_epochs) == 'dyn':
                    sch_dyn.step()

            epoch_t = (time.time() - epoch_time) / 60
            eta = (n_epochs - epoch - 1) * epoch_t / 60
            # Print
            log_results(results, epoch, phase)
            info = "[%d/%d] %s epoch: %.3fmins eta: %.3fhrs" % (
                epoch, n_epochs, phase, epoch_t, eta)
            for key, value in results.items():
                if not isinstance(value[0], np.ndarray):
                    info += " %s: %.3f" % (key, float(np.mean(value)))
            print(info)

            # ES
            if phase == 'valid':
                if model.reset_best_valid_loss(epoch):
                    val_best_score = np.inf

                if np.mean(results['loss']) < val_best_score:
                    # Save the best model or early stopping
                    print('Best model so far, saving...')
                    model.save(exp_dir)
                    best_model = copy.deepcopy(model.state_dict())
                    val_best_score = np.mean(results['loss'])
                    val_best_epoch = epoch
                    val_best_results = results
                elif epoch - val_best_epoch >= es_patience:
                    print('Validation score did not improve for {} epochs. '
                          'Early stopping.'.format(es_patience))
                    do_early_stopping = True
        print()

    log_results(val_best_results, val_best_epoch, "best.valid")
    info = 'Training complete. Best val acc: {:4f}'.format(val_best_score)
    print(info)
    model.load_state_dict(best_model)

    return model, val_best_epoch


@ex.command
def evaluate(exp_dir, model, val_best_epoch,
             test_loader, meta_data, render_traj_test,
             rollout_seq_len,
             overlay_alpha, overlay_eps, overlay_n, seed, _config):

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set up dirs
    eval_graph_dir = os.path.join(exp_dir, 'eval_graphs')
    eval_traj_render_dir = os.path.join(exp_dir, 'eval_traj_render')
    eval_traj_render_overlay_dir = os.path.join(exp_dir, 'eval_traj_render_overlay')
    eval_pred_imgs_dir = os.path.join(exp_dir, 'eval_pred_imgs')
    eval_pred_imgs_dir_wo_objs = os.path.join(exp_dir, 'eval_pred_imgs_wo_objs')
    eval_pred_imgs_dir_overlay = os.path.join(exp_dir, 'eval_pred_imgs_overlay')
    eval_test_pred_imgs_dir = os.path.join(exp_dir, 'eval_test_pred_imgs')
    eval_test_pred_imgs_dir_wo_objs = os.path.join(exp_dir, 'eval_test_pred_imgs_wo_objs')
    eval_test_pred_imgs_dir_overlay = os.path.join(exp_dir, 'eval_test_pred_imgs_overlay')
    os.makedirs(eval_graph_dir, exist_ok=True)
    os.makedirs(eval_traj_render_dir, exist_ok=True)
    os.makedirs(eval_traj_render_overlay_dir, exist_ok=True)
    os.makedirs(eval_pred_imgs_dir, exist_ok=True)
    os.makedirs(eval_pred_imgs_dir_wo_objs, exist_ok=True)
    os.makedirs(eval_pred_imgs_dir_overlay, exist_ok=True)
    os.makedirs(eval_test_pred_imgs_dir, exist_ok=True)
    os.makedirs(eval_test_pred_imgs_dir_wo_objs, exist_ok=True)
    os.makedirs(eval_test_pred_imgs_dir_overlay, exist_ok=True)

    # Load the best model
    model.eval()

    model.first_batch = True
    start_time = time.time()
    results = {
        'loss': [],
        'traj_nll': [],
        'traj_nll_partial': [],
        'sf_traj_nll': [],
        'traj_mse': [],
        'traj_pred_mse': [],
        'kl_edge': [],
        'kl_obj': [],
        'edge_acc': [],
        'edge_acc_sparse': [],
        'graph_acc': [],
        'imgs_nll': [],
        'imgs_mse': [],
        'imgs_pred_mse': [],
        'imgs_pred_nll': [],
    }

    for i_batch, batch in enumerate(test_loader):
        for key in batch.keys():
            batch[key] = batch[key].cuda()
        batch['imgs'] = batch['imgs'].float() / 255.0
        batch['imgs'] = batch['imgs'].transpose(-1, -3)
        # Forward pass
        outputs = model(batch, val_best_epoch, is_test=True)

        # Calculate the loss
        loss, loss_report = compute_loss(batch, outputs, meta_data, model)

        for key in results.keys():
            results[key].append(loss_report[key])

        # Debug rendering stuff
        if model.first_batch:
            # Write the predicted images
            if outputs['imgs_pred'] is not None:
                concat_pred_imgs_and_save(
                    batch['imgs'][:, 1:, :, :, :], outputs['imgs_pred'],
                    outputs['imgs_pred_objs'], eval_pred_imgs_dir,
                    rollout_seq_len)
                concat_pred_imgs_and_save(
                    batch['imgs'][:, 1:, :, :, :], outputs['imgs_pred'],
                    outputs['imgs_pred_objs'],
                    eval_pred_imgs_dir_wo_objs, rollout_seq_len,
                    draw_objs=False)
                concat_pred_imgs_and_save(
                    batch['imgs'][:, 1:, :, :, :], outputs['imgs_pred'],
                    outputs['imgs_pred_objs'],
                    eval_pred_imgs_dir_overlay, rollout_seq_len,
                    draw_objs=False,
                    overlay_rollouts=True, overlay_alpha=overlay_alpha,
                    overlay_eps=overlay_eps, overlay_n=overlay_n)

            # Write the predicted (unrolled - for 20 steps!) images
            if outputs['test_imgs_pred'] is not None:
                concat_pred_imgs_and_save(
                    outputs['test_imgs_target'], outputs['test_imgs_pred'],
                    outputs['test_imgs_pred_objs'],
                    eval_test_pred_imgs_dir, rollout_seq_len)
                concat_pred_imgs_and_save(
                    outputs['test_imgs_target'], outputs['test_imgs_pred'],
                    outputs['test_imgs_pred_objs'],
                    eval_test_pred_imgs_dir_wo_objs, rollout_seq_len,
                    draw_objs=False)
                concat_pred_imgs_and_save(
                    outputs['test_imgs_target'], outputs['test_imgs_pred'],
                    outputs['test_imgs_pred_objs'],
                    eval_test_pred_imgs_dir_overlay, rollout_seq_len,
                    draw_objs=False,
                    overlay_rollouts=True, overlay_alpha=overlay_alpha,
                    overlay_eps=overlay_eps, overlay_n=overlay_n)

            # Render the images from trajectories
            if render_traj_test:
                render_traj_and_save(
                    outputs['traj_target'], outputs["traj_pred"],
                    eval_traj_render_dir,
                    eval_traj_render_overlay_dir,
                    n_children=meta_data["n_children"],
                    rollout_seq_len=rollout_seq_len,
                    last_level_nodes=meta_data["hierarchy_nodes_list"][-1],
                    tedges=batch['edges'],
                    pedges=outputs['latent_edge_samples'],
                    overlay_alpha=overlay_alpha, overlay_eps=overlay_eps,
                    overlay_n=overlay_n)

            # Plot the inferred latent graphs
            if outputs['latent_edge_samples'] is not None:
                plot_batch_graphs(batch['edges'].cpu(),
                                  outputs['latent_edge_samples'].cpu(),
                                  eval_graph_dir,
                                  model.mp_full_adj)
            model.first_batch = False

    # Print
    log_results(results, val_best_epoch, 'test')
    duration = time.time() - start_time

    # Print
    info = "TEST epoch/min: %.3f" % (1. / (duration / 60.))
    for key, value in results.items():
        if not isinstance(value[0], np.ndarray):
            info += " %s: %.3f" % (key, float(np.mean(value)))
        else:
            # list of arrays (for example prediction mse for t=1,2,..,20 steps)
            v = np.array(value.copy())
            v = np.mean(v, axis=0)
            info += " %s: %s" % (key, v)
    print(info)

    log_results(results, val_best_epoch, "best.test")

    return np.mean(results['loss'])


@ex.automain
def main(exp_dir, resume_dir, evaluate_only,
         lr, lr_decay_epochs, lr_decay_rate,
         seed, _config):

    # Save model and meta-data. Always saves in a new sub-folder.
    log_stdout, log_stderr = setup_experiment_dir(exp_dir)

    # Re-rout stdout and stderr - mainly for server runs to store logs and errs
    tmp_stdout = sys.stdout
    tmp_stderr = sys.stderr
    # sys.stdout = open(log_stdout, 'w')
    # sys.stderr = open(log_stderr, 'w')

    # Write config to a json file and copy src files to logs dir
    with open(os.path.join(exp_dir, 'flags.json'), 'w') as outfile:
        json.dump(_config, outfile, indent=4)
    copy_git_src_files_to_logdir(exp_dir)

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Dataset
    train_loader, valid_loader, test_loader, meta_data = load_data()

    # Model
    model = Model(meta_data)

    if resume_dir is not '':
        model_file = os.path.join(resume_dir, 'model_all.pt')
        model.load_state_dict(torch.load(model_file))
        info = 'Loaded pretrained model: {}'.format(model_file)
        print(info)
    model.cuda()

    # Statistics
    print(model)
    print_model_size(model)

    # Train
    val_best_epoch = 0
    if not evaluate_only:
        # Optimizer
        params_model_all = list(model.parameters())
        opt_all = optim.Adam(params_model_all, lr=lr)
        sch_all = lr_scheduler.StepLR(opt_all, step_size=lr_decay_epochs,
                                      gamma=lr_decay_rate)

        params_model_img = list(model.get_image_enc_dec_params())
        opt_img, sch_img = None, None
        if params_model_img:
            opt_img = optim.Adam(params_model_img, lr=lr)
            sch_img = lr_scheduler.StepLR(opt_img, step_size=lr_decay_epochs,
                                          gamma=lr_decay_rate)

        params_model_dyn = list(model.get_dynamics_params())
        opt_dyn, sch_dyn = None, None
        if params_model_dyn:
            opt_dyn = optim.Adam(params_model_dyn, lr=lr)
            sch_dyn = lr_scheduler.StepLR(opt_dyn, step_size=lr_decay_epochs,
                                          gamma=lr_decay_rate)

        dataloaders = {'train': train_loader, 'valid': valid_loader}

        model, val_best_epoch = train(
            exp_dir, model, dataloaders, meta_data,
            opt_all, sch_all, opt_img, sch_img, opt_dyn, sch_dyn)


    # Test
    test_acc = evaluate(exp_dir, model, val_best_epoch, test_loader, meta_data)

    info = '***** Experiment directory: {} *****'.format(exp_dir)
    print(info)

    # Return back the standard handles
    sys.stdout = tmp_stdout
    sys.stderr = tmp_stderr
    return test_acc
