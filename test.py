# 改自 main.py

import copy
import glob
import os
import time
from collections import deque
import h5py

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    all_episode_rewards = []   ### 记录 6/29
    all_temp_rewards = []   ### 记录 6/29
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print('num_updates ', num_updates)
    print('num_steps ', args.num_steps)
    count = 0
    h5_path = './data/' + args.env_name
    if not os.path.exists(h5_path):
        os.makedirs(h5_path)
    h5_filename = h5_path + '/trajs_'+args.env_name+'_%05d.h5'%(count)
    data = {}
    data['states'] = []
    data['actions'] = []
    data['rewards'] = []
    data['done'] = []
    data['lengths'] = []
    
    episode_step = 0
    
    for j in range(num_updates):   ### num-steps

        temp_states = []
        temp_actions = []
        temp_rewards = []
        temp_done = []
        temp_lenthgs = []
        
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            if j==0 and step ==0:
                print('obs ',type(rollouts.obs[step]),rollouts.obs[step].shape)
                print('hidden_states ',
                      type(rollouts.recurrent_hidden_states[step]),rollouts.recurrent_hidden_states[step].shape)
                print('action ',type(action), action.shape)
                print('action prob ',type(action_log_prob),action_log_prob.shape)
                print('-'*20)
                
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            
            #print(infos)
            #print(reward)
            temp_states += [np.array(rollouts.obs[step].cpu())]
            temp_actions += [np.array(action.cpu())]
            #temp_rewards += [np.array(reward.cpu())]
            temp_rewards += [np.array([infos[0]['myrewards']])]  ### for halfcheetah不能直接用 reward ！！ 6/29
            temp_done += [np.array(done)]
            
            if j==0 and step ==0:
                print('obs ',type(obs), obs.shape)
                print('reward ',type(reward), reward.shape)
                print('done ',type(done), done.shape)
                print('infos ',len(infos))
                for k,v in infos[0].items():
                    print(k,v.shape)
                print()
            

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    all_episode_rewards += [info['episode']['r']]  ### 记录 6/29

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        temp_lengths = len(temp_states)
        temp_states = np.concatenate(temp_states)
        temp_actions = np.concatenate(temp_actions)
        temp_rewards = np.concatenate(temp_rewards)
        temp_done = np.concatenate(temp_done)
        #print('temp_lengths',temp_lengths)
        #print('temp_states', temp_states.shape)
        #print('temp_actions', temp_actions.shape)
        #print('temp_rewards', temp_rewards.shape)
        if j > int(0.4*num_updates):
            data['states'] += [temp_states]
            data['actions'] += [temp_actions]
            data['rewards'] += [temp_rewards]
            data['lengths'] += [temp_lengths]
            data['done'] += [temp_done]
            #print('temp_lengths',data['lengths'].shape)
            #print('temp_states', data['states'].shape)
            #print('temp_actions', data['actions'].shape)
            #print('temp_rewards', data['rewards'].shape)
            

            if args.save_expert and len(data['states']) >= 100:
                with h5py.File(h5_filename, 'w') as f:
                    f['states'] = np.array(data['states'])
                    f['actions'] = np.array(data['actions'])
                    f['rewards'] = np.array(data['rewards'])
                    f['done'] = np.array(data['done'])
                    f['lengths'] = np.array(data['lengths'])
                    #print('f_lengths',f['lengths'].shape)
                    #print('f_states', f['states'].shape)
                    #print('f_actions', f['actions'].shape)
                    #print('f_rewards', f['rewards'].shape)
                
                count += 1
                h5_filename = h5_path + '/trajs_'+args.env_name+'_%05d.h5'%(count)
                data['states'] = []
                data['actions'] = []
                data['rewards'] = []
                data['done'] = []
                data['lengths'] =[]
                
                
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_%d.pt"%(args.seed)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            #np.save(os.path.join(save_path, args.env_name+"_%d"%(args.seed)), all_episode_rewards)  ### 保存记录 6/29
            #print(temp_rewards)
            print("temp rewards size", temp_rewards.shape,
                  "mean",np.mean(temp_rewards),"min",np.min(temp_rewards),
                  "max",np.max(temp_rewards))
            all_temp_rewards += [temp_rewards]
            np.savez(os.path.join(save_path, args.env_name+"_%d"%(args.seed)),
                     episode=all_episode_rewards, timestep=all_temp_rewards)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


    '''data['states'] = np.array(data['states'])
    data['actions'] = np.array(data['actions'])
    data['rewards'] = np.array(data['rewards'])
    data['done'] = np.array(data['done'])
    data['lengths'] = np.array(data['lengths'])
    if args.save_expert:
        with h5py.File(h5_filename, 'w') as f:
            f['states'] = data['states']
            f['actions'] = data['states']
            f['rewards'] = data['rewards']
            f['done'] = data['done']
            f['lengths'] = data['lengths']'''
    
            
if __name__ == "__main__":
    main()
