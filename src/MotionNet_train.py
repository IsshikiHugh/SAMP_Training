import os
import os.path as osp
import torch
from torch.nn import functional as F
from data_parser import MotionNetData
from torch.utils.data import DataLoader
import glob
import SAMP_models as models
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from cmd_parser import parse_config
import misc_utils
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    # Set to train mode.
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    autoregressive_count = 0

    # Iterate the data.
    for batch_idx, data in enumerate(train_data_loader):
        # Decode data to variable.
        x1 = data['x1'].to(device)  # previous pose
        x2 = data['x2'].to(device)  # environment
        y  = data['y'].to(device)   # next pose

        for i in range(args.L):
            optimizer.zero_grad()

            p_prev = x1[:, i, :]
            I = x2[:, i, :]
            p = y[:, i, :]

            # TODO: Figure out what is this.
            if i != 0 and Bernoulli.sample().int() == 1:
                p_prev = p_hat
                # Always pass gt goal features
                # 13 here might be the t mentioned @ paper.
                # TODO: but where is c_i?
                p_prev[:, :] = torch.cat((p_prev[:, :-13 * (6 + args.num_actions)],
                                          x1[:, i, -13 * (6 + args.num_actions):]),
                                         dim=-1)
                autoregressive_count += 1

            # Forward pass.
            p_hat, mu, logvar = model(p, p_prev, I)
            
            # Compute loss.
            recon_loss = F.mse_loss(p_hat, p, reduction='sum') / float(args.batch_size)
            # KLD: https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/
            # Related knowledge: Entropy formula of normal distribution
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(args.batch_size)
            # Formula (3) @ paper.
            loss = recon_loss + args.kl_w * kld
            
            # Accumulate loss.
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld.item()
            total_loss += loss.item()
            
            # Backward pass.
            loss.backward()
            # Update parameters.
            optimizer.step()

            # Extract p_hat from the graph.
            p_hat = p_hat.detach()
            # Do scheduled sampling.
            if args.scheduled_sampling and P < 1:
                # TODO: Figure out what's this?
                p_hat = misc_utils.transform_output(p_prev, I, p_hat, input_mean, input_std, output_mean, output_std,
                                                    **args_dict)

    total_recon_loss /= float(n_batches_train * args.L)
    total_kld_loss /= float(n_batches_train * args.L)
    total_loss /= float(n_batches_train * args.L)
    autoregressive_count /= float(n_batches_train * args.L)

    writer.add_scalar('reconstruction/train', total_recon_loss, epoch)
    writer.add_scalar('kld/train', total_kld_loss, epoch)
    writer.add_scalar('total/train', total_loss, epoch)
    writer.add_scalar('scheduled_sampling/autoregressive_count_train', autoregressive_count, epoch)
    print('====> Total_train_loss: {:.4f}, recon_loss: {:.4f}, kld_loss: {:.4f}'.format(total_loss,
                                                                                        total_recon_loss,
                                                                                        total_kld_loss))
    return total_loss


def test():
    # Set to the test mode.
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0

    for batch_idx, data in enumerate(test_data_loader):
        x1 = data['x1'].to(device)  # previous pose
        x2 = data['x2'].to(device)  # environment
        y  = data['y'].to(device)   # next pose

        for i in range(args.L):
            p_prev = x1[:, i, :]
            I = x2[:, i, :]
            p = y[:, i, :]

            if i != 0 and args.scheduled_sampling:
                p_prev = p_hat
                p_prev[:, :] = torch.cat((p_prev[:, :-13 * (6 + args.num_actions)],
                                          x1[:, i, -13 * (6 + args.num_actions):]),
                                         dim=-1)

            # Forward pass.
            p_hat, mu, logvar = model(p, p_prev, I)

            # Calculate loss.
            recon_loss = F.mse_loss(p_hat, p, reduction='sum') / float(args.batch_size)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(args.batch_size)
            loss = recon_loss + args.kl_w * kld
            # Accumulate loss.
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld.item()
            total_loss += loss.item()

            # Extract p_hat from the graph.
            p_hat = p_hat.detach()
            if args.scheduled_sampling:
                p_hat = misc_utils.transform_output(p_prev, I, p_hat, input_mean, input_std, output_mean, output_std,
                                                    **args_dict)

    total_recon_loss /= float(n_batches_test * args.L)
    total_kld_loss /= float(n_batches_test * args.L)
    total_loss /= float(n_batches_test * args.L)

    writer.add_scalar('reconstruction/test', total_recon_loss, epoch)
    writer.add_scalar('kld/test', total_kld_loss, epoch)
    writer.add_scalar('total/test', total_loss, epoch)
    print('====> Total_test_loss: {:.4f}, recon_loss: {:.4f}, kld_loss: {:.4f}'.format(total_loss,
                                                                                       total_recon_loss,
                                                                                       total_kld_loss))


if __name__ == '__main__':
    torch.manual_seed(0)
    from threadpoolctl import threadpool_limits

    # Read in config files.
    with threadpool_limits(limits=1, user_api='blas'):
        args, args_dict = parse_config()

    args_dict['data_dir'] = osp.expandvars(args_dict.get('data_dir'))
    args_dict['output_dir'] = osp.expandvars(args_dict.get('output_dir'))

    # Get training data and output directories.
    data_dir = args_dict.get('data_dir')
    output_dir = args_dict.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    # Write basic output config to file.
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    conf_fn = osp.join(output_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        # delete all None arguments
        conf_data = vars(args)
        keys = [x for x in conf_data.keys() if conf_data[x] is None]
        for key in keys:
            del (conf_data[key])
        yaml.dump(conf_data, conf_file)

    # Set seeds.
    rng = np.random.RandomState(23456)
    args_dict['rng'] = rng

    # Move device.
    device = torch.device("cuda" if args.use_cuda else "cpu")
    # Set data type.
    if args_dict.get('float_dtype') == 'float64':
        dtype = torch.float64
    elif args_dict.get('float_dtype') == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(args_dict.get('float_dtype')))

    # Load models.
    model = models.load_model(**args_dict).to(device)

    # Get the number of the params.
    n_params = models.count_parameters(model) 
    print('Num of trainable param = {:.2f}M, exactly = {}'.format(n_params / (10 ** 6), n_params))
    
    # Get the optimizer (Adam with lr).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Use lambda function to decay the learning rate.
    # Ref: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html?highlight=schedu#torch.optim.lr_scheduler.LambdaLR
    lmbda = lambda epoch: (args.epochs - epoch + 1) / float(args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    epoch = 1
    # Load target checkpoints.
    checkpoints_list = glob.glob(osp.join(args_dict['output_dir'], 'checkpoints', '*.pt'))
    if len(checkpoints_list) == 0:
        print("No check point found, starting training at epoch 1")
    else:
        # Choose which checkpoint to play with.
        if args.load_latest_checkpoint:
            last_checkpoint = sorted(checkpoints_list)[-1]
            print('loading stats of epoch {}'.format(osp.basename(last_checkpoint)))
            checkpoint = torch.load(last_checkpoint)
        elif args.load_checkpoint > 0:
            print('loading stats of epoch {}'.format(args.load_checkpoint))
            checkpoint = torch.load(
                osp.join(osp.join(args_dict['output_dir'], 'checkpoints'),
                         'epoch_{:04d}.pt'.format(args.load_checkpoint)))
        
        # Load sates.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1

    # Prepare some directories and write logs.
    checkpoints_dir = osp.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = osp.join(output_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Params and data of train.
    train_data_set = MotionNetData(device=torch.device("cpu"), train_data=True, **args_dict)
    train_data_loader = DataLoader(train_data_set, batch_size=args_dict.get('batch_size'),
                                   num_workers=args_dict.get('num_workers'), shuffle=args.shuffle,
                                   drop_last=True)
    train_set_size = len(train_data_set)
    n_batches_train = train_set_size // args.batch_size
    print('No of training example: {}, No of batches: {}'.format(train_set_size, n_batches_train))
    
    # Params and data of test.
    if args.test:
        test_data_set = MotionNetData(device=torch.device("cpu"), train_data=False, **args_dict)
        test_data_loader = DataLoader(test_data_set, batch_size=args_dict.get('batch_size'),
                                      num_workers=args_dict.get('num_workers'), shuffle=False,
                                      drop_last=True)
        test_set_size = len(test_data_set)
        n_batches_test = test_set_size // args.batch_size
        print('Number of testing example: {}'.format(test_set_size))

    input_mean = torch.tensor(train_data_set.input_mean, dtype=dtype, device=device)
    input_std = torch.tensor(train_data_set.input_std, dtype=dtype, device=device)
    output_mean = torch.tensor(train_data_set.output_mean, dtype=dtype, device=device)
    output_std = torch.tensor(train_data_set.output_std, dtype=dtype, device=device)

    # Start training.
    P = 1
    for epoch in range(epoch, args.epochs + 1):
        print('Training epoch {}'.format(epoch))
        
        # Get a Bernoulli distribution.
        # P : 1 -> 0  |  1 - P : 0 -> 1
        if args.scheduled_sampling:
            if epoch <= args.C1:
                P = 1
            elif args.C1 < epoch <= args.C2:
                P = 1 - (epoch - args.C1) / float(args.C2 - args.C1)
            else:
                P = 0
                
        # They take the value 1 with probability (1 - P) and 0 with probability P.
        # More and more likely to give 1 as epoch increases.
        Bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(1 - P, dtype=torch.float))
        
        print('p value = {}'.format(P))
        writer.add_scalar('scheduled_sampling/P', P, epoch)

        start = time.time()
        # Main training line.
        total_train_loss = train()
        training_time = time.time() - start

        # If should test, then test.
        if args.test:
            print('Testing epoch {}'.format(epoch))
            start = time.time()
            # Testing program.
            total_test_loss = test()
            testing_time = time.time() - start

        # If the learning rate should adjust, optimized lr with scheduler.
        if args.reduce_lr:
            scheduler.step()
            print('current lr = {}'.format(get_lr(optimizer)))
        writer.add_scalar('lr', get_lr(optimizer), epoch)
        
        # Write log msg.
        print('training_time = {:.4f}'.format(training_time))
        writer.add_scalar('time/training_time', training_time / 60.0, epoch)
        if args.test:
            print('test_time = {:.4f}'.format(testing_time))
            writer.add_scalar('time/testing_time', testing_time / 60.0, epoch)

        # Save data.
        if args.save_checkpoints and epoch % args.log_interval == 0:
            data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'total_train_loss': total_train_loss,
            }
            if args.test:
                data['total_test_loss'] = total_test_loss
            torch.save(data, osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(epoch)))
