import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default='runs/test_mlp')
    parser.add_argument("--name", default='test_mlp')

    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--display", default=False, type=bool)
    parser.add_argument('--forward_looking', action='store_true')

    parser.add_argument("--ndim", default=6, type=int)
    parser.add_argument("--nop", default=5, type=int)


    # Plottings
    parser.add_argument("--save_figs", default='./figs', type=str)
    parser.add_argument("--info_file", default='test', type=str)
    parser.add_argument("--log_dir", default='logs', type=str)


    # sampling experiment
    parser.add_argument("--forward_temperature", default=1.0, type=float)
    parser.add_argument("--forward_epsilon", default=0.0, type=float)
    parser.add_argument("--backward_temperature", default=1.0, type=float)
    parser.add_argument("--backward_epsilon", default=0.0, type=float)
    parser.add_argument("--backward_augment", action='store_true')
    parser.add_argument("--KL_weight", default=0.0, type=float)
    parser.add_argument("--CE_weight", default=0.0, type=float)
    parser.add_argument("--loss_type", default='TB', type=str, choices=['FM', 'TB', 'DB', 'subTB'])
    parser.add_argument("--valid_hp", default='200', type=str)
    parser.add_argument("--train_hp", default='12', type=str)

    parser.add_argument('--forward_sample_round', default=50, type=int)
    parser.add_argument('--forward_sample_per_round', default=10, type=int)
    parser.add_argument('--forward_sample_init_round', default=64, type=int)
    parser.add_argument('--forward_sample_trial', default=100, type=int)
    parser.add_argument("--train_n_steps", default=200, type=int)
    parser.add_argument("--train_n_steps_init", default=2000, type=int)

    parser.add_argument("--backward_augment_per_terminal", default=20, type=int)
    parser.add_argument("--backward_augment_training_steps", default=20, type=int)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument('--proxy', action='store_true')

    return parser