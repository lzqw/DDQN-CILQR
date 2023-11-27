import argparse

from RL_method.utils import str2bool


def add_arguments(parser):
    parser.add_argument('--use_esc', action="store_true", default=True,
                        help='None')
    parser.add_argument('--obs_dis', type=float, default=20,
                        help='None')
    parser.add_argument('--obs_num', type=int, default=15,
                        help='None')
    parser.add_argument('--car_dim', type=int, default=4,
                        help='None')

    # ___________________ Carla Parameters ___________________ #
    parser.add_argument('--add_npc_agents', action="store_true", default=False,
                        help='Should there be NPC agents in the simulator')
    parser.add_argument('--verbose', action="store_true", default=False, help='Show debugging data')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')

    parser.add_argument('--number_of_npc', type=int, default=10, help='Number of NPC vehicles')
    parser.add_argument('--camera_width', type=int, default=800, help='Width of the image rendered by the cameras')
    parser.add_argument('--camera_height', type=int, default=800, help='Height of the image rendered by the cameras')

    parser.add_argument('--debug_simulator', action="store_true", default=False,
                        help='Use visualizations in simulator for debugging')

    # ___________________ Planning Parameters ___________________ #
    parser.add_argument('--number_of_local_wpts', type=int, default=20, help='Number of local waypoints')
    parser.add_argument('--poly_order', type=int, default=5, help='Order of the polynomial to fit on')
    parser.add_argument('--use_pid', action="store_true", default=False, help='If we want to use PID instead of iLQR')
    parser.add_argument('--desired_speed', type=float, default=6.0, help='Desired Speed')
    parser.add_argument('--use_mpc', action="store_true", default=False, help='To use or not to use (MPC)')
    parser.add_argument('--mpc_horizon', type=int, default=5, help='For how many timesteps to use MPC')

    # ___________________ iLQR Parameters ___________________ #
    parser.add_argument('--timestep', type=float, default=0.1,
                        help='Timestep at which forward and backward pass are done by iLQR')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Planning horizon for iLQR in num of steps (T=horizon*timesteps)')
    parser.add_argument('--tol', type=float, default=1e-4, help='iLQR tolerance parameter for convergence')
    parser.add_argument('--max_iters', type=int, default=20, help='Total number of iterations for iLQR')
    parser.add_argument('--num_states', type=int, default=4, help='Number of states in the model')
    parser.add_argument('--num_ctrls', type=int, default=2, help='Number of control inputs in the model')

    # ___________________ Cost Parameters ___________________ #

    parser.add_argument('--w_acc', type=float, default=1.00, help="Acceleration cost")
    parser.add_argument('--w_yawrate', type=float, default=3.00, help="Yaw rate cost")
    parser.add_argument('--w_steer', type=float, default=3.00, help="Steer rate cost")

    parser.add_argument('--w_pos', type=float, default=3.0, help="Path deviation cost")
    parser.add_argument('--w_vel', type=float, default=1.0, help="Velocity cost")#0.2

    parser.add_argument('--q1_acc', type=float, default=1.0, help="Barrier function q1, acc")
    parser.add_argument('--q2_acc', type=float, default=1.0, help="Barrier function q2, acc")

    parser.add_argument('--q1_steer', type=float, default=1.0, help="Barrier function q1, steer")
    parser.add_argument('--q2_steer', type=float, default=1.0, help="Barrier function q2, steer")

    parser.add_argument('--q1_yawrate', type=float, default=1.00, help="Barrier function q1, yawrate")
    parser.add_argument('--q2_yawrate', type=float, default=1.00, help="Barrier function q2, yawrate")

    parser.add_argument('--q1_front', type=float, default=2.75, help="Barrier function q1, obs with ego front")
    parser.add_argument('--q2_front', type=float, default=2.75, help="Barrier function q2, obs with ego front")

    parser.add_argument('--q1_rear', type=float, default=2.5, help="Barrier function q1, obs with ego rear")
    parser.add_argument('--q2_rear', type=float, default=2.5, help="Barrier function q2, obs with ego rear")

    # ___________________ Constraint Parameters ___________________ #
    parser.add_argument('--acc_limits', nargs="*", type=float, default=[-5.5, 5.5],
                        help="Acceleration limits for the ego vehicle (min,max)")
    parser.add_argument('--steer_angle_limits', nargs="*", type=float, default=[-1.0, 1.0],
                        help="Steering Angle limits (rads) for the ego vehicle (min,max)")

    # ___________________ Ego Vehicle Parameters ___________________ #
    parser.add_argument('--wheelbase', type=float, default=2.94, help="Ego Vehicle's wheelbase")
    parser.add_argument('--max_speed', type=float, default=80.0, help="Ego Vehicle's max speed")
    parser.add_argument('--const_speed', type=int, default=5, help="Ego Vehicle's supposing constant speed")
    parser.add_argument('--tractor_l', type=int, default=3, help="Distance between front and rear axis of tractor")
    parser.add_argument('--trailer_d', type=int, default=5, help="Distance between front and rear axis of trailer")
    parser.add_argument('--steering_control_limits', nargs="*", type=float, default=[-1.0, 1.0],
                        help="Steering control input limits (min,max)")
    parser.add_argument('--throttle_control_limits', nargs="*", type=float, default=[-1.0, 1.0],
                        help="Throttle control input limits (min,max)")

    # ___________________ Obstacle Parameters ___________________ #
    parser.add_argument('--t_safe', type=float, default=0.1, help="Time safety headway")
    parser.add_argument('--s_safe_a', type=float, default=2.5, help="safety margin longitudinal")
    parser.add_argument('--s_safe_b', type=float, default=2, help="safety margin lateral")
    parser.add_argument('--ego_rad', type=float, default=2, help="Ego Vehicle's radius")
    parser.add_argument('--ego_lf', type=float, default=1.5, help="Distance to front tire")
    parser.add_argument('--ego_lr', type=float, default=1.5, help="Distance to rear tire")

    # ___________________ Map and Track Parameters ___________________ #
    parser.add_argument('--map_name', type=str, default='DR_USA_Intersection_MA')
    parser.add_argument('--track_file_number', type=int, default=0)

    # ___________________ DQN Parameters ___________________ #
    parser.add_argument('--EnvIdex', type=int, default=0, help='my-env, CP-v1, LLd-v2')
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=64000, help='which model to load')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Max_train_steps', type=int, default=1e6, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=2000, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=300, help='Model evaluating interval, in steps.')
    parser.add_argument('--random_steps', type=int, default=2, help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=20, help='training frequency')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='lenth of sliced trajectory')
    parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
    parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
    parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')

    # ______________________Env Parameters_______________________________#
    parser.add_argument('--K', type=int, default=3, help="Number of K-nearest")
    parser.add_argument('--car_dims', nargs="*", type=float, default=[4, 2])
    parser.add_argument('--p_times', type=int, default=5, help="low-controler for p time steps")
    parser.add_argument('--low_controller_num', type=int, default=10, help="low-controller")
    parser.add_argument('--sim_time', type=int, default=500)
    parser.add_argument('--ego_id', type=int, default=121)





