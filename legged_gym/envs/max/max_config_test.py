from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

TYPE = "Max"


class MaxCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        if TYPE == "Max":
            pos = [0.0, 0.0, 0.31]
            offset = 0.2
            default_joint_angles = {  # = target angles [rad] when action = 0.0
                'joint_FL1': 0.2,  # [rad]
                'joint_HL1': 0.2,  # [rad]
                'joint_FR1': -0.2,  # [rad]
                'joint_HR1': -0.2,  # [rad]

                'joint_FL2': -0.8 - offset,  # [rad]
                'joint_HL2': -1.0 - offset,  # [rad]
                'joint_FR2': -0.8 - offset,  # [rad]
                'joint_HR2': -1.0 - offset,  # [rad]

                'joint_FL3': 1.5 + offset,  # [rad]
                'joint_HL3': 1.5 + offset,  # [rad]
                'joint_FR3': 1.5 + offset,  # [rad]
                'joint_HR3': 1.5 + offset,  # [rad]
            }
        else:
            pos = [0.0, 0.0, 0.42]  # x,y,z [m]
            default_joint_angles = {  # = target angles [rad] when action = 0.0
                'FL_hip_joint': 0.1,  # [rad]
                'RL_hip_joint': 0.1,  # [rad]
                'FR_hip_joint': -0.1,  # [rad]
                'RR_hip_joint': -0.1,  # [rad]

                'FL_thigh_joint': 0.8,  # [rad]
                'RL_thigh_joint': 1.,  # [rad]
                'FR_thigh_joint': 0.8,  # [rad]
                'RR_thigh_joint': 1.,  # [rad]

                'FL_calf_joint': -1.5,  # [rad]
                'RL_calf_joint': -1.5,  # [rad]
                'FR_calf_joint': -1.5,  # [rad]
                'RR_calf_joint': -1.5,  # [rad]
            }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        if TYPE == "Max":
            stiffness = {'joint': 50.}  # [N*m/rad]
            damping = {'joint': 0.5}  # [N*m*s/rad]
            # action scale: target angle = actionScale * action + defaultAngle
            action_scale = 0.25 * 0.5  # 0.5
        else:
            stiffness = {'joint': 20.}  # [N*m/rad]
            damping = {'joint': 0.5}  # [N*m*s/rad]
            # action scale: target angle = actionScale * action + defaultAngle
            action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    # class env(LeggedRobotCfg.env):
    #     num_observations = 48
    #
    # class terrain(LeggedRobotCfg.terrain):
    #     mesh_type = 'plane'
    #     measure_heights = False

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.0, 1.0]  # [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

            # lin_vel_x = [1, 1]  # min max [m/s]
            # lin_vel_y = [0.0, 0.0]  # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            # heading = [0, 0]

    class asset(LeggedRobotCfg.asset):
        if TYPE == "Max":
            file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/max/urdf/max.urdf'
            name = "max"
            foot_name = "3"
            penalize_contacts_on = ["1", "2"]
            # terminate_after_contacts_on = ["body", "1", "2"]
            terminate_after_contacts_on = ["body"]
            flip_visual_attachments = False
            fix_base_link = False
            # default_dof_drive_mode = 4
        else:
            file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
            name = "a1"
            foot_name = "foot"
            penalize_contacts_on = ["thigh", "calf"]
            terminate_after_contacts_on = ["base"]
            flip_visual_attachments = True
            fix_base_link = False

        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            if TYPE == "Max":
                # penalize angular velocity in xy axis
                ang_vel_xy = -0.05
                # penalize linear velocity in z axis
                lin_vel_z = -2.0
                # Penalize joint target changes
                action_rate = -0.01
                torques = -0.0002 * 0.5  # 0.5
                feet_air_time = 1.0 * 0.5  # 0.5

                collision = -1.0
                dof_acc = -2.5e-7 * 0.1  # 0.1
                dof_pos_limits = -10.0

                base_height = 0.0
                dof_vel = 0.0
                feet_stumble = 0.0
                orientation = 0.0
                stand_still = 0.0
                termination = 0.0
            else:
                torques = -0.0002
                dof_pos_limits = -10.0

    class viewer:
        ref_env = 0
        # pos = [5, 5, 3]  # [m]
        # lookat = [0., 0., 0.]  # [m]

        pos = [25, 8, 3]  # [m]
        lookat = [20.0, 3.8, 0.0]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200Hz
        substeps = 1


class MaxCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 1.e-3 * 0.1  # 0.1
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_max'
        max_iterations = 1500
