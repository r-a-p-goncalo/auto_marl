

from automarl.utils.shapes_util import gym_to_gymnasium_space, gymnasium_to_gym_space
from huggingface_sb3 import load_from_hub
from automarl.components.loggers.global_logger import globalWriteLine
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

import torch



def model_checkpoint_from_model_name(model_name: str) -> str:
    checkpoint = load_from_hub(
        repo_id=f"sb3/{model_name}",
        filename=f"{model_name}.zip",
    )
    return checkpoint


def data_from_model_name(model_name: str):
    checkpoint = model_checkpoint_from_model_name(model_name)

    # Load the state_dict and metadata without creating a full DQN object
    data, params, pytorch_variables = load_from_zip_file(checkpoint)

    return data, params, pytorch_variables



def load_policy_from_class_and_data(model_class, observation_space, action_space, policy_kwargs, params=None):

    policy = model_class(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,  # dummy,
        **policy_kwargs
    )

    if params != None:

        # Load weights into the policy
        policy.load_state_dict(params["policy"], strict=True)


    return policy



def load_policy_from_class_and_architecture(model_class, architecture, params=None):

    return load_policy_from_class_and_data(model_class, architecture["observation_space"], architecture["action_space"], architecture["policy_kwargs"], params)




def load_policy_network_from_architecture(architecture, params=None):

    if architecture["type"] == "dqn":
        q_policy = load_policy_from_class_and_architecture(DQNPolicy, architecture, params)
    
        return load_dqn_net_from_policy(q_policy)

    elif architecture["type"] == "ppo-actor" or architecture["type"] == "ppo":
        ppo_policy = load_policy_from_class_and_architecture(PPOPolicy, architecture, params)

        return load_ppo_actor_net_from_policy(ppo_policy)
    
    elif architecture["type"] == "ppo-critic":
        ppo_policy = load_policy_from_class_and_architecture(PPOPolicy, architecture, params)

        return load_ppo_critic_net_from_policy(ppo_policy)
    
    else:
        raise ValueError(f"Unsupported SB3 model type {architecture['type']}")
    


def load_policy_from_model_name_and_class(model_name: str, model_class):
    
    data, params, pytorch_variables = data_from_model_name(model_name)

    architecture = {
            "observation_space": gym_to_gymnasium_space(data["observation_space"]),
            "action_space": gym_to_gymnasium_space(data["action_space"]),
            "policy_kwargs": data["policy_kwargs"],
            }

    # Rebuild just the policy (with the right architecture)
    policy = load_policy_from_class_and_architecture(model_class, architecture, params)
    
    return policy, architecture


def load_dqn_net_from_policy(policy : DQNPolicy):

    return policy.q_net
    

def load_sb3_q_net(model_name: str):

    policy, architecture = load_policy_from_model_name_and_class(model_name, DQNPolicy)

    architecture["type"] = "dqn"

    return load_dqn_net_from_policy(policy), architecture


def load_ppo_actor_net_from_policy(policy: PPOPolicy):
    #this is done to ignore the critic, we only want the actor for inference
    actor_layers = torch.nn.Sequential(
            policy.features_extractor, #encodes features into input for policy net
            policy.mlp_extractor.policy_net,
            policy.action_net,
        )
    return actor_layers


def load_ppo_critic_net_from_policy(policy: PPOPolicy):

    """
    Loads the critic (value) network from a PPO policy.
    Returns a torch.nn.Module suitable for computing value estimates.
    """
    critic_layers = torch.nn.Sequential(
        policy.features_extractor,  # shared feature extractor
        policy.mlp_extractor.value_net,  # critic-specific MLP
        policy.value_net  # final linear layer producing scalar value
    )
    return critic_layers



def load_sb3_ppo_model(model_name : str):


    policy, architecture = load_policy_from_model_name_and_class(model_name, PPOPolicy)
    
    architecture["type"] = "ppo-actor"

    return load_ppo_actor_net_from_policy(policy), architecture


def load_sb3_ppo_critic(model_name: str):
    """
    Loads only the critic network from a PPO SB3 checkpoint.
    Returns a torch.nn.Module and architecture info.
    """
    policy, architecture = load_policy_from_model_name_and_class(model_name, PPOPolicy)
    architecture["type"] = "ppo-critic"
    return load_ppo_critic_net_from_policy(policy), architecture



def load_sb3_net(model_name: str):
    
    """
    Loads only the neural network policy from a SB3 checkpoint (DQN or PPO).
    Returns the relevant torch.nn.Module and architecture info.
    """

    globalWriteLine(f"Loading sb3 model with name: {model_name}")

    if model_name.startswith("dqn-"):
        return load_sb3_q_net(model_name)
    
    elif model_name.startswith("ppo-"):

        if model_name.endswith("-critic"):
            return load_sb3_ppo_critic(model_name.removesuffix('-critic'))

        elif model_name.endswith("-actor"):
            return load_sb3_ppo_critic(model_name.removesuffix('-actor'))


        else:
            return load_sb3_ppo_model(model_name)
    else:
        raise ValueError(f"Unsupported SB3 model type in name '{model_name}'")