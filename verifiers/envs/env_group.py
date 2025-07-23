import logging
from typing import List, Dict, Tuple, Union

from datasets import concatenate_datasets
from openai import AsyncOpenAI

from verifiers import (
    Environment,
    ChatMessage,
    Info,
    State,
    SamplingArgs,
    Rubric,
)


class EnvGroupRubric(Rubric):
    """
    Custom rubric for EnvGroup that routes scoring to appropriate environment rubrics.
    """
    
    def __init__(self, env_map: Dict[str, Environment]):
        super().__init__()
        self.env_map = env_map
        self.logger = logging.getLogger("verifiers.envs.EnvGroupRubric")
        
        # Collect all unique reward function names across all environments
        all_names_set = set()
        for env_name, env in env_map.items():
            env_reward_names = env.rubric.get_reward_func_names()
            all_names_set.update(env_reward_names)
            self.logger.debug(f"Environment '{env_name}' has reward functions: {env_reward_names}")
        self.all_reward_names = sorted(list(all_names_set))
        
        self.logger.info(f"EnvGroupRubric tracking {len(self.all_reward_names)} unique reward functions: {self.all_reward_names}")
    
    def get_reward_func_names(self) -> List[str]:
        """Return all unique reward function names across all environments."""
        return self.all_reward_names
    
    async def score_rollout(self,
                            prompt: Union[str, List[ChatMessage]],
                            completion: Union[str, List[ChatMessage]],
                            answer: str = "",
                            state: State = {},
                            task: str = "default",
                            info: dict = {},
                            **kwargs) -> Dict[str, float]:
        """
        Route scoring to the appropriate environment's rubric based on task.
        
        Returns a dict with all reward function names, using 0.0 for functions
        not applicable to this sample's environment.
        """
        self.logger.debug(f"Scoring rollout for task='{task}'")
        # Initialize results with all reward names set to 0.0
        results = {name: 0.0 for name in self.all_reward_names}
        results['reward'] = 0.0
         
        # Get the appropriate environment
        env = self.env_map.get(task)
        if env is None:
            self.logger.warning(f"No environment found for task '{task}', available tasks: {list(self.env_map.keys())}")
            return results
        
        # Score with the environment's rubric
        self.logger.debug(f"Delegating scoring to environment '{task}'")
        env_results = await env.rubric.score_rollout(
            prompt, completion, answer, state, task, info, **kwargs
        )
        self.logger.debug(f"Environment '{task}' returned results: keys={list(env_results.keys())}, reward={env_results.get('reward', 'N/A')}")
        
        # Update results with scores 
        for reward_name, score in env_results.items():
            if reward_name in results:
                results[reward_name] = score
                self.logger.debug(f"Setting {reward_name}={score} from environment '{task}'")
        # dummy scores for all reward functions not in the environment
        for reward_name in self.all_reward_names:
            if reward_name not in env_results:
                results[reward_name] = 0.0
        return results


class EnvGroup(Environment):
    """
    Environment group that acts as a mixture of multiple environments.
    
    Routes operations to appropriate sub-environments based on the 'task' column.
    """
    
    def __init__(self, 
                 envs: List[Environment], 
                 env_names: List[str] | None = None,
                 **kwargs):
        """
        Initialize EnvGroup with a list of environments.
        
        Args:
            envs: List of Environment instances
            env_names: Optional list of names for each environment. 
                      If not provided, uses "env_0", "env_1", etc.
            **kwargs: Additional arguments passed to parent Environment
        """
        if not envs:
            raise ValueError("EnvGroup requires at least one environment")
        
        self.envs = envs
        self.env_names = env_names or [f"env_{i}" for i in range(len(envs))]
        
        if len(self.env_names) != len(self.envs):
            raise ValueError("Number of env_names must match number of envs")
        
        # Create mapping for quick lookup
        self.env_map = {name: env for name, env in zip(self.env_names, self.envs)}
        
        # concatenate datasets with task labels
        datasets = []
        eval_datasets = []
        for env, name in zip(self.envs, self.env_names):
            def add_task(example):
                example['task'] = name
                return example
            env_dataset = env.get_dataset()
            if env_dataset is not None and 'task' not in env_dataset.column_names:
                env_dataset = env_dataset.map(add_task)
            if env_dataset is not None:
                datasets.append(env_dataset)
            env_eval_dataset = env.get_eval_dataset()
            if env_eval_dataset is not None and 'task' not in env_eval_dataset.column_names:
                env_eval_dataset = env_eval_dataset.map(add_task)
            if env_eval_dataset is not None:        
                eval_datasets.append(env_eval_dataset)
        dataset = concatenate_datasets(datasets) if datasets else None
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
        # wrap rubrics
        rubric = EnvGroupRubric(self.env_map)

        # initialize parent Environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            **kwargs
        ) 
        self.logger.info(f"Initialized EnvGroup with {len(envs)} environments: {self.env_names}")
        self.logger.debug(f"Dataset sizes: train={len(dataset) if dataset else 0}, eval={len(eval_dataset) if eval_dataset else 0}")
    
    async def rollout(self,
                      client: AsyncOpenAI,
                      model: str,
                      prompt: Union[str, List[ChatMessage]],
                      answer: str = "",
                      task: str = "default",
                      info: Info = {},
                      sampling_args: SamplingArgs = {},
                      **kwargs) -> Tuple[Union[str, List[ChatMessage]], State]:
        """
        Route rollout to the appropriate sub-environment based on task.
        
        The task is determined from (in order of priority):
        1. kwargs['task']
        2. info['task']  
        3. First environment name (default)
        """
        self.logger.debug(f"Routing rollout to task='{task}'")
        # Route to appropriate environment
        env = self.env_map[task]

        # Pass through all arguments
        self.logger.debug(f"Delegating rollout to environment '{task}' ({env.__class__.__name__})")
        return await env.rollout(client, model, prompt, answer, task, info, sampling_args, **kwargs)

    def get_env_for_task(self, task: str) -> Environment:
        """Get the environment instance for a given task name."""
        env = self.env_map.get(task, self.envs[0])
        self.logger.debug(f"Getting environment for task='{task}': {env.__class__.__name__}")
        return env 
        