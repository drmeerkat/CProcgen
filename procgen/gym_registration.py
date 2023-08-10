from gymnasium import logger
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
from gym3.interop import multimap, np
from gym3 import ToGymEnv, ViewerWrapper, ExtractDictObWrapper
from .env import ENV_NAMES, ProcgenGym3Env
from collections import deque


class ToGymEnvFrameStack(ToGymEnv):
    """
    This is adopted from gym3's ToGymEnv!

    Notes:
        * The `render()` method does nothing in "human" mode, in "rgb_array" mode the info dict is checked
            for a key named "rgb" and info["rgb"][0] is returned if present
        * `seed()` and `close() are ignored since gym3 environments do not require these methods
        * `reset()` is ignored if used before an episode is complete because gym3 environments
            reset automatically, if `reset()` was called before the end of an episode, a warning is printed
    
    :param env: gym3 environment to adapt

    ML: Further adopted from raylib's framestack implementation
    """

    def __init__(self, env, render_mode, k=2, sep=1):
        super().__init__(env)
        self.render_mode = render_mode
        assert render_mode in ['rgb_array', 'human'] and 'render_mode should be either rgb_array or human!'
        self.metadata = {
            'render_modes':['rgb_array', 'human'],
            'render_fps': 50
        }
        # number of frames to be stacked
        self.k = k
        self.sep = sep
        if self.k == 1:
            self.sep = 1
        self.frames = deque([], maxlen=k*sep)
        single_obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=single_obs_shape[:2] + (k * single_obs_shape[2],),
            dtype=self.observation_space.dtype
        )

    def reset(self, seed=None, options=None):
        _rew, ob, first = self.env.observe()
        if not first[0]:
            print("Warning: early reset ignored")
        # The same as initial ob
        prev_ob = ob
        info = self.env.get_info()[0]
        info['prev_ob'] = multimap(lambda x: x[0], prev_ob)
        cur_ob = multimap(lambda x: x[0], ob)
        for _ in range(self.k*self.sep):
            self.frames.append(cur_ob)
        # return multimap(lambda x: x[0], ob), info
        # eg: 64, 64, 3 -> 64, 64, k*3
        return np.concatenate([self.frames[i] for i in range(len(self.frames) - 1, -1, -self.sep)], axis=2), info
        
    def step(self, ac):
        _, prev_ob, _ = self.env.observe()
        self.env.act(np.array([ac]))
        rew, ob, first = self.env.observe()
        info = self.env.get_info()[0]
        if first[0]: # equivalent to done signal
            ob = prev_ob
            # if self.render_mode == "rgb_array" and "rgb" in info:
                # if a human player is using this, need to render the last frame!
                # seems to be a gym3 level problem...they are not rendering the last frame
                # pass            
        info['prev_ob'] = multimap(lambda x: x[0], prev_ob)
        self.frames.append(multimap(lambda x: x[0], ob))
        # return multimap(lambda x: x[0], ob), rew[0], bool(info['gameterm']), bool(info['truncated']), info
        return np.concatenate([self.frames[i] for i in range(len(self.frames) - 1, -1, -self.sep)], axis=2), rew[0], bool(info['gameterm']), bool(info['truncated']), info

    def render(self):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        if self.render_mode == 'human':
            logger.warn('Procgen doesn\'t support \'human\' render mode currently as its backend is gym3! So, calling render(mode=\'human\') has no effect.')
        info = self.env.get_info()[0]
        if self.render_mode == "rgb_array" and "rgb" in info:
            # this is a hi-resolution one! human perferred view
            return info["rgb"]


def make_env(render_mode=None, render=False, **kwargs):
    # the render option is kept here for backwards compatibility
    # users should use `render_mode="human"` or `render_mode="rgb_array"`
    if render:
        render_mode = "human"

    use_viewer_wrapper = False
    kwargs["render_mode"] = render_mode
    if render_mode == "human":
        # procgen does not directly support rendering a window
        # instead it's handled by gym3's ViewerWrapper
        # procgen only supports a render_mode of "rgb_array"
        use_viewer_wrapper = True
        kwargs["render_mode"] = "rgb_array"

    k = kwargs.pop('k', 2)
    sep = kwargs.pop('sep', 1)
    env = ProcgenGym3Env(num=1, num_threads=0, **kwargs)
    env = ExtractDictObWrapper(env, key="rgb")
    if use_viewer_wrapper:
        env = ViewerWrapper(env, tps=15, info_key="rgb")
    gym_env = ToGymEnvFrameStack(env, render_mode, k=k, sep=sep)
    return gym_env


def register_environments():
    for env_name in ENV_NAMES:
        register(
            id=f'procgen-{env_name}-v0',
            entry_point='procgen.gym_registration:make_env',
            kwargs={"env_name": env_name},
        )

# quick test code
# python -c "import gymnasium as gym; import procgen; env = gym.make('procgen:procgen-conchaser-v0', render_mode='rgb_array'); _ = env.reset(); obs = env.render(); import matplotlib.pyplot as plt; plt.figure('initial obs');plt.imshow(obs); env.step(1);obs2 = env.render();plt.figure('step obs'); plt.imshow(obs2);plt.show()"