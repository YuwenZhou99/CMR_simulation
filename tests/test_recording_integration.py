import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class RecordingTests(unittest.TestCase):
 def test_rl_capture_precedes_autoreset(self):
  rl=load('rl_control');events=[]
  class Toy(gym.Env):
   observation_space=gym.spaces.Box(-1.,1.,(1,));action_space=gym.spaces.Box(-1.,1.,(1,))
   dt=.02
   def reset(self,seed=None,options=None):
    events.append('reset');return np.zeros(1),{}
   def step(self,a):
    events.append('terminal');return np.ones(1),0.,True,False,{'time':.02}
  class Recorder:
   def capture(self,t):events.append(('capture',t))
  wrapped=rl.RecordingEnv(Toy());wrapped.recorder=Recorder()
  env=DummyVecEnv([lambda:wrapped])
  try:
   env.reset();env.step(np.zeros((1,1)))
   self.assertEqual(events,['reset','terminal',('capture',0.),'reset'])
  finally:env.close()
  self.assertEqual(rl.episode_video_path('',Path('output'),1,3),Path('output/episode_001.mp4'))
  self.assertEqual(rl.episode_video_path('demo.mp4',Path('output'),1,3),Path('demo_001.mp4'))

 def test_classic_recording_lifecycle_and_unchanged_clock(self):
  m=load('classic_control')
  from ur_simulation.pybullet import PyBullet
  from matplotlib.figure import Figure
  class Fast(PyBullet):
   def __init__(self,**kw):super().__init__(**kw,render_mode='rgb_array',renderer='Tiny')
   def step(self,*a,**kw):super().step(realtime=False)
  events=[]
  class Recorder:
   def __init__(self,sim,path):self.sim=sim;events.append('open')
   def capture(self,t):events.append(t)
   def close(self):
    assert self.sim.physics_client.isConnected()
    events.append('close')
  with patch('ur_simulation.classic_control.robots.ur7e.PyBullet',Fast), patch('ur_simulation.gui_recording.GUIRecorder',Recorder), patch.object(Figure,'savefig'), patch.object(m.np,'savez_compressed'), patch.object(m.plt,'show'):
   m.run('point',[.4,.3,.5],record='unused.mp4')
  self.assertEqual(events[0],'open');self.assertEqual(events[-1],'close')
  self.assertEqual(len(events)-2,11000)
  self.assertEqual(events[1],0.);self.assertAlmostEqual(events[-2],10.999)

if __name__=='__main__':unittest.main()
