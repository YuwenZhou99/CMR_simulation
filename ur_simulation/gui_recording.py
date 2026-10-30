"""Record the simulation's OpenGL scene using the current GUI camera."""
import json
from pathlib import Path
import shutil
import subprocess
import os
import numpy as np
import pybullet as p

class GUIRecorder:
    def __init__(self, sim, filename, fps=25):
        self.sim = sim
        self.path = Path(filename)
        if self.path.exists():
            raise FileExistsError(f'Video already exists; choose another --record path: {self.path}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.frames = fps, 0
        exe = shutil.which('ffmpeg')
        if not exe:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                bundled = Path(__file__).resolve().parents[1]/'outputs/animation_capture/deps/imageio_ffmpeg/binaries'
                exe = next(bundled.glob('ffmpeg*.exe'), None)
        if not exe:
            raise RuntimeError('Recording needs FFmpeg. Install: python -m pip install imageio-ffmpeg')
        self.error_log = self.path.with_suffix('.ffmpeg.log').open('wb')
        try:
            self.process = subprocess.Popen([
                str(exe), '-hide_banner', '-loglevel', 'error', '-n',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720',
                '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except Exception:
            self.error_log.close()
            raise

    def capture(self, simulation_time):
        if simulation_time + 1e-9 < self.frames/self.fps:
            return
        pb = self.sim.physics_client
        camera = pb.getDebugVisualizerCamera()
        if camera[0] <= 0 or camera[1] <= 0:
            raise RuntimeError('Recording requires an initialized GUI camera')
        projection = list(camera[3])
        projection[0] *= (camera[0]/camera[1])/(1280/720)
        pixels = pb.getCameraImage(1280, 720, viewMatrix=camera[2],
                                   projectionMatrix=projection, shadow=1,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.asarray(pixels[2], dtype=np.uint8).reshape(720, 1280, 4)[:, :, :3].copy()
        # Camera-image sky is white by default; match the configured GUI background.
        sky = np.asarray(pixels[4]).reshape(720, 1280) == -1
        rgb[sky] = np.clip(np.rint(self.sim.background_color*255), 0, 255).astype(np.uint8)
        self.process.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.error_log.close()
        if code or not self.frames:
            raise RuntimeError(f'Video encoding failed or no frames captured; see {self.path.with_suffix(".ffmpeg.log")}')
        self.path.with_suffix('.video.json').write_text(json.dumps(dict(
            method='OpenGL scene render using GUI camera, with configured background color',
            fps=self.fps, frames=self.frames, playback_seconds=self.frames/self.fps,
            resolution=[1280, 720], clock='simulation time'), indent=2), encoding='utf8')
        print(f'Video saved: {self.path}')
