"""R1: Stefan's LeKiwi-style robot (SO-101 arm, omni base, Jetson Orin Nano), wrist Orbbec Gemini 305, on the shared Embodiment interface.

The AgileX recipe (embodiments/agilex: Gemma boxes the object, the wrist depth camera measures it) with R1's wrist camera: the Orbbec
looks past the jaws, colour and depth aligned in software. Perception only for now: `locate` measures the object in the wrist
camera's optical frame (x right, y down, z forward, metres), and `pick` refuses before moving. A base-frame target needs the arm's
joint state at capture and a hand-eye calibration, and motion needs R1's arm interface (Stefan's seraphos stack owns the servos).

The camera is shared with Stefan's `r1-wristcam` user service, which holds its depth nodes while it runs; stop it first (README).
The SDK runs on its V4L2 backend from our own config copy with the device auto-reboot off (`ob-config.xml`, made by README's setup),
because R1 has no Orbbec udev rule for libusb.

  .venv/bin/python -B run.py -e r1 --typed --opt ob_config=~/r1-gemma/ob-config.xml     # look, choose, locate; nothing moves

Options (--opt key=value): ob_config (the SDK config, required), min_height (0.008 m: a point counts as the object when it stands
this far above the floor plane fitted around the box), ring (0.5: the floor ring around the box, as a fraction of the box size)."""
import os, time
from core.embodiment import Embodiment, Frame, Refused, Target


class R1(Embodiment):
    name = "r1"
    capabilities = dict(Embodiment.capabilities, depth=True, mic=False, speaker=False, detector=False, grip_sensor=False, mobile=True)

    def __init__(self, ob_config=None, min_height=0.008, ring=0.5, **options):
        self.ob_config, self.min_height, self.ring = ob_config and os.path.expanduser(ob_config), min_height, ring
        self.pipe = self.np = None

    # --- lifecycle --------------------------------------------------------------------------------------------------------
    def start(self):
        import numpy as np, pyorbbecsdk as ob
        if not self.ob_config or not os.path.exists(self.ob_config):
            raise RuntimeError("--opt ob_config=<path>: the Orbbec config with LinuxUVCBackend V4L2 and auto-reboot off (README)")
        self.np, self.ob = np, ob
        self.ctx = ob.Context(self.ob_config)              # kept: a freed context frees the device manager ("NULL pointer deviceMgr")
        dev = self.ctx.query_devices().get_device_by_index(0); info = dev.get_device_info()
        self.dev_info = "%s %s fw %s" % (info.get_name(), info.get_serial_number(), info.get_firmware_version())
        self.pipe, cfg = ob.Pipeline(dev), ob.Config()
        for st in (ob.OBSensorType.COLOR_SENSOR, ob.OBSensorType.DEPTH_SENSOR):
            cfg.enable_stream(self.pipe.get_stream_profile_list(st).get_default_video_stream_profile())
        cfg.set_align_mode(ob.OBAlignMode.SW_MODE)        # depth resampled onto the colour pixels
        try: self.pipe.start(cfg)
        except Exception as e:
            raise RuntimeError("Orbbec busy (%s): stop Stefan's r1-wristcam service first (README)" % e)
        prm = self.pipe.get_camera_param(); ci = prm.rgb_intrinsic
        self.K = (ci.fx, ci.fy, ci.cx, ci.cy)
        tr = prm.transform                                 # depth -> colour: p_colour = R p_depth + t (t in mm)
        self.d2c = (np.array(tr.rot, float).reshape(3, 3), np.array(tr.transform, float) / 1000.0)
        for _ in range(15): self.pipe.wait_for_frames(1000)   # auto exposure settles

    def shutdown(self):
        if self.pipe is not None: self.pipe.stop(); self.pipe = None

    # --- senses -----------------------------------------------------------------------------------------------------------
    def frame(self):
        import cv2
        np = self.np
        for _ in range(30):
            fs = self.pipe.wait_for_frames(1000)
            c, d = (fs.get_color_frame(), fs.get_depth_frame()) if fs is not None else (None, None)
            if c is None or d is None: continue
            w, h, raw, f = c.get_width(), c.get_height(), np.frombuffer(c.get_data(), np.uint8), str(c.get_format())
            if "YUYV" in f: img = cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
            elif "MJPG" in f: img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            elif "RGB" in f: img = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            else: img = raw.reshape(h, w, -1)[:, :, :3].copy()
            depth = np.frombuffer(d.get_data(), np.uint16).reshape(d.get_height(), d.get_width()).astype(np.float32) * d.get_depth_scale() / 1000.0
            return Frame(img, depth=depth, meta={"K": self.K, "camera": "wrist Orbbec"}, t=time.time())
        raise RuntimeError("no frames from the wrist Orbbec")

    # --- geometry ---------------------------------------------------------------------------------------------------------
    def to_depth_frame(self, xyz):
        """A colour-camera point (what locate returns: depth aligned to colour) in the depth camera's frame, the frame Stefan's
        hand-eye (camera_extrinsics.yaml) was fitted in."""
        R, t = self.d2c
        return R.T @ (self.np.asarray(xyz, float) - t)

    def rect(self, shape, x1, y1, x2, y2):
        """A boolean pixel mask of a rectangle, clipped to the image."""
        m = self.np.zeros(shape, bool); H, W = shape
        m[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))] = True
        return m

    def points(self, depth, K, mask):
        """Camera-frame points (N x 3, metres) of the valid depth pixels in a mask, and the valid fraction."""
        np = self.np; fx, fy, cx, cy = K
        v, u = np.nonzero(mask & (depth > 0)); z = depth[v, u]
        return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1), len(z) / max(1, mask.sum())

    def locate(self, frame, box, desc=None):
        """The object in `box`: the floor plane fitted to a ring of depth around the box, the box's points that stand above it, their
        centroid. A flat object (nothing above the plane) is measured where its box meets the plane. Camera frame (module docstring)."""
        np, depth, K = self.np, frame.depth, frame.meta["K"]
        H, W = depth.shape; x1, y1, x2, y2 = box; name = desc or "that"
        if min(x1, y1, W - 1 - x2, H - 1 - y2) <= 2:
            raise Refused("The %s one is at the edge of my view. Please move it a little toward my centre." % name)
        box_m = self.rect(depth.shape, x1, y1, x2, y2)
        inner, valid = self.points(depth, K, box_m)
        if valid < 0.2: raise Refused("I cannot measure the %s one: my depth camera sees too little of it." % name)
        bw, bh = (x2 - x1) * self.ring, (y2 - y1) * self.ring
        ring, _ = self.points(depth, K, self.rect(depth.shape, x1 - bw, y1 - bh, x2 + bw, y2 + bh) & ~box_m)
        if len(ring) < 50: raise Refused("I cannot see the floor around the %s one." % name)
        A = np.c_[ring[:, 0], ring[:, 2], np.ones(len(ring))]          # floor: y = a x + b z + c (y points down)
        coef, *_ = np.linalg.lstsq(A, ring[:, 1], rcond=None)
        for _ in range(2):                                             # refit on the ring points close to the plane
            r = ring[:, 1] - A @ coef; keep = np.abs(r) < max(0.005, 2 * np.median(np.abs(r)))
            coef, *_ = np.linalg.lstsq(A[keep], ring[keep, 1], rcond=None)
        s = np.sqrt(1 + coef[0] ** 2 + coef[1] ** 2)
        n = np.array([-coef[0], 1.0, -coef[1]]) / s                    # unit normal, pointing down (into the floor)
        height = (np.c_[inner[:, 0], inner[:, 2], np.ones(len(inner))] @ coef - inner[:, 1]) / s   # above the plane, metres
        r = (ring[:, 1] - A @ coef) / s
        noise = 1.4826 * float(np.median(np.abs(r - np.median(r))))    # the floor's depth noise around the box (robust sigma)
        cut = max(self.min_height, 3 * noise); up = height > cut; obj = inner[up]
        if len(obj) >= max(30, 0.05 * len(inner)):                     # bare floor: a few noisy points pass, never 5% of the box
            xyz, top, flat = np.median(obj, 0), float(np.percentile(height[up], 95)), False
        else:
            xyz, top, flat = np.median(inner, 0), 0.0, True
        span = obj if not flat else inner
        width = float(np.percentile(span[:, 0], 95) - np.percentile(span[:, 0], 5))
        return Target([round(float(v), 4) for v in xyz], box=box, desc=desc,
                      data={"frame": "wrist camera optical (x right, y down, z forward)", "height_m": round(top, 4),
                            "width_m": round(width, 4), "flat": flat, "points": int(len(span)), "depth_valid": round(float(valid), 2),
                            "cut_m": round(cut, 4),
                            "floor_normal": [round(float(v), 3) for v in n]})

    # --- skills (not yet: perception only) --------------------------------------------------------------------------------
    def pick(self, target):
        raise Refused("I can see the %s one, but my arm is not connected to me yet." % (target.desc or "that"))

    def release(self): pass
    def home(self): pass
    def extra(self): return {"camera": getattr(self, "dev_info", None), "K": getattr(self, "K", None)}


def make(**options): return R1(**options)
