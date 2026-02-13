#!/usr/bin/env python3
"""
Keyboard control command publisher.

Supports:
1. pynput backend (desktop sessions)
2. stdin backend (SSH/headless terminals)
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
    PYNPUT_IMPORT_ERROR = ""
except Exception as exc:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False
    PYNPUT_IMPORT_ERROR = str(exc)


class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = 0.0
        self._last_value = 0.0

    def update(self, new_value, max_accel=1.5):
        delta = new_value - self._last_value
        delta = np.clip(delta, -max_accel, max_accel)
        filtered = self.alpha * (self._last_value + delta) + (1 - self.alpha) * self._value
        self._last_value = filtered
        self._value = filtered
        return self._value


class KeyboardController:
    def __init__(self, backend: str = "auto"):
        self.control_params = {
            "x_vel": 0.0,
            "y_vel": 0.0,
            "yaw_vel": 0.0,
            "height": 0.0,
        }

        # Key increment step size
        self.increment = 0.05

        # control range
        self.ranges = {
            "x_vel": (-0.6, 1.0),
            "y_vel": (-0.5, 0.5),
            "yaw_vel": (-1.57, 1.57),
            "height": (-0.5, 0.0),
        }

        # key state
        self.key_states = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "z": False,
            "x": False,
            "c": False,
        }
        self._last_key_event_ts = {k: 0.0 for k in self.key_states}
        self._stdin_old_settings = None
        self._stdin_thread = None

        self.backend = self._resolve_backend(backend)
        self.param_lock = threading.Lock()
        self.running = True

        self._filters = {
            "x_vel": LowPassFilter(alpha=0.3),
            "y_vel": LowPassFilter(alpha=0.3),
            "yaw_vel": LowPassFilter(alpha=0.3),
            "height": LowPassFilter(alpha=0.3),
        }

        self._default_values = {
            "x_vel": 0.0,
            "y_vel": 0.0,
            "yaw_vel": 0.0,
            "height": 0.0,
        }

        # Start threads
        self._control_thread = threading.Thread(target=self._control_update)
        self._control_thread.daemon = True
        self._control_thread.start()

        # Start keyboard listener
        self._start_keyboard_listener()

    def _resolve_backend(self, backend: str) -> str:
        if backend not in ("auto", "pynput", "stdin"):
            raise ValueError(f"unsupported backend: {backend}")
        if backend == "pynput":
            if not PYNPUT_AVAILABLE:
                raise RuntimeError(f"pynput not available: {PYNPUT_IMPORT_ERROR}")
            return "pynput"
        if backend == "stdin":
            return "stdin"
        # auto mode
        if PYNPUT_AVAILABLE and os.environ.get("DISPLAY"):
            return "pynput"
        return "stdin"

    def _start_keyboard_listener(self):
        """Start keyboard listener based on selected backend."""
        if self.backend == "pynput":
            self._start_pynput_listener()
        else:
            self._start_stdin_listener()

    def _set_key_state(self, key_char: str, is_pressed: bool):
        if key_char in self.key_states:
            self.key_states[key_char] = is_pressed
            if is_pressed:
                self._last_key_event_ts[key_char] = time.time()

    def _start_pynput_listener(self):
        def on_press(key):
            """Key press event."""
            try:
                key_char = key.char.lower() if hasattr(key, 'char') and key.char else None
                with self.param_lock:
                    if key_char in self.key_states:
                        if not self.key_states[key_char]:
                            self._set_key_state(key_char, True)
                            print(f"[KEY] {key_char.upper()}: press")
                    elif key_char == "q":
                        print("exit program...")
                        self.running = False
                        return False
            except AttributeError:
                pass

        def on_release(key):
            """Key release event."""
            try:
                key_char = key.char.lower() if hasattr(key, 'char') and key.char else None
                with self.param_lock:
                    if key_char in self.key_states:
                        if self.key_states[key_char]:
                            self._set_key_state(key_char, False)
                            print(f"[KEY] {key_char.upper()}: release")
            except AttributeError:
                pass

        self.listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self.listener.start()

        print("keyboard listener started with pynput backend")
        print("press W/A/S/D/Z/X/C keys to control, Q to exit")

    def _start_stdin_listener(self):
        """Start stdin fallback listener for SSH/headless terminals."""
        if not sys.stdin.isatty():
            raise RuntimeError("stdin backend requires an interactive terminal (TTY)")
        self._stdin_fd = sys.stdin.fileno()
        self._stdin_old_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)

        self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._stdin_thread.start()
        print("keyboard listener started with stdin backend")
        print("press W/A/S/D/Z/X/C keys to control, SPACE to reset, Q to exit")

    def _stdin_loop(self):
        while self.running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                continue
            key_char = ch.lower()
            with self.param_lock:
                if key_char in self.key_states:
                    self._set_key_state(key_char, True)
                elif key_char == " ":
                    for k in self.key_states:
                        self._set_key_state(k, False)
                    for k in self.control_params:
                        self.control_params[k] = 0.0
                    print("commands reset")
                elif key_char == "q":
                    print("exit program...")
                    self.running = False
                    return

    def _control_update(self):
        """Control parameter update thread."""
        while self.running:
            with self.param_lock:
                # Stdin backend has no key-release events; emulate release after repeat timeout.
                if self.backend == "stdin":
                    now = time.time()
                    for key_char in self.key_states:
                        if self.key_states[key_char] and (now - self._last_key_event_ts[key_char] > 0.15):
                            self._set_key_state(key_char, False)

                # forward/backward (x_vel)
                if self.key_states["w"]:
                    self.control_params["x_vel"] = min(
                        self.control_params["x_vel"] + self.increment,
                        self.ranges["x_vel"][1],
                    )
                elif self.key_states["s"]:
                    self.control_params["x_vel"] = max(
                        self.control_params["x_vel"] - self.increment,
                        self.ranges["x_vel"][0],
                    )
                else:
                    if self.control_params["x_vel"] > 0:
                        self.control_params["x_vel"] = max(0, self.control_params["x_vel"] - self.increment * 2)
                    elif self.control_params["x_vel"] < 0:
                        self.control_params["x_vel"] = min(0, self.control_params["x_vel"] + self.increment * 2)

                # left/right (y_vel)
                if self.key_states["a"]:
                    self.control_params["y_vel"] = max(
                        self.control_params["y_vel"] - self.increment,
                        self.ranges["y_vel"][0],
                    )
                elif self.key_states["d"]:
                    self.control_params["y_vel"] = min(
                        self.control_params["y_vel"] + self.increment,
                        self.ranges["y_vel"][1],
                    )
                else:
                    if self.control_params["y_vel"] > 0:
                        self.control_params["y_vel"] = max(0, self.control_params["y_vel"] - self.increment * 2)
                    elif self.control_params["y_vel"] < 0:
                        self.control_params["y_vel"] = min(0, self.control_params["y_vel"] + self.increment * 2)

                # left/right rotation (yaw_vel)
                if self.key_states["z"]:
                    self.control_params["yaw_vel"] = max(
                        self.control_params["yaw_vel"] - self.increment,
                        self.ranges["yaw_vel"][0],
                    )
                elif self.key_states["x"]:
                    self.control_params["yaw_vel"] = min(
                        self.control_params["yaw_vel"] + self.increment,
                        self.ranges["yaw_vel"][1],
                    )
                else:
                    if self.control_params["yaw_vel"] > 0:
                        self.control_params["yaw_vel"] = max(0, self.control_params["yaw_vel"] - self.increment * 2)
                    elif self.control_params["yaw_vel"] < 0:
                        self.control_params["yaw_vel"] = min(0, self.control_params["yaw_vel"] + self.increment * 2)

                # crouch (height)
                if self.key_states["c"]:
                    self.control_params["height"] = max(
                        self.control_params["height"] - self.increment,
                        self.ranges["height"][0],
                    )
                else:
                    if self.control_params["height"] < 0:
                        self.control_params["height"] = min(0, self.control_params["height"] + self.increment * 2)

                for key in self.control_params:
                    self.control_params[key] = round(self.control_params[key], 3)

            time.sleep(0.02)

    # === external interface ===

    def get_control_params(self):
        with self.param_lock:
            return self.control_params.copy()

    def get_key_states(self):
        with self.param_lock:
            return self.key_states.copy()
    
    def stop(self):
        """Stop keyboard controller."""
        self.running = False
        if hasattr(self, "listener"):
            self.listener.stop()
        if self._stdin_old_settings is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
            except Exception:
                pass


def publish_reset_category(category, publisher):
    msg = String_(data=str(category))
    publisher.Write(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "pynput", "stdin"], default="auto", help="keyboard input backend")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    args = parser.parse_args()

    print("=" * 50)
    print("keyboard control instructions:")
    print("W: forward    S: backward")
    print("A: left  D: right")
    print("Z: left rotation  X: right rotation")
    print("C: crouch    Q: exit program")
    print("SPACE: reset command (stdin backend)")
    print("backend auto-select: pynput(with DISPLAY) -> stdin")
    print("=" * 50)

    try:
        print("initializing DDS communication...")
        ChannelFactoryInitialize(args.channel)
        publisher = ChannelPublisher("rt/run_command/cmd", String_)
        publisher.Init()
        print("DDS communication initialized")

        print("initializing keyboard controller...")
        keyboard_controller = KeyboardController(backend=args.backend)
        print(f"active backend: {keyboard_controller.backend}")
        default_height = 0.8

        print("=" * 50)
        print("program started, waiting for keyboard input...")
        print("press Ctrl+C to exit program")
        print("=" * 50)

        last_commands = [0.0, 0.0, 0.0, 0.8]

        while keyboard_controller.running:
            time.sleep(0.01)
            commands = keyboard_controller.get_control_params()
            commands["height"] = default_height + commands["height"]

            commands_list = [
                float(commands["x_vel"]),
                -float(commands["y_vel"]),
                -float(commands["yaw_vel"]),
                float(commands["height"]),
            ]
            commands_str = str(commands_list)

            if commands_list != last_commands:
                print(f"commands: {commands_str}")
                last_commands = commands_list.copy()

            publish_reset_category(commands_str, publisher)

    except KeyboardInterrupt:
        print("\nprogram interrupted by user (Ctrl+C)")
        if "keyboard_controller" in locals():
            keyboard_controller.stop()
    except Exception as e:
        print(f"\nprogram error: {e}")
        if "keyboard_controller" in locals():
            keyboard_controller.stop()

    print("program ended")
