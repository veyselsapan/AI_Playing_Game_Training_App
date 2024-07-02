# src/gui.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
from DQN.dqn import main as train_model_main
from DQN.training_utils import load_model
from Common.game_env import create_env
from stable_baselines3 import DQN
import numpy as np
import time

class BreakoutAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breakout AI Training Interface")

        self.create_widgets()

    def create_widgets(self):
        # Training section
        self.training_frame = tk.LabelFrame(self.root, text="Train Model", padx=10, pady=10)
        self.training_frame.pack(fill="both", expand="yes", padx=10, pady=5)

        self.env_label = tk.Label(self.training_frame, text="Choose Game Environment:")
        self.env_label.grid(row=0, column=0, padx=5, pady=5)
        self.env_var = tk.StringVar(self.training_frame)
        self.env_options = ["Breakout-v4", "Pong-v4", "Adventure-v4", "AirRaid-v4", "Alien-v4", "Amidar-v4", "Assault-v4", 
                            "Asterix-v4", "Asteroids-v4", "Atlantis-v4"]
        self.env_var.set(self.env_options[0])
        self.env_menu = tk.OptionMenu(self.training_frame, self.env_var, *self.env_options)
        self.env_menu.grid(row=0, column=1, padx=5, pady=5)

        self.save_button = tk.Button(self.training_frame, text="Choose Save Directory", command=self.choose_save_dir)
        self.save_button.grid(row=0, column=2, padx=5, pady=5)

        self.train_button = tk.Button(self.training_frame, text="Train Model", command=self.train_model)
        self.train_button.grid(row=0, column=3, padx=5, pady=5)

        # TensorBoard section
        self.tensorboard_frame = tk.LabelFrame(self.root, text="View Training Results", padx=10, pady=10)
        self.tensorboard_frame.pack(fill="both", expand="yes", padx=10, pady=5)

        self.log_button = tk.Button(self.tensorboard_frame, text="Choose Log Directory", command=self.choose_log_dir)
        self.log_button.grid(row=0, column=0, padx=5, pady=5)

        self.tensorboard_button = tk.Button(self.tensorboard_frame, text="Show TensorBoard", command=self.run_tensorboard)
        self.tensorboard_button.grid(row=0, column=1, padx=5, pady=5)

        self.tensorboard_output = tk.Label(self.tensorboard_frame, text="", wraplength=400)
        self.tensorboard_output.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Monitor section
        self.monitor_frame = tk.LabelFrame(self.root, text="Monitor Agent", padx=10, pady=10)
        self.monitor_frame.pack(fill="both", expand="yes", padx=10, pady=5)

        self.model_button = tk.Button(self.monitor_frame, text="Choose Model File", command=self.choose_model_file)
        self.model_button.grid(row=0, column=0, padx=5, pady=5)

        self.monitor_env_label = tk.Label(self.monitor_frame, text="Choose Game Environment:")
        self.monitor_env_label.grid(row=0, column=1, padx=5, pady=5)
        self.monitor_env_var = tk.StringVar(self.monitor_frame)
        self.monitor_env_var.set(self.env_options[0])
        self.monitor_env_menu = tk.OptionMenu(self.monitor_frame, self.monitor_env_var, *self.env_options)
        self.monitor_env_menu.grid(row=0, column=2, padx=5, pady=5)

        self.monitor_button = tk.Button(self.monitor_frame, text="Monitor Agent", command=self.monitor_agent)
        self.monitor_button.grid(row=0, column=3, padx=5, pady=5)

    def choose_save_dir(self):
        self.save_dir = filedialog.askdirectory()
        if self.save_dir:
            messagebox.showinfo("Save Directory Selected", f"Save directory: {self.save_dir}")

    def train_model(self):
        env_name = self.env_var.get()
        save_dir = self.save_dir if hasattr(self, 'save_dir') else './Training/Saved_Models/'
        threading.Thread(target=train_model_main).start()  # Use threading to prevent blocking

    def choose_log_dir(self):
        self.log_dir = filedialog.askdirectory()
        if self.log_dir:
            messagebox.showinfo("Log Directory Selected", f"Log directory: {self.log_dir}")

    def run_tensorboard(self):
        if hasattr(self, 'log_dir'):
            tb_thread = threading.Thread(target=self._run_tensorboard)
            tb_thread.start()
        else:
            messagebox.showwarning("Log Directory Not Selected", "Please select a log directory first.")

    def _run_tensorboard(self):
        try:
            process = subprocess.Popen(["tensorboard", "--logdir", self.log_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            self.tensorboard_output.config(text=f"TensorBoard is running at: http://localhost:6006\n{out.decode('utf-8')}")
        except Exception as e:
            self.tensorboard_output.config(text=f"Error running TensorBoard: {e}")

    def choose_model_file(self):
        self.model_file = filedialog.askopenfilename()
        if self.model_file:
            messagebox.showinfo("Model File Selected", f"Model file: {self.model_file}")
    
    def start_monitoring_thread(self):
        threading.Thread(target=self.monitor_agent, daemon=True).start()

    def monitor_agent(self):
        env_name = self.monitor_env_var.get()
        model_file = self.model_file if hasattr(self, 'model_file') else None
        if model_file:
            self.env = create_env(environment_name=env_name)
            self.model = load_model(env=self.env, model_path=model_file)
            self.obs = self.env.reset()
            self.done = np.array([False])
            self._monitor_agent()
        else:
            messagebox.showwarning("Model File Not Selected", "Please select a model file first.")

    def _monitor_agent(self):
        if not self.done.any():
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, self.done, info = self.env.step(action)
            self.env.render(mode='human')
            self.root.after(600, self._monitor_agent)  # Schedule the next update (10 fps -> 100ms per frame)
        else:
            self.env.close()

    # def monitor_agent(self):
    #     env_name = self.monitor_env_var.get()
    #     model_file = self.model_file if hasattr(self, 'model_file') else None
    #     if model_file:
    #         self.env = create_env(environment_name=env_name)
    #         self.model = load_model(env=self.env, model_path=model_file)
    #         self.obs = self.env.reset()
    #         self.done = False
    #         self._monitor_agent()
    #     else:
    #         messagebox.showwarning("Model File Not Selected", "Please select a model file first.")

    # def _monitor_agent(self):
    #     if not np.any(self.done):  # Use np.any() to check if any element in the done array is True
    #         action, _ = self.model.predict(self.obs, deterministic=True)
    #         self.obs, reward, self.done, info = self.env.step(action)
    #         self.env.render(mode='human')
    #         self.root.after(10, self._monitor_agent)  # Schedule the next update after 10 ms



    # def _monitor_agent(self, env_name, model_file):
    #     env = create_env(environment_name=env_name)
    #     model = load_model(env=env, model_path=model_file)
    #     obs = env.reset()
    #     done = False
    #     while not np.any(done):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #         env.render(mode='human')

if __name__ == "__main__":
    root = tk.Tk()
    app = BreakoutAIApp(root)
    root.mainloop()
