# src/gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
from DQN.dqn import main as train_model_main
from DQN.training_utils import load_model
from Common.game_env import create_env, record_env, create_single_env, record_fsm_env
from FSM.fsm import FSMAgent
from FSM.fsm_utils import run_fsm_performance
from stable_baselines3 import DQN
import numpy as np
from tkinter import ttk


class BreakoutAIApp:
    def __init__(self, root):
        """
        Initialize the AI training interface application.
        
        Parameters:
        root (Tk): The root window of the Tkinter application.
        """
        self.root = root
        self.root.title("AI Playing Games Training Interface")
        # List of Atari games available in OpenAI Gym
        self.env_options = ["Breakout-v4", "AirRaid-v4", "Alien-v4", "Amidar-v4", "Assault-v4", "Asterix-v4", "Asteroids-v4",
                            "Atlantis-v4", "BankHeist-v4", "BattleZone-v4", "BeamRider-v4", "Berzerk-v4",
                            "Bowling-v4", "Boxing-v4", "Carnival-v4", "Centipede-v4",
                            "ChopperCommand-v4", "CrazyClimber-v4", "Defender-v4", "DemonAttack-v4",
                            "DoubleDunk-v4", "ElevatorAction-v4", "Enduro-v4", "FishingDerby-v4",
                            "Freeway-v4", "Frostbite-v4", "Gopher-v4", "Gravitar-v4", "Hero-v4",
                            "IceHockey-v4", "Jamesbond-v4", "JourneyEscape-v4", "Kangaroo-v4",
                            "Krull-v4", "KungFuMaster-v4", "MontezumaRevenge-v4", "MsPacman-v4",
                            "NameThisGame-v4", "Phoenix-v4", "Pitfall-v4", "Pong-v4", "PrivateEye-v4",
                            "Qbert-v4", "Riverraid-v4", "RoadRunner-v4", "Robotank-v4", "Seaquest-v4",
                            "Skiing-v4", "Solaris-v4", "SpaceInvaders-v4", "StarGunner-v4",
                            "Tennis-v4", "TimePilot-v4", "Tutankham-v4", "UpNDown-v4",
                            "Venture-v4", "VideoPinball-v4", "WizardOfWor-v4", "YarsRevenge-v4",
                            "Zaxxon-v4"]
        # Create and initialize the GUI widgets
        self.create_widgets()
        
        self.env = None
        self.fsm_agent = None
        self.monitoring = False
        
    def create_widgets(self):
        """
        Create and set up the GUI widgets for the application.
        """
        # DQN training section
        self.training_frame = tk.LabelFrame(self.root, text="Train DQN Model", padx=10, pady=10)
        self.training_frame.pack(fill="both", expand="yes", padx=10, pady=5)
        # Label for choosing the game environment
        self.env_label = tk.Label(self.training_frame, text="Choose Game Environment:")
        self.env_label.grid(row=0, column=0, padx=5, pady=5)
        # Dropdown menu for selecting the game environment
        self.env_var = tk.StringVar(self.training_frame)
        self.env_var.set(self.env_options[0])
        self.env_menu = ttk.Combobox(self.training_frame, textvariable=self.env_var, values=self.env_options, state="readonly")
        self.env_menu.grid(row=0, column=1, padx=5, pady=5)
        # Label and entry for specifying the number of timesteps
        self.timesteps_label = tk.Label(self.training_frame, text="Timesteps:")
        self.timesteps_label.grid(row=1, column=0, padx=5, pady=5)
        self.timesteps_entry = tk.Entry(self.training_frame)
        self.timesteps_entry.insert(0, "50000")
        self.timesteps_entry.grid(row=1, column=1, padx=5, pady=5)
        # Label and entry for specifying the number of vector stacks
        self.vectorstacks_label = tk.Label(self.training_frame, text="Vector Stacks:")
        self.vectorstacks_label.grid(row=1, column=2, padx=5, pady=5)
        self.vectorstacks_entry = tk.Entry(self.training_frame)
        self.vectorstacks_entry.insert(0, "4")
        self.vectorstacks_entry.grid(row=1, column=3, padx=5, pady=5)
        # Label and entry for specifying the number of environments
        self.env_count_label = tk.Label(self.training_frame, text="Number of Environments:")
        self.env_count_label.grid(row=1, column=4, padx=5, pady=5)
        self.env_count_entry = tk.Entry(self.training_frame)
        self.env_count_entry.insert(0, "4")
        self.env_count_entry.grid(row=1, column=5, padx=5, pady=5)
        # Button to choose the directory for saving the model
        self.save_button = tk.Button(self.training_frame, text="Choose Save Directory", command=self.choose_save_dir)
        self.save_button.grid(row=2, column=0, padx=5, pady=5)
        # Button to choose the directory for logging
        self.log_button = tk.Button(self.training_frame, text="Choose Log Directory", command=self.choose_log_dir)
        self.log_button.grid(row=2, column=1, padx=5, pady=5)
        # Button to start training the DQN model
        self.train_button = tk.Button(self.training_frame, text="Train Model", command=self.train_model)
        self.train_button.grid(row=2, column=2, padx=5, pady=5)

        # TensorBoard section
        self.tensorboard_frame = tk.LabelFrame(self.root, text="View DQN Training Results", padx=10, pady=10)
        self.tensorboard_frame.pack(fill="both", expand="yes", padx=10, pady=5)
        # Button to choose the log directory for TensorBoard
        self.log_button = tk.Button(self.tensorboard_frame, text="Choose Log Directory", command=self.choose_log_dir)
        self.log_button.grid(row=0, column=0, padx=5, pady=5)
        # Button to start TensorBoard
        self.tensorboard_button = tk.Button(self.tensorboard_frame, text="Show TensorBoard", command=self.run_tensorboard)
        self.tensorboard_button.grid(row=0, column=1, padx=5, pady=5)
        # Label for displaying the output of TensorBoard
        self.tensorboard_output = tk.Label(self.tensorboard_frame, text="", wraplength=400)
        self.tensorboard_output.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Monitor DQN section
        self.monitor_frame = tk.LabelFrame(self.root, text="Monitor DQN Agent", padx=10, pady=10)
        self.monitor_frame.pack(fill="both", expand="yes", padx=10, pady=5)
        # Button to choose the model file
        self.model_button = tk.Button(self.monitor_frame, text="Choose Model File", command=self.choose_model_file)
        self.model_button.grid(row=0, column=0, padx=5, pady=5)
        # Label and dropdown menu to choose the game environment for monitoring
        self.monitor_env_label = tk.Label(self.monitor_frame, text="Choose Game Environment:")
        self.monitor_env_label.grid(row=0, column=1, padx=5, pady=5)
        self.monitor_env_var = tk.StringVar(self.monitor_frame)
        self.monitor_env_var.set(self.env_options[0])
        self.monitor_env_menu = ttk.Combobox(self.monitor_frame, textvariable=self.monitor_env_var, values=self.env_options, state="readonly")
        self.monitor_env_menu.grid(row=0, column=2, padx=5, pady=5)
        # Label and entry for specifying the number of environments
        self.env_count_label = tk.Label(self.monitor_frame, text="Number of Environments:")
        self.env_count_label.grid(row=1, column=0, padx=5, pady=5)
        self.env_count_entry = tk.Entry(self.monitor_frame)
        self.env_count_entry.insert(0, "4")
        self.env_count_entry.grid(row=1, column=1, padx=5, pady=5)
        # Label and entry for specifying the number of vector stacks
        self.vectorstacks_label = tk.Label(self.monitor_frame, text="Vector Stacks:")
        self.vectorstacks_label.grid(row=1, column=2, padx=5, pady=5)
        self.vectorstacks_entry = tk.Entry(self.monitor_frame)
        self.vectorstacks_entry.insert(0, "4")
        self.vectorstacks_entry.grid(row=1, column=3, padx=5, pady=5)
        # Label and entry for specifying the recording time
        self.recording_time_label = tk.Label(self.monitor_frame, text="Recording Time (seconds):")
        self.recording_time_label.grid(row=1, column=4, padx=5, pady=5)
        self.recording_time_entry = tk.Entry(self.monitor_frame)
        self.recording_time_entry.insert(0, "60")
        self.recording_time_entry.grid(row=1, column=5, padx=5, pady=5)
        # Button to start monitoring the DQN agent
        self.monitor_button = tk.Button(self.monitor_frame, text="Monitor Agent", command=self.start_monitoring_thread)
        self.monitor_button.grid(row=2, column=0, columnspan=6, padx=5, pady=5)
        
        # FSM Section
        self.fsm_frame = tk.LabelFrame(self.root, text="FSM Agent", padx=10, pady=10)
        self.fsm_frame.pack(fill="both", expand="yes", padx=10, pady=5)
        # Button to select the Python FSM file
        self.select_python_button = tk.Button(self.fsm_frame, text="Select Python FSM File", command=self.select_python_file)
        self.select_python_button.grid(row=0, column=0, padx=5, pady=5)
        # Button to save the FSM agent as a Pickle file
        self.save_fsm_button = tk.Button(self.fsm_frame, text="Save FSM as Pickle", command=self.save_fsm_agent)
        self.save_fsm_button.grid(row=0, column=1, padx=5, pady=5)
        # Button to select the Pickle FSM file
        self.select_pickle_button = tk.Button(self.fsm_frame, text="Select Pickle FSM File", command=self.select_pickle_file)
        self.select_pickle_button.grid(row=1, column=0, padx=5, pady=5)
        # Label and dropdown menu to choose the game environment for monitoring
        self.fsm_env_label = tk.Label(self.fsm_frame, text="Choose Game Environment:")
        self.fsm_env_label.grid(row=1, column=1, padx=5, pady=5)
        self.fsm_env_var = tk.StringVar(self.fsm_frame)
        self.fsm_env_var.set(self.env_options[0])
        self.fsm_env_menu = ttk.Combobox(self.fsm_frame, textvariable=self.fsm_env_var, values=self.env_options, state="readonly")
        self.fsm_env_menu.grid(row=1, column=2, padx=5, pady=5)
        # Label and entry for specifying the number of environments
        self.fsm_env_count_label = tk.Label(self.fsm_frame, text="Number of Environments:")
        self.fsm_env_count_label.grid(row=2, column=0, padx=5, pady=5)
        self.fsm_env_count_entry = tk.Entry(self.fsm_frame)
        self.fsm_env_count_entry.insert(0, "1")
        self.fsm_env_count_entry.grid(row=2, column=1, padx=5, pady=5)
        # Label and entry for specifying the recording time
        self.fsm_recording_time_label = tk.Label(self.fsm_frame, text="Recording Time (seconds):")
        self.fsm_recording_time_label.grid(row=2, column=2, padx=5, pady=5)
        self.fsm_recording_time_entry = tk.Entry(self.fsm_frame)
        self.fsm_recording_time_entry.insert(0, "60")
        self.fsm_recording_time_entry.grid(row=2, column=3, padx=5, pady=5)
        # Button to start monitoring the FSM agent
        self.monitor_fsm_button = tk.Button(self.fsm_frame, text="Monitor FSM", command=self.start_fsm_monitoring_thread)
        self.monitor_fsm_button.grid(row=3, column=0, columnspan=4, padx=5, pady=5)
        # Button to measure the performance of the FSM agent
        self.performance_fsm_button = tk.Button(self.fsm_frame, text="Measure FSM Performance", command=self.measure_fsm_performance)
        self.performance_fsm_button.grid(row=4, column=0, columnspan=4, padx=5, pady=5)
        # Label for displaying the performance of the FSM agent
        self.fsm_performance_output = tk.Label(self.fsm_frame, text="", wraplength=400)
        self.fsm_performance_output.grid(row=5, column=0, columnspan=4, padx=5, pady=5)

    # Train DQN Agent Section
    def choose_save_dir(self):
        """
        Allow the user to select a directory for saving the trained DQN model.
        """
        self.save_dir = filedialog.askdirectory()
        if self.save_dir:
            messagebox.showinfo("Save Directory Selected", f"Save directory: {self.save_dir}")

    def choose_log_dir(self):
        """
        Allow the user to select a directory for storing training logs.
        """
        self.log_dir = filedialog.askdirectory()
        if self.log_dir:
            messagebox.showinfo("Log Directory Selected", f"Log directory: {self.log_dir}")

    def train_model(self):
        """
        Start the training process for the DQN model in a separate thread.
        """
        env_name = self.env_var.get()
        save_dir = self.save_dir if hasattr(self, 'save_dir') else './Training/Saved_Models/'
        log_dir = self.log_dir if hasattr(self, 'log_dir') else './Training/Logs/'
        timesteps = int(self.timesteps_entry.get())
        vectorstacks = int(self.vectorstacks_entry.get())
        env_count = int(self.env_count_entry.get())
        threading.Thread(target=train_model_main, args=(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)).start()
        
    # TensorBoard Section
    def run_tensorboard(self):
        """
        Start TensorBoard in a separate thread to visualize training logs.
        """
        if hasattr(self, 'log_dir'):
            tb_thread = threading.Thread(target=self._run_tensorboard)
            tb_thread.start()
        else:
            messagebox.showwarning("Log Directory Not Selected", "Please select a log directory first.")

    def _run_tensorboard(self):
        """
        Run the TensorBoard process and display the output URL in the application.
        """
        try:
            process = subprocess.Popen(["tensorboard", "--logdir", self.log_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            self.tensorboard_output.config(text=f"TensorBoard is running at: http://localhost:6006\n{out.decode('utf-8')}")
        except Exception as e:
            self.tensorboard_output.config(text=f"Error running TensorBoard: {e}")

    # Monitor DQN Section
    def choose_model_file(self):
        """
        Allow the user to select the trained DQN model file for monitoring.
        """
        self.model_file = filedialog.askopenfilename()
        if self.model_file:
            messagebox.showinfo("Model File Selected", f"Model file: {self.model_file}")

    def start_monitoring_thread(self):
        """
        Start a separate thread to monitor the DQN agent.
        """
        self.monitoring_thread = threading.Thread(target=self.monitor_agent, daemon=True)
        self.monitoring_thread.start()

    def monitor_agent(self):
        """
        Monitor the performance of the trained DQN agent and record a video of its gameplay.
        """
        env_name = self.monitor_env_var.get()
        model_file = self.model_file
        env_count = int(self.env_count_entry.get())
        vectorstacks = int(self.vectorstacks_entry.get())
        recording_time = int(self.recording_time_entry.get())

        if model_file:
            self.env = create_env(environment_name=env_name, n_envs=env_count, n_stack=vectorstacks)
            self.model = load_model(env=self.env, model_path=model_file)
            video_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
            if video_path:
                self.record_agent(video_path, recording_time)
        else:
            messagebox.showwarning("Model File Not Selected", "Please select a model file first.")

    def record_agent(self, video_path, recording_time):
        """
        Record the gameplay of the DQN agent and save it as a video file.
        """
        record_env(self.env, self.model, video_path, recording_time=recording_time)
        messagebox.showinfo("Recording Complete", f"Video saved to: {video_path}")
    
    # FSM Section
    def select_python_file(self):
        """
        Allow the user to select a Python file for the FSM agent.
        """
        self.python_file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
        if self.python_file_path:
            messagebox.showinfo("Python File Selected", f"Python file selected: {self.python_file_path}")

    def save_fsm_agent(self):
        """
        Save the FSM agent as a pickle file.
        """
        if hasattr(self, 'python_file_path') and self.python_file_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
            if save_path:
                fsm_agent = FSMAgent()
                fsm_agent.save(save_path)
                messagebox.showinfo("FSM Saved", f"FSM agent saved to: {save_path}")
        else:
            messagebox.showwarning("No Python File", "Please select a Python file first.")

    def select_pickle_file(self):
        """
        Allow the user to select a pickle file for the FSM agent.
        """
        self.pickle_file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if self.pickle_file_path:
            messagebox.showinfo("Pickle File Selected", f"Pickle file selected: {self.pickle_file_path}")

    def start_fsm_monitoring_thread(self):
        """
        Start a separate thread to monitor the FSM agent.
        """
        self.monitoring_thread = threading.Thread(target=self.monitor_fsm_agent, daemon=True)
        self.monitoring_thread.start()

    def monitor_fsm_agent(self):
        """
        Monitor the performance of the FSM agent and record a video of its gameplay.
        """
        if hasattr(self, 'pickle_file_path') and self.pickle_file_path:
            env_name = self.fsm_env_var.get()
            env_count = int(self.fsm_env_count_entry.get())
            recording_time = int(self.fsm_recording_time_entry.get())
            self.env = create_single_env(env_name, n_envs=env_count)
            self.fsm_agent = FSMAgent.load(self.pickle_file_path)
            video_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
            if video_path:
                self.record_fsm_agent(video_path, recording_time)
        else:
            messagebox.showwarning("No Pickle File", "Please select a Pickle file first.")

    def record_fsm_agent(self, video_path, recording_time):
        """
        Record the gameplay of the FSM agent and save it as a video file.
        """
        record_fsm_env(self.env, self.fsm_agent, video_path, recording_time=recording_time)
        messagebox.showinfo("Recording Complete", f"Video saved to: {video_path}")

    def measure_fsm_performance(self):
        """
        Measure the performance of the FSM agent and display the results.
        """
        if hasattr(self, 'pickle_file_path') and self.pickle_file_path:
            env_name = self.fsm_env_var.get()
            performance = run_fsm_performance(self.pickle_file_path, env_name)
            self.fsm_performance_output.config(text=f"FSM Performance: {performance}")
        else:
            messagebox.showwarning("No Pickle File", "Please select a Pickle file first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BreakoutAIApp(root)
    root.mainloop()
