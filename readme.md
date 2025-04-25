# Lunar Lander Deep-Q Trainer

A “set-and-forget” pipeline for continuous LunarLander-v3 reinforcement-learning runs, auto-archiving every 300 episodes with annotated GIFs & MP4 highlights.

⸻

1. Why this repo exists

You can hack on a DQN agent without babysitting the run.
The script:

✔	What it does
🏃	Trains indefinitely, checkpointing every EPISODES_PER_RUN (default = 300)
🗃	Creates numbered run folders & keeps adding (0001_run_<uuid>/, 0002_…)
🎞	Records a GIF every RENDER_EVERY episodes (default = 25)
🟢	Flashes a green check when the lander scores > 200 (safe landing)
📼	Builds two MP4 highlight reels per run:• *_normals_*.mp4 – all recorded episodes• *_highs_*.mp4 – top-10 episodes by reward
💾	Saves / resumes full training state to dqn_model.pkl
📊	Logs losses & rewards to TensorBoard



⸻

2. Project layout

.
├── dqn_best.py          # single-file trainer
├── videos/                # auto-created; contains all run folders
│   ├── 0001_run_<uuid>/   #   GIFs + MP4s for first 300-episode slab
│   ├── 0002_run_<uuid>/   #   next 300, and so on…
│   └── …
├── runs/                  # TensorBoard logs
└── README.md              # you’re reading it



⸻

1. Requirements

Package	Tested version
gymnasium[box2d]	≥ 0.29
torch	≥ 2.0
opencv-python	≥ 4.8
imageio	≥ 2.34
tensorboard	any
ffmpeg (system)	5.x

Install (Linux/macOS):

# System-wide ffmpeg (Ubuntu / Homebrew):
sudo apt install ffmpeg          # Ubuntu
# or
brew install ffmpeg              # macOS

# Python deps:
python -m pip install gymnasium[box2d] torch opencv-python imageio tensorboard



⸻

4. Quick-start

python dqn_lander.py
tensorboard --logdir runs

	•	Training never stops — Ctrl-C to quit; restart later and it resumes automatically.
	•	Watch live MP4s forming inside each videos/000N_run_* folder.

⸻

5. Changing defaults

Edit these constants at the top of dqn_lander.py:

Constant	Purpose	Default
EPISODES_PER_RUN	How many episodes before making a new run folder & MP4s	300
RENDER_EVERY	Record a GIF every N episodes	25
MIN_REPLAY_SIZE	Buffer prefilling before learning begins	1 000
TARGET_UPDATE_FREQ	Sync target network steps	1 000
Learning-rate / epsilon	LR, EPSILON_*	see file



⸻

6. Files generated

Path	What it is
dqn_model.pkl	binary pickle with all torch weights, optimizer, episode counters, epsilon
videos/000N_run_<uuid>/…gif	one per recorded episode
…normals_<uuid>.mp4	concatenation of all GIFs in that run
…highs_<uuid>.mp4	top-10 GIFs (by reward)



⸻

7. TensorBoard tips

tensorboard --logdir runs --bind_all

	•	Scalars → reward : see learning curve
	•	Scalars → loss   : track TD-error convergence

⸻

8. Troubleshooting

Symptom	Fix
box2d-py errors	Ensure gymnasium[box2d] extras installed
“ffmpeg failed”	Confirm ffmpeg in PATH; run ffmpeg -version
No MP4 generated	Check GIFs exist; look at console logs for [ffmpeg] fail / [concat] fail
Train restarts at ep 1	dqn_model.pkl missing / corrupted → delete & start fresh



⸻

9. Next ideas
	•	Double-DQN / Dueling nets
	•	Prioritized replay
	•	Saving best model separately (best_model.pt)
	•	Real-time dashboard (Gradio or Streamlit)

⸻

10. License

MIT — do anything, just credit the original when you do.