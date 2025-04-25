"""
Deep Q‑Network (DQN) trainer for LunarLander‑v3
------------------------------------------------
• Trains indefinitely, checkpointing & archiving every `EPISODES_PER_RUN` episodes
• Creates numbered run folders: 0001_run_<uuid>/ …
• Records GIFs every `RENDER_EVERY` episodes, annotates frames, flashes green check on success
• Builds
    – normals_<uuid>.mp4  (all recorded episodes)
    – highs_<uuid>.mp4    (top‑10 by score)
• Resumes cleanly from `dqn_model.pkl`
• Robust ffmpeg handling with logs

Dependencies: gymnasium[box2d] torch tensorboard opencv‑python imageio ffmpeg
"""

import os, re, uuid, random, subprocess, pickle
from collections import deque
from datetime import datetime

import gymnasium as gym
import cv2, imageio
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ---------------- CONFIG ------------------
GAMMA              = 0.99
LR                 = 1e-3
BATCH_SIZE         = 64
BUFFER_SIZE        = 100_000
MIN_REPLAY_SIZE    = 1_000
TARGET_UPDATE_FREQ = 1_000
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.01, 0.9995

EPISODES_PER_RUN   = 300
RENDER_EVERY       = 25
VIDEO_DIR          = "videos"
MODEL_PATH         = "dqn_model.pkl"

# -------------- HOUSEKEEP ---------------
os.makedirs(VIDEO_DIR, exist_ok=True)
writer = SummaryWriter(f"runs/DQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- ENV ----------------------
make_env = lambda rec=False: gym.make("LunarLander-v3", render_mode="rgb_array" if rec else None)
env = make_env()
OBS_DIM, ACT_DIM = env.observation_space.shape[0], env.action_space.n

# -------------- MODEL --------------------
class DQN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,out))
    def forward(self,x): return self.net(x)

policy_net, target_net = DQN(OBS_DIM,ACT_DIM).to(DEVICE), DQN(OBS_DIM,ACT_DIM).to(DEVICE)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

epsilon, step_cnt, start_ep, total_ep = EPSILON_START, 0, 1, 0
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH,'rb') as f:
        ck=pickle.load(f)
    policy_net.load_state_dict(ck['policy_state']); target_net.load_state_dict(ck['target_state'])
    optimizer.load_state_dict(ck['optimizer_state'])
    epsilon, step_cnt, start_ep, total_ep = ck['epsilon'], ck['step'], ck['episode']+1, ck.get('total_episode',0)
    print(f"[resume] from ep {start_ep}")
else:
    target_net.load_state_dict(policy_net.state_dict())

# -------------- BUFFER -------------------
replay=deque(maxlen=BUFFER_SIZE)
obs,_=env.reset()
for _ in range(MIN_REPLAY_SIZE):
    a=env.action_space.sample(); nobs,r,t,tr,_=env.step(a)
    replay.append((obs,a,r,nobs,t or tr)); obs= nobs if not (t or tr) else env.reset()[0]

# -------------- UTIL ---------------------
annot=lambda fr,sc,ep,land=False: cv2.cvtColor(cv2.putText(cv2.putText(cv2.cvtColor(fr,cv2.COLOR_RGB2BGR),f"Score:{sc:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2),f"Episode:{ep}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) if not land else cv2.circle(cv2.putText(cv2.putText(cv2.cvtColor(fr,cv2.COLOR_RGB2BGR),f"Score:{sc:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2),f"Episode:{ep}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2),(300,150),30,(0,255,0),-1),cv2.COLOR_BGR2RGB)

def save_gif(fr,ep,rdir,tag):
    gid=uuid.uuid4().hex[:6]; p=os.path.join(rdir,f"{tag}_ep{ep:04d}_{gid}.gif"); imageio.mimsave(p,fr,fps=30); return p

def gifs2mp4(gifs, out_path):
    """
    Convert a list of GIFs -> single MP4 via ffmpeg concat.
    Prints detailed logs if anything goes wrong.
    """
    if not gifs:
        print(f"[i] gifs2mp4 skipped – no GIFs given for {out_path}")
        return

    # 1) Convert each GIF → MP4
    mp4_files = []
    for gif in gifs:
        mp4 = os.path.splitext(gif)[0] + ".mp4"
        try:
            print(f"[ffmpeg] GIF → MP4  {os.path.basename(gif)}")
            subprocess.run(
                ["ffmpeg", "-y", "-i", gif,
                 "-vf", "scale=600:-2", "-pix_fmt", "yuv420p", mp4],
                check=True    # raise CalledProcessError on failure
            )
            mp4_files.append(mp4)
        except subprocess.CalledProcessError as e:
            print(f"[!] ffmpeg failed for {gif}\n{e}")

    if not mp4_files:
        print(f"[!] No MP4s produced — cannot build {out_path}")
        return

    # 2) concat list
    concat_list = out_path + "_list.txt"
    with open(concat_list, "w") as f:
        for mp in mp4_files:
            f.write(f"file '{os.path.abspath(mp)}'\n")

    # 3) ffmpeg concat
    try:
        print(f"[ffmpeg] Concatenating {len(mp4_files)} clips → {os.path.basename(out_path)}")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", out_path],
            check=True
        )
        if os.path.isfile(out_path):
            print(f"[✓] MP4 created: {out_path}")
        else:
            print(f"[!] concat reported success but file missing: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"[!] concat failed for {out_path}\n{e}")

# -------------- TRAIN LOOP ---------------
while True:
    run_idx=max([int(m.group(1)) for d in os.listdir(VIDEO_DIR) if (m:=re.match(r"(\d{4})_run_",d))],default=0)+1
    run_tag=f"{run_idx:04d}"; run_uuid=uuid.uuid4().hex[:8]
    run_dir=os.path.join(VIDEO_DIR,f"{run_tag}_run_{run_uuid}"); os.makedirs(run_dir)
    print(f"[run {run_tag}] saving to {run_dir}")
    vids=[]
    for ep in range(start_ep,start_ep+EPISODES_PER_RUN):
        total_ep+=1; rec=(ep%RENDER_EVERY==0); env.close(); env=make_env(rec)
        obs,_=env.reset(); frames=[]; score=0; done=False
        while not done:
            act=env.action_space.sample() if random.random()<epsilon else torch.argmax(policy_net(torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0))).item()
            nobs,r,term,trunc,_=env.step(act); done=term or trunc
            replay.append((obs,act,r,nobs,done)); obs=nobs; score+=r; step_cnt+=1
            if rec: frames.append(annot(env.render(),score,ep))
            if len(replay)>=BATCH_SIZE:
                ob,a,rw,nob,dn=zip(*random.sample(replay,BATCH_SIZE))
                ob_t=torch.tensor(ob,dtype=torch.float32,device=DEVICE); a_t=torch.tensor(a,device=DEVICE).unsqueeze(1)
                rw_t=torch.tensor(rw,dtype=torch.float32,device=DEVICE).unsqueeze(1); nob_t=torch.tensor(nob,dtype=torch.float32,device=DEVICE); dn_t=torch.tensor(dn,dtype=torch.float32,device=DEVICE).unsqueeze(1)
                with torch.no_grad(): tgt=rw_t+GAMMA*target_net(nob_t).max(1,keepdim=True)[0]*(1-dn_t)
                cur=policy_net(ob_t).gather(1,a_t); loss=nn.MSELoss()(cur,tgt)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); writer.add_scalar('loss',loss.item(),step_cnt)
            if step_cnt%TARGET_UPDATE_FREQ==0: target_net.load_state_dict(policy_net.state_dict())
        epsilon=max(EPSILON_END,epsilon*EPSILON_DECAY); writer.add_scalar('reward',score,total_ep); print(f"Ep {total_ep} R {score:.1f} eps {epsilon:.3f}")
        if rec:
            land= score>200; [frames.append(annot(frames[-1],score,ep,True)) for _ in range(15)] if land else None
            vids.append((save_gif(frames,ep,run_dir,run_tag),score))
        pickle.dump({'policy_state':policy_net.state_dict(),'target_state':target_net.state_dict(),'optimizer_state':optimizer.state_dict(),'epsilon':epsilon,'step':step_cnt,'episode':ep,'total_episode':total_ep},open(MODEL_PATH,'wb'))
    normals=[v for v,_ in vids]; highs=[v for v,_ in sorted(vids,key=lambda x:x[1],reverse=True)[:10]]
    gifs2mp4(normals,os.path.join(run_dir,f"{run_tag}_normals_{run_uuid}.mp4"))
    gifs2mp4(highs,  os.path.join(run_dir,f"{run_tag}_highs_{run_uuid}.mp4"))
    start_ep+=EPISODES_PER_RUN
