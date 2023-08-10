import numpy as np
from matplotlib import pyplot as plt

mos_clip_fid = np.array([
    [0.5455478, 33.08071518, 278.0810471],
    [0.5324941, 32.39567566, 286.8700363],
    [0.5382145, 32.33951187, 281.8502487],
    [0.5218165, 32.75608826, 269.6192841],
    [0.5269595, 32.46159363, 270.4892807],
    [0.50454307, 33.15788269, 271.6655121],
    [0.52530944, 32.37728882, 294.498822],
    [0.5086229, 32.09280014, 267.3688507],
    [0.52532375, 31.40999794, 277.0564301],
    [0.5181005, 32.13401413, 268.9576279],
    [0.5195663, 31.882267, 271.903212],
    [0.5200906, 31.94002342, 286.7533066],
    [0.52608037, 30.95668221, 270.3350388],
    [0.5231348, 31.49349976, 258.4729584],
    [0.537731, 30.74035454, 255.1261505],
    [0.532946, 30.31161499, 256.9206573],
    [0.5410548, 30.17402267, 251.7462799],
    [0.53773713, 30.28982925, 236.7787476],
    [0.54198974, 30.34298134, 249.8763824],
    [0.5288592, 30.46555138, 261.8818497],
    [0.5468169, 29.96123123, 247.0635605],
    [0.52521545, 29.77492142, 258.3609268],
    [0.5291022, 30.25540543, 280.672084],
    [0.53427774, 29.8724308, 274.9344696],
    [0.5450267, 30.89665985, 255.1625961],
])

plt.figure(figsize=(8 * 2, 6))
plt.title('VILA MOS (higher is better)')
plt.grid()
plt.plot(np.arange(200, 5200, 200), mos_clip_fid[:, 0])
plt.axhline(0.5143854, c='r', label='Baseline')
plt.xticks(np.arange(200, 5200, 200))
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8 * 2, 6))
plt.title('CLIP Score (higher is better)')
plt.grid()
plt.plot(np.arange(200, 5200, 200), mos_clip_fid[:, 1])
plt.axhline(32.473182678222656, c='r', label='Baseline')
plt.xticks(np.arange(200, 5200, 200))
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8 * 2, 6))
plt.title('FID (lower is better)')
plt.grid()
plt.plot(np.arange(200, 5200, 200), mos_clip_fid[:, 2])
plt.axhline(289.1153137207031, c='r', label='Baseline')
plt.xticks(np.arange(200, 5200, 200))
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8 * 2, 6))
plt.title('VILA/FID (higher is better)')
plt.grid()
plt.plot(np.arange(200, 5200, 200), mos_clip_fid[:, 0] / mos_clip_fid[:, 2])
plt.axhline(0.5143854 / 289.1153137207031, c='r', label='Baseline')
plt.xticks(np.arange(200, 5200, 200))
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()
