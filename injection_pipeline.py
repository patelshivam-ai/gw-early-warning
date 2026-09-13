import torch
import numpy as np
import matplotlib.pyplot as plt
from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components

SAMPLE_RATE = 2048
DURATION    = 4
F_MIN       = 20.0
F_REF       = 20.0
SEED        = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

chirp_mass = torch.tensor([1.2189])
mass_ratio = torch.tensor([1.0])
mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(chirp_mass, mass_ratio)

params = {
    "mass_1":      mass_1,
    "mass_2":      mass_2,
    "s1z":         torch.tensor([0.0]),
    "s2z":         torch.tensor([0.0]),
    "distance":    torch.tensor([100.0]), 
    "phic":        torch.tensor([0.0]),
    "inclination": torch.tensor([0.0]),
    "chirp_mass":  chirp_mass,
    "mass_ratio":  mass_ratio,
    "chi1":        torch.tensor([0.0]),
    "chi2":        torch.tensor([0.0]),
}

generator = TimeDomainCBCWaveformGenerator(
    approximant=IMRPhenomD(),
    sample_rate=SAMPLE_RATE,
    f_min=F_MIN,
    duration=DURATION,
    f_ref=F_REF,
    right_pad=0.1,
)

hc, hp = generator(**params)
assert not torch.all(torch.isnan(hp)), "Waveform is all NaN — check params"
hp = torch.nan_to_num(hp, nan=0.0) 
hc = torch.nan_to_num(hc, nan=0.0)

print(f"Waveform shape: {hp.shape}  |  samples: {hp.shape[-1]}")

F_plus_H1,  F_cross_H1  =  0.6,  0.4
F_plus_L1,  F_cross_L1  = -0.4,  0.6

h_H1 = F_plus_H1 * hp + F_cross_H1 * hc  
h_L1 = F_plus_L1 * hp + F_cross_L1 * hc   

signal = torch.cat([h_H1, h_L1], dim=0).unsqueeze(0)
print(f"Projected signal shape: {signal.shape}  →  (batch, detectors, samples)")

N       = hp.shape[-1]
freqs   = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)
df      = freqs[1] - freqs[0]

def aligo_psd(f):
    f   = np.asarray(f, dtype=float)
    psd = np.ones_like(f) * 1e-40
    mask = f >= F_MIN
    fs   = f[mask]
    p = (
        0.0152 * fs**(-4)
        + 0.2935 * (fs / 245.4)**9.99
        + (1 - (fs / 145.3)**2 + 0.4 * (fs / 145.3)**4)
        / (1 + 0.5 * (fs / 145.3)**2)
    )**2 * 1e-49
    psd[mask] = p
    return psd

psd   = aligo_psd(freqs)
psd[psd <= 0] = 1e-80
sigma = np.sqrt(psd / (2 * df)) 

def coloured_noise(sigma):
    wn_fd  = (np.random.randn(len(sigma)) + 1j * np.random.randn(len(sigma)))
    cn_fd  = wn_fd * sigma
    cn_td  = np.fft.irfft(cn_fd, n=N)
    return torch.tensor(cn_td, dtype=torch.float32)

noise_H1 = coloured_noise(sigma)   
noise_L1 = coloured_noise(sigma)   

noise = torch.stack([noise_H1, noise_L1], dim=0).unsqueeze(0)

network_input = signal + noise   

print(f"Network input tensor shape: {network_input.shape}")
print(f"  dim 0 = batch size  ({network_input.shape[0]})")
print(f"  dim 1 = detectors   ({network_input.shape[1]}: H1, L1)")
print(f"  dim 2 = time samples ({network_input.shape[2]} @ {SAMPLE_RATE} Hz = {DURATION}s)")

t = np.linspace(0, DURATION, N)

fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharey="row")
det_labels = ["H1", "L1"]
colors     = ["steelblue", "darkorange"]

for i, (label, color) in enumerate(zip(det_labels, colors)):
    axes[i][0].plot(t, noise[0, i].numpy(), color=color, lw=0.6, alpha=0.8)
    axes[i][0].set_title(f"{label} — noise only")
    axes[i][0].set_ylabel("Strain")

    axes[i][1].plot(t, network_input[0, i].detach().numpy(), color=color, lw=0.6, alpha=0.8)
    axes[i][1].set_title(f"{label} — signal + noise  (BNS @ 100 Mpc)")

for ax in axes[1]:
    ax.set_xlabel("Time (s)")

fig.suptitle(
    "Network input: 2-channel strain  |  1.4+1.4 M☉ BNS at 100 Mpc\n"
    "Shape: (batch=1, detectors=2, samples=8192)",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("injection_pipeline.png", dpi=150)
print("Saved injection_pipeline.png")