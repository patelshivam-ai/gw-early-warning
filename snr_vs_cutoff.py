import torch
import numpy as np
import matplotlib.pyplot as plt
from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components

SAMPLE_RATE = 2048
DURATION = 4
F_MIN = 10.0
F_REF = 20.0

chirp_mass = torch.tensor([1.2189])
mass_ratio = torch.tensor([1.0])
mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(chirp_mass, mass_ratio)

params = {
    "chirp_mass":  chirp_mass,
    "mass_ratio":  mass_ratio,
    "mass_1":      mass_1,
    "mass_2":      mass_2,
    "chi1":        torch.tensor([0.0]),
    "chi2":        torch.tensor([0.0]),
    "s1z":         torch.tensor([0.0]),
    "s2z":         torch.tensor([0.0]),
    "distance":    torch.tensor([100.0]),
    "phic":        torch.tensor([0.0]),
    "inclination": torch.tensor([0.0]),
}

approximant = IMRPhenomD()
generator = TimeDomainCBCWaveformGenerator(
    approximant=approximant,
    sample_rate=SAMPLE_RATE,
    f_min=F_MIN,
    duration=DURATION,
    f_ref=F_REF,
    right_pad=0.1,
)

hc, hp = generator(**params)

print("hp shape:", hp.shape)
print("hp nan count:", torch.isnan(hp).sum().item())
print("hp min:", hp[~torch.isnan(hp)].min().item() if not torch.all(torch.isnan(hp)) else "all nan")

hp_clean = torch.nan_to_num(hp, nan=0.0)

hp_td = hp_clean.squeeze().numpy()
hp_fd = np.fft.rfft(hp_td)
freqs_np = np.fft.rfftfreq(len(hp_td), d=1.0 / SAMPLE_RATE)
df = freqs_np[1] - freqs_np[0]

def aligo_psd(f):
    f = np.asarray(f, dtype=float)
    psd = np.ones_like(f) * 1e-40
    mask = f >= F_MIN
    fs = f[mask]
    p = (
        0.0152 * fs**(-4)
        + 0.2935 * (fs / 245.4)**9.99
        + (1 - (fs / 145.3)**2 + 0.4 * (fs / 145.3)**4)
        / (1 + 0.5 * (fs / 145.3)**2)
    ) ** 2 * 1e-49
    psd[mask] = p
    return psd

psd = aligo_psd(freqs_np)
print("freqs range:", freqs_np[1], "to", freqs_np[-1])
print("psd min:", psd.min(), "psd max:", psd.max())
print("hp_fd min:", np.abs(hp_fd).min(), "hp_fd max:", np.abs(hp_fd).max())
print("any nan in psd:", np.any(np.isnan(psd)))
print("any nan in hp_fd:", np.any(np.isnan(hp_fd)))
psd[psd <= 0] = 1e-80

f_cutoffs = np.arange(20, 500, 5)
snrs = []

for f_cut in f_cutoffs:
    mask = (freqs_np >= F_MIN) & (freqs_np <= f_cut)
    integrand = (np.abs(hp_fd[mask])**2) / psd[mask]
    snr = np.sqrt(4 * df * np.sum(integrand))
    snrs.append(snr)

snrs = np.array(snrs)
snr_max = snrs[-1]
snr_fraction = snrs / snr_max

plt.figure(figsize=(9, 5))
plt.plot(f_cutoffs, snr_fraction * 100)
plt.xlabel("Frequency cutoff (Hz)")
plt.ylabel("Accumulated SNR (% of total)")
plt.title("SNR accumulation vs frequency cutoff — 1.4+1.4 M☉ BNS at 100 Mpc")
plt.axvline(30, color="red", linestyle="--", label="30 Hz (O3 noise floor)")
plt.axvline(100, color="orange", linestyle="--", label="100 Hz (~merger approach)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("snr_vs_cutoff.png", dpi=150)
print("Saved snr_vs_cutoff.png")
print(f"SNR at 30 Hz cutoff:  {snr_fraction[np.argmin(np.abs(f_cutoffs - 30))] * 100:.1f}% of total")
print(f"SNR at 100 Hz cutoff: {snr_fraction[np.argmin(np.abs(f_cutoffs - 100))] * 100:.1f}% of total")