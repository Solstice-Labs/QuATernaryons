import torch, os, resource, time, glob, sys
torch.set_num_threads(2)
print("torch", torch.__version__, "| torch threads:", torch.get_num_threads(),
      "| os.cpu_count:", os.cpu_count(), flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
print("rss start:", rss(), "MB", flush=True)
from safetensors.torch import load_file
snap = glob.glob("/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/*")[0]
st = glob.glob(snap + "/*.safetensors")[0]
W = load_file(st)["model.layers.0.self_attn.q_proj.weight"].float()
print("W:", rss(), "MB", flush=True)
Ht = torch.load("/content/hessians/model_layers_0_self_attn_q_proj.pt",
                map_location="cpu", weights_only=False)
if isinstance(Ht, dict): Ht = next(iter(Ht.values()))
H = Ht.float()
print("H:", rss(), "MB", H.dtype, flush=True)
del Ht
Hn = torch.ones(1, 1)
while Hn.shape[0] < 2048:
    Hn = torch.cat([torch.cat([Hn, Hn], 1), torch.cat([Hn, -Hn], 1)], 0)
Hn /= 2048 ** 0.5
print("Hn:", rss(), "MB", flush=True)
t0 = time.time(); H_rot = Hn @ H @ Hn
print("H_rot:", rss(), "MB", f"{time.time()-t0:.1f}s", flush=True)
# golden loop with per-iter rss
Wf = W.reshape(-1, 32)
amax = Wf.abs().max(1, keepdim=True).values.clamp_min(1e-12)
Wn = Wf / amax
lo = torch.full_like(amax, 1e-4); hi = torch.ones_like(amax)
gr = (5 ** 0.5 - 1) / 2
c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
for i in range(36):
    Tc = torch.clamp(torch.round(Wn / c), -1, 1)
    ec = ((Wn - c * Tc) ** 2).sum(1)
    Td = torch.clamp(torch.round(Wn / d), -1, 1)
    ed = ((Wn - d * Td) ** 2).sum(1)
    m = ec < ed
    hi = torch.where(m, d, hi); lo = torch.where(m, lo, c)
    c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
    if i % 6 == 0:
        print(f"  iter {i}: rss {rss()} MB", flush=True)
print("golden done:", rss(), "MB", flush=True)
del Tc, Td, ec, ed, Wf, Wn, amax
Q = None
a = 0.5 * (lo + hi)
T = torch.clamp(torch.round(W.reshape(-1, 32) / a), -1, 1)
Q = (a * T * amax2 if False else a * T * (W.reshape(-1,32).abs().max(1, keepdim=True).values)).reshape(2048, 2048)
print("Q done:", rss(), "MB", flush=True)
E = W @ Hn - Q @ Hn
EH = E @ H
v = float((EH.double() * E.double()).sum()) / float(torch.trace(H.double()))
print("rel2 no-rot:", round(v, 4), "rss:", rss(), "MB", flush=True)
