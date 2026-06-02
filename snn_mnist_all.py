"""
snn_mnist_all.py  (단일 파일 통합 버전)
==================================================================
소자 실측 특성곡선 기반 Light-to-Spike SNN — MNIST 분류 (올인원)

  [1] DeviceCharacteristicEncoder : 소자 실측곡선 피팅 + 인코딩  (핵심 IP)
  [2] SpikingMLP                  : snnTorch 기반 SNN 분류망
  [3] train loop                  : 학습 / 평가 / 결과 저장

핵심 차이점: 픽셀 명암 -> 광 세기 -> (소자 실측곡선) -> 스파이크 주파수 -> 발화확률
            일반 MNIST SNN과 달리 입력 인코딩이 소자 물리특성을 반영함.

사용 예:
  python snn_mnist_all.py --epochs 5 --batch 128 --steps 25
  python snn_mnist_all.py --epochs 1 --batch 128 --steps 20 --subset 2000
  python snn_mnist_all.py --resume --epochs 1           # 이어서 학습
==================================================================
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate, spikegen

HERE = os.path.dirname(os.path.abspath(__file__))


# ==================================================================
# [1] 소자 특성곡선 인코딩 모듈  (핵심 IP)
# ==================================================================
def _linear(I, a, b):
    return a * I + b


def _hill(I, f_max, I_half, n):
    """포화형(Hill) 모델: 광 세기가 커지면 주파수가 포화."""
    I = np.maximum(I, 0.0)
    return f_max * (I ** n) / (I_half ** n + I ** n)


def _logistic(I, f_max, k, I0):
    """로지스틱(S자) 모델: 임계 광 세기 부근에서 급격히 켜짐."""
    return f_max / (1.0 + np.exp(-k * (I - I0)))


_MODELS = {
    "linear":   (_linear,   [1.0, 0.0]),
    "hill":     (_hill,     [1e3, 1.0, 2.0]),
    "logistic": (_logistic, [1e3, 1.0, 1.0]),
}


class DeviceCharacteristicEncoder:
    """소자 실측곡선을 피팅하고 픽셀->주파수->발화확률로 인코딩한다.

    csv 컬럼: intensity_mW_cm2, frequency_Hz  (최소 3점)
    model   : 'auto' | 'linear' | 'hill' | 'logistic'
    pixel_max: 입력 픽셀 최대치 ([0,1]이면 1.0, [0,255]면 255.0)
    """

    def __init__(self, csv_path: str, model: str = "auto", pixel_max: float = 1.0):
        self.csv_path = csv_path
        self.pixel_max = float(pixel_max)
        df = pd.read_csv(csv_path).dropna().sort_values("intensity_mW_cm2")
        self.I_meas = df["intensity_mW_cm2"].to_numpy(dtype=float)
        self.f_meas = df["frequency_Hz"].to_numpy(dtype=float)
        if len(self.I_meas) < 3:
            raise ValueError("실측 점이 3개 미만입니다.")
        self.I_min, self.I_max = float(self.I_meas.min()), float(self.I_meas.max())
        self.f_max_meas = float(self.f_meas.max())
        self._fit(model)

    def _fit(self, model: str):
        candidates = _MODELS.keys() if model == "auto" else [model]
        best = None
        for name in candidates:
            func, p0 = _MODELS[name]
            try:
                popt, _ = curve_fit(func, self.I_meas, self.f_meas, p0=p0, maxfev=20000)
                pred = func(self.I_meas, *popt)
                ss_res = np.sum((self.f_meas - pred) ** 2)
                ss_tot = np.sum((self.f_meas - self.f_meas.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if best is None or r2 > best[2]:
                    best = (name, func, r2, popt)
            except Exception:
                continue
        if best is None:
            raise RuntimeError("곡선 피팅 실패. 데이터를 확인하세요.")
        self.model_name, self._func, self.r2, self._popt = best

    def pixel_to_intensity(self, pixel):
        p = np.asarray(pixel, dtype=float) / self.pixel_max
        return self.I_min + p * (self.I_max - self.I_min)

    def intensity_to_frequency(self, I):
        return np.clip(self._func(np.asarray(I, dtype=float), *self._popt), 0.0, None)

    def pixel_to_frequency(self, pixel):
        return self.intensity_to_frequency(self.pixel_to_intensity(pixel))

    def pixel_to_rate(self, pixel):
        f = self.pixel_to_frequency(pixel)
        return np.clip(f / max(self.f_max_meas, 1e-9), 0.0, 1.0)

    def summary(self):
        return (f"[Encoder] model={self.model_name} R2={self.r2:.4f} "
                f"params={np.round(self._popt,4).tolist()} "
                f"I=[{self.I_min:.4g},{self.I_max:.4g}]mW/cm^2 "
                f"fmax={self.f_max_meas:.4g}Hz")


# ==================================================================
# [2] snnTorch 기반 Spiking Neural Network
# ==================================================================
class SpikingMLP(nn.Module):
    def __init__(self, n_in=784, n_hidden=256, n_out=10, beta=0.9, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spk_in):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for t in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(spk_in[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)


def spike_count_loss(spk_out, targets):
    return nn.functional.cross_entropy(spk_out.sum(dim=0), targets)


def accuracy(spk_out, targets):
    return (spk_out.sum(dim=0).argmax(dim=1) == targets).float().mean().item()


# ==================================================================
# [3] 데이터 / 인코딩 / 학습 루프
# ==================================================================
def build_loaders(batch, subset, data_root):
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    if subset > 0:
        train = Subset(train, range(min(subset, len(train))))
        test = Subset(test, range(min(subset // 5 or 1, len(test))))
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(test, batch_size=batch, shuffle=False))


def encode_batch(images, encoder, num_steps):
    # 핵심: 픽셀 -> 소자 실측곡선 -> 발화확률 -> 스파이크 열
    flat = images.view(images.size(0), -1)
    rate = torch.from_numpy(encoder.pixel_to_rate(flat.cpu().numpy())).float()
    return spikegen.rate(rate, num_steps=num_steps)


def save_results(history, encoder, args):
    results = {
        "config": {"epochs": args.epochs, "batch": args.batch, "steps": args.steps,
                   "hidden": args.hidden, "lr": args.lr, "subset": args.subset},
        "encoder": {"model": encoder.model_name, "r2": encoder.r2},
        "history": history,
        "best_test_acc": max((h["test_acc"] for h in history), default=0.0),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        ac = [h["test_acc"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(ep, ac, "o-", color="navy")
        plt.xlabel("Epoch"); plt.ylabel("Test accuracy")
        plt.title("MNIST SNN (device-curve encoding)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(HERE, "training_curve.png"), dpi=130)
        plt.close()
    except Exception as e:
        print("plot skip:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "characteristic_curve.csv"))
    ap.add_argument("--data-root", default=os.path.join(HERE, "data"))
    ap.add_argument("--resume", action="store_true", help="이전 체크포인트에서 이어서 학습")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    encoder = DeviceCharacteristicEncoder(args.csv, model="auto", pixel_max=1.0)
    print(encoder.summary(), flush=True)

    train_loader, test_loader = build_loaders(args.batch, args.subset, args.data_root)
    net = SpikingMLP(n_hidden=args.hidden, num_steps=args.steps).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    history, start_epoch = [], 0
    ckpt = os.path.join(HERE, "snn_mnist.pt")
    res = os.path.join(HERE, "results.json")
    if args.resume and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device))
        print("resumed from snn_mnist.pt", flush=True)
        if os.path.exists(res):
            history = json.load(open(res)).get("history", [])
            start_epoch = (history[-1]["epoch"] + 1) if history else 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        net.train()
        for i, (imgs, labels) in enumerate(train_loader):
            spk_in = encode_batch(imgs, encoder, args.steps).to(device)
            labels = labels.to(device)
            spk_out = net(spk_in)
            loss = spike_count_loss(spk_out, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 50 == 0:
                print(f"epoch {epoch} it {i:4d}  loss {loss.item():.4f}  "
                      f"acc {accuracy(spk_out, labels):.3f}", flush=True)

        net.eval()
        accs = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                spk_in = encode_batch(imgs, encoder, args.steps).to(device)
                labels = labels.to(device)
                accs.append(accuracy(net(spk_in), labels))
        test_acc = sum(accs) / len(accs)
        history.append({"epoch": epoch, "test_acc": test_acc})
        print(f"== epoch {epoch}  test acc {test_acc:.4f} ==", flush=True)
        torch.save(net.state_dict(), ckpt)
        save_results(history, encoder, args)

    print("saved -> snn_mnist.pt, results.json, training_curve.png", flush=True)


if __name__ == "__main__":
    main()
