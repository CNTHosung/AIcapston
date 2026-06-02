"""
app.py — 시각센서 SNN 데모 앱 (GUI, 마법사형)
==================================================================
흐름:
  1) 시작 → 엑셀 입력 포맷 안내 → 실측 데이터(.xlsx/.csv) 불러오기 (특성곡선 피팅)
  2) 학습 파라미터 입력 (epochs, batch size, time steps, hidden, lr)
  3) 학습/평가 진행 표시(epoch별) → 최종 결과 → 엑셀(.xlsx) 저장
  4) "숫자 예측을 진행할까요?" → Yes면 그리기 예측 창

ML 코어(인코더/모델/학습/추론)는 snn_mnist_all.py를 재사용하며,
tkinter는 launch_gui() 안에서만 임포트하여 ML 로직을 headless 검증 가능.
==================================================================
"""
from __future__ import annotations
import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import torch

from snn_mnist_all import (
    DeviceCharacteristicEncoder, SpikingMLP, encode_batch, build_loaders,
    spike_count_loss, accuracy,
)

if getattr(sys, "frozen", False):
    HERE = os.path.dirname(sys.executable)
else:
    HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(HERE, "data")
WORKING_CSV = os.path.join(DATA_DIR, "characteristic_curve.csv")
CKPT = os.path.join(HERE, "snn_mnist.pt")

# 입력 엑셀 포맷 안내용 예시
FORMAT_COLS = ["intensity_mW_cm2", "frequency_Hz"]
FORMAT_EXAMPLE = [
    ("0.00", "0"), ("0.46", "1500"), ("0.67", "7000"),
    ("1.12", "22000"), ("2.07", "42000"), ("9.43", "50000"),
]


# ------------------------------------------------------------------
# ML 코어 (tkinter 비의존)
# ------------------------------------------------------------------
def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """다양한 컬럼명을 (intensity_mW_cm2, frequency_Hz)로 표준화."""
    cols = {c.lower().strip(): c for c in df.columns}
    icol = next((cols[k] for k in cols
                 if any(t in k for t in ["intens", "light", "mw", "세기", "광"])), None)
    fcol = next((cols[k] for k in cols
                 if any(t in k for t in ["freq", "hz", "spike", "주파", "스파이크"])), None)
    if icol is None or fcol is None:
        if df.shape[1] >= 2:          # 컬럼명 못 찾으면 처음 두 열 사용
            icol, fcol = df.columns[0], df.columns[1]
        else:
            raise ValueError("두 개 열(광 세기, 주파수)이 필요합니다.")
    out = pd.DataFrame({
        "intensity_mW_cm2": pd.to_numeric(df[icol], errors="coerce"),
        "frequency_Hz":     pd.to_numeric(df[fcol], errors="coerce"),
    }).dropna().sort_values("intensity_mW_cm2").reset_index(drop=True)
    if len(out) < 3:
        raise ValueError("유효한 측정점이 3개 미만입니다.")
    return out


def load_measurements_to_csv(path: str) -> pd.DataFrame:
    """엑셀/CSV 실측 데이터를 읽어 표준 CSV(WORKING_CSV)로 저장. 표 반환."""
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    norm = _normalize_table(df)
    os.makedirs(DATA_DIR, exist_ok=True)
    norm.to_csv(WORKING_CSV, index=False)
    return norm


def build_encoder():
    return DeviceCharacteristicEncoder(WORKING_CSV, model="auto", pixel_max=1.0)


def build_model(hidden=256, steps=25):
    return SpikingMLP(n_hidden=hidden, num_steps=steps)


def load_model(net):
    if os.path.exists(CKPT):
        net.load_state_dict(torch.load(CKPT, map_location="cpu"))
        net.eval()
        return True
    return False


def predict_digit(img28: np.ndarray, encoder, net, steps=25):
    x = torch.from_numpy(img28.reshape(1, 1, 28, 28).astype("float32"))
    spk = encode_batch(x, encoder, steps)
    with torch.no_grad():
        counts = net(spk).sum(dim=0)
        probs = torch.softmax(counts, dim=1)[0].numpy()
    return int(counts.argmax(dim=1).item()), probs


def train_model(epochs=3, batch=128, steps=25, hidden=256, lr=1e-3,
                subset=0, on_epoch=None, on_log=None):
    """학습/평가. on_epoch(dict), on_log(str). 반환: (history, encoder)."""
    device = torch.device("cpu")
    encoder = build_encoder()
    if on_log:
        on_log(encoder.summary())
    train_loader, test_loader = build_loaders(batch, subset, DATA_DIR)
    net = build_model(hidden, steps).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        net.train(); t0 = time.time()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for i, (imgs, labels) in enumerate(train_loader):
            spk = encode_batch(imgs, encoder, steps).to(device)
            labels = labels.to(device)
            out = net(spk)
            loss = spike_count_loss(out, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * labels.size(0)
            tr_correct += (out.sum(0).argmax(1) == labels).sum().item()
            tr_n += labels.size(0)
            if on_log and i % 50 == 0:
                on_log(f"  epoch {epoch+1} it {i}  loss {loss.item():.3f}")
        net.eval(); accs = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                spk = encode_batch(imgs, encoder, steps).to(device)
                accs.append(accuracy(net(spk), labels.to(device)))
        rec = {"epoch": epoch + 1,
               "train_loss": tr_loss / max(tr_n, 1),
               "train_acc": tr_correct / max(tr_n, 1),
               "test_acc": sum(accs) / len(accs),
               "sec": time.time() - t0}
        history.append(rec)
        torch.save(net.state_dict(), CKPT)
        if on_epoch:
            on_epoch(rec)
    return history, encoder


def save_results_excel(history, params, encoder, path):
    per = pd.DataFrame(history)
    best = max(history, key=lambda h: h["test_acc"])
    summary = pd.DataFrame([{
        **params,
        "encoder_model": encoder.model_name,
        "encoder_R2": round(encoder.r2, 5),
        "best_test_acc": round(best["test_acc"], 4),
        "best_epoch": best["epoch"],
        "final_test_acc": round(history[-1]["test_acc"], 4),
        "final_train_acc": round(history[-1]["train_acc"], 4),
        "total_sec": round(sum(h["sec"] for h in history), 1),
    }])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        per.to_excel(w, sheet_name="per_epoch", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)
    return path


# ------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageDraw

    BG = "#F4F6FB"; NAVY = "#1F4E79"; BLUE = "#2E75B6"
    state = {"encoder": None, "history": None, "params": None}

    root = tk.Tk()
    root.title("시각센서 SNN — 학습/예측 프로그램")
    root.geometry("680x640")
    root.configure(bg=BG)

    container = tk.Frame(root, bg=BG)
    container.pack(fill="both", expand=True, padx=16, pady=12)

    def clear():
        for w in container.winfo_children():
            w.destroy()

    def header(t, sub=None):
        tk.Label(container, text=t, font=("Malgun Gothic", 15, "bold"),
                 fg=NAVY, bg=BG).pack(anchor="w", pady=(2, 2))
        if sub:
            tk.Label(container, text=sub, font=("Malgun Gothic", 10),
                     fg="#555", bg=BG, justify="left", wraplength=620).pack(anchor="w", pady=(0, 8))

    def btn(parent, text, cmd, primary=True):
        return tk.Button(parent, text=text, command=cmd, width=22,
                         font=("Malgun Gothic", 11, "bold"),
                         bg=BLUE if primary else "#9AA7B5", fg="white",
                         activebackground=NAVY, relief="flat", pady=6)

    # ---------- STEP 1: 데이터 불러오기 + 포맷 안내 ----------
    def step_data():
        clear()
        header("1단계 · 실측 소자 데이터 불러오기",
               "빛 세기(mW/cm²)에 따른 스파이크 주파수(Hz) 측정값을 담은 Excel/CSV 파일을 불러옵니다.\n"
               "아래 형식으로 첫 행에 열 이름을 넣어 주세요. (열 이름이 달라도 자동 인식 시도)")

        # 포맷 안내 표
        fmt = tk.LabelFrame(container, text=" 입력 파일 형식 예시 ",
                            font=("Malgun Gothic", 10, "bold"), fg=NAVY, bg="white")
        fmt.pack(fill="x", pady=6)
        tv = ttk.Treeview(fmt, columns=FORMAT_COLS, show="headings", height=6)
        for c, wd in zip(FORMAT_COLS, (220, 220)):
            tv.heading(c, text=c); tv.column(c, width=wd, anchor="center")
        for r in FORMAT_EXAMPLE:
            tv.insert("", "end", values=r)
        tv.pack(padx=8, pady=8)
        tk.Label(fmt, text="· 첫 열: 광 세기(mW/cm²)   · 둘째 열: 스파이크 주파수(Hz)   · 최소 3점 이상",
                 font=("Malgun Gothic", 9), fg="#666", bg="white").pack(anchor="w", padx=10, pady=(0, 6))

        info = tk.StringVar(value="")
        tk.Label(container, textvariable=info, font=("Malgun Gothic", 10),
                 fg=NAVY, bg=BG, justify="left", wraplength=620).pack(anchor="w", pady=6)

        def choose():
            path = filedialog.askopenfilename(
                title="실측 데이터 선택",
                filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("모든 파일", "*.*")])
            if not path:
                return
            try:
                tbl = load_measurements_to_csv(path)
                enc = build_encoder()
                state["encoder"] = enc
                info.set(f"✔ 불러옴: {os.path.basename(path)}  ({len(tbl)}개 측정점)\n"
                         f"   특성곡선 피팅: {enc.model_name} 모델, R² = {enc.r2:.4f}")
                next_btn.config(state="normal")
            except Exception as e:
                messagebox.showerror("불러오기 오류", f"형식을 확인해 주세요.\n\n{e}")

        def use_example():
            try:
                state["encoder"] = build_encoder()  # 번들된 예시 CSV
                enc = state["encoder"]
                info.set(f"✔ 예시 데이터 사용  ·  피팅: {enc.model_name}, R²={enc.r2:.4f}")
                next_btn.config(state="normal")
            except Exception as e:
                messagebox.showerror("오류", str(e))

        row = tk.Frame(container, bg=BG); row.pack(pady=8)
        btn(row, "엑셀/CSV 파일 선택", choose).pack(side="left", padx=6)
        btn(row, "예시 데이터 사용", use_example, primary=False).pack(side="left", padx=6)

        next_btn = btn(container, "다음: 학습 설정 →", step_params)
        next_btn.pack(pady=10); next_btn.config(state="disabled")

    # ---------- STEP 2: 파라미터 ----------
    def step_params():
        clear()
        header("2단계 · 학습 파라미터 설정", "값을 확인/수정한 뒤 학습을 시작하세요.")
        form = tk.Frame(container, bg=BG); form.pack(pady=8)
        fields = [("epochs (학습 반복)", "epochs", "3"),
                  ("batch size", "batch", "128"),
                  ("time steps T", "steps", "25"),
                  ("hidden size", "hidden", "256"),
                  ("learning rate", "lr", "0.001")]
        vars_ = {}
        for i, (label, key, default) in enumerate(fields):
            tk.Label(form, text=label, font=("Malgun Gothic", 11), bg=BG, anchor="e",
                     width=18).grid(row=i, column=0, padx=8, pady=6, sticky="e")
            v = tk.StringVar(value=default)
            tk.Entry(form, textvariable=v, width=14, justify="center",
                     font=("Malgun Gothic", 11)).grid(row=i, column=1, padx=8, pady=6)
            vars_[key] = v

        def go():
            try:
                params = {"epochs": int(vars_["epochs"].get()),
                          "batch": int(vars_["batch"].get()),
                          "steps": int(vars_["steps"].get()),
                          "hidden": int(vars_["hidden"].get()),
                          "lr": float(vars_["lr"].get())}
            except ValueError:
                messagebox.showerror("입력 오류", "숫자를 올바르게 입력해 주세요.")
                return
            state["params"] = params
            step_train()

        row = tk.Frame(container, bg=BG); row.pack(pady=14)
        btn(row, "← 이전", step_data, primary=False).pack(side="left", padx=6)
        btn(row, "학습 시작 ▶", go).pack(side="left", padx=6)

    # ---------- STEP 3: 학습/평가 진행 ----------
    def step_train():
        clear()
        p = state["params"]
        header("3단계 · 학습 / 평가 진행",
               f"epochs={p['epochs']}, batch={p['batch']}, T={p['steps']}, "
               f"hidden={p['hidden']}, lr={p['lr']}  (CPU 기준 1 epoch 약 40초)")

        cols = ("epoch", "train_loss", "train_acc", "test_acc", "sec")
        tv = ttk.Treeview(container, columns=cols, show="headings", height=8)
        for c, wd in zip(cols, (70, 110, 110, 110, 90)):
            tv.heading(c, text=c); tv.column(c, width=wd, anchor="center")
        tv.pack(fill="x", pady=6)

        bar = ttk.Progressbar(container, mode="determinate", maximum=p["epochs"])
        bar.pack(fill="x", pady=4)
        status = tk.StringVar(value="준비 중...")
        tk.Label(container, textvariable=status, font=("Malgun Gothic", 10),
                 fg=NAVY, bg=BG, wraplength=620, justify="left").pack(anchor="w", pady=4)

        result_frame = tk.Frame(container, bg=BG); result_frame.pack(fill="x", pady=6)

        def on_epoch(rec):
            tv.insert("", "end", values=(
                rec["epoch"], f"{rec['train_loss']:.4f}", f"{rec['train_acc']*100:.2f}%",
                f"{rec['test_acc']*100:.2f}%", f"{rec['sec']:.1f}"))
            tv.yview_moveto(1.0)
            bar["value"] = rec["epoch"]
            status.set(f"epoch {rec['epoch']}/{p['epochs']} 완료 · test acc {rec['test_acc']*100:.2f}%")

        def worker():
            try:
                hist, enc = train_model(
                    epochs=p["epochs"], batch=p["batch"], steps=p["steps"],
                    hidden=p["hidden"], lr=p["lr"],
                    on_epoch=lambda r: root.after(0, on_epoch, r),
                    on_log=lambda m: print(m))
                state["history"] = hist
                root.after(0, lambda: finish(hist, enc))
            except Exception as e:
                root.after(0, lambda: (status.set("학습 오류: " + str(e)),
                                       messagebox.showerror("학습 오류", str(e))))

        def finish(hist, enc):
            best = max(hist, key=lambda h: h["test_acc"])
            status.set(f"학습 완료!  최고 test acc {best['test_acc']*100:.2f}% (epoch {best['epoch']})")
            # 엑셀 자동 저장
            out = os.path.join(HERE, "training_results.xlsx")
            try:
                save_results_excel(hist, p, enc, out)
                saved = f"결과 저장: {out}"
            except Exception as e:
                saved = f"엑셀 저장 실패: {e}"
            tk.Label(result_frame, text=saved, font=("Malgun Gothic", 10),
                     fg="#1A7F37", bg=BG, wraplength=620, justify="left").pack(anchor="w")

            def save_as():
                path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                    filetypes=[("Excel", "*.xlsx")], initialfile="training_results.xlsx")
                if path:
                    save_results_excel(hist, p, enc, path)
                    messagebox.showinfo("저장 완료", path)

            def ask_predict():
                if messagebox.askyesno("숫자 예측", "학습이 끝났습니다. 숫자 예측을 진행할까요?"):
                    step_predict()

            r = tk.Frame(result_frame, bg=BG); r.pack(pady=10)
            btn(r, "엑셀로 다시 저장", save_as, primary=False).pack(side="left", padx=6)
            btn(r, "숫자 예측 진행 →", ask_predict).pack(side="left", padx=6)
            # 자동으로도 한 번 물어봄
            root.after(300, ask_predict)

        status.set("학습 중입니다... 잠시만 기다려 주세요.")
        threading.Thread(target=worker, daemon=True).start()

    # ---------- STEP 4: 숫자 예측 ----------
    def step_predict():
        clear()
        steps = state["params"]["steps"] if state["params"] else 25
        encoder = state["encoder"] or build_encoder()
        net = build_model(state["params"]["hidden"] if state["params"] else 256, steps)
        if not load_model(net):
            messagebox.showinfo("모델 없음", "학습된 모델이 없습니다.")
            return
        header("4단계 · 숫자 예측", "검은 칸에 숫자(0~9)를 그리고 [예측]을 누르세요.")

        CANVAS = 280
        pil = Image.new("L", (CANVAS, CANVAS), 0)
        d = ImageDraw.Draw(pil)
        body = tk.Frame(container, bg=BG); body.pack()
        canvas = tk.Canvas(body, width=CANVAS, height=CANVAS, bg="black",
                           highlightthickness=1, highlightbackground=NAVY, cursor="pencil")
        canvas.grid(row=0, column=0, rowspan=5, padx=8, pady=4)
        last = {"x": None, "y": None}

        def down(e): last["x"], last["y"] = e.x, e.y
        def move(e):
            if last["x"] is not None:
                canvas.create_line(last["x"], last["y"], e.x, e.y, fill="white",
                                   width=18, capstyle=tk.ROUND, smooth=True)
                d.line([last["x"], last["y"], e.x, e.y], fill=255, width=18)
            last["x"], last["y"] = e.x, e.y
        def up(e): last["x"], last["y"] = None, None
        canvas.bind("<Button-1>", down); canvas.bind("<B1-Motion>", move)
        canvas.bind("<ButtonRelease-1>", up)

        res = tk.StringVar(value="예측: -"); prob = tk.StringVar(value="")
        tk.Label(body, textvariable=res, font=("Malgun Gothic", 22, "bold"),
                 fg=NAVY, bg=BG).grid(row=0, column=1, padx=14, sticky="w")
        tk.Label(body, textvariable=prob, font=("Malgun Gothic", 10),
                 fg="#444", bg=BG).grid(row=1, column=1, padx=14, sticky="w")

        def predict():
            img = np.asarray(pil.resize((28, 28)), dtype="float32") / 255.0
            digit, probs = predict_digit(img, encoder, net, steps)
            res.set(f"예측: {digit}")
            top = probs.argsort()[::-1][:3]
            prob.set("  ".join(f"{i}:{probs[i]*100:.0f}%" for i in top))

        def clearc():
            canvas.delete("all"); d.rectangle([0, 0, CANVAS, CANVAS], fill=0)
            res.set("예측: -"); prob.set("")

        btn(body, "예측", predict).grid(row=2, column=1, padx=14, pady=4, sticky="w")
        btn(body, "지우기", clearc, primary=False).grid(row=3, column=1, padx=14, pady=4, sticky="w")
        btn(container, "처음으로", step_data, primary=False).pack(pady=12)

    step_data()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
