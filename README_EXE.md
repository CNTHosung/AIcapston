# 시각센서 SNN 데모 앱 — EXE 만들기

소자 실측 특성곡선 기반 Light-to-Spike SNN으로 손글씨 숫자를 인식하는 GUI 앱입니다.
이 폴더의 파일을 Windows PC에서 빌드하면 `.exe` 실행파일이 만들어집니다.

## 폴더 구성

| 파일 | 역할 |
|------|------|
| `app.py` | GUI 앱 본체 (그리기 → 예측, 학습 버튼) |
| `snn_mnist_all.py` | 인코더 + SNN + 학습 코어 (앱이 가져다 씀) |
| `snn_mnist.pt` | 미리 학습된 모델 (정확도 약 96%) — 앱 실행 즉시 예측 가능 |
| `data/characteristic_curve.csv` | 소자 특성곡선 데이터 |
| `data/MNIST/` | 학습용 데이터 (앱의 "학습 시작"에만 필요) |
| `app.spec` / `build_exe.bat` | PyInstaller 빌드 설정/스크립트 |
| `requirements.txt` | 필요한 패키지 목록 |

## 빌드 방법 (Windows)

> ⚠️ EXE는 **Windows에서** 만들어야 합니다(맥/리눅스 exe와 호환 안 됨).
> Python 3.10 이상이 설치돼 있어야 합니다.

1. 이 `app_exe` 폴더를 통째로 Windows PC에 둡니다.
2. `build_exe.bat` 를 더블클릭합니다. (또는 명령창에서 실행)
   - 처음엔 패키지 설치 + 빌드로 수 분이 걸립니다.
3. 끝나면 `dist\SensorSNN\SensorSNN.exe` 가 생깁니다. 더블클릭하면 앱이 뜹니다.

명령창에서 직접 하려면:

```bat
pip install -r requirements.txt
pyinstaller --noconfirm --clean app.spec
xcopy /E /I /Y data "dist\SensorSNN\data"
copy /Y snn_mnist.pt "dist\SensorSNN\"
```

## 사용법 (4단계 마법사)

1. **실측 데이터 불러오기**: 시작하면 입력 파일 형식(열 1=광 세기 mW/cm², 열 2=스파이크 주파수 Hz)을 표로 보여줍니다. `엑셀/CSV 파일 선택`으로 본인 측정 파일을 불러오면 특성곡선이 자동 피팅됩니다(열 이름이 달라도 자동 인식 시도). `예시 데이터 사용`도 가능.
2. **학습 파라미터 입력**: epochs, batch size, time steps(T), hidden size, learning rate를 입력합니다.
3. **학습/평가 진행**: epoch별 train_loss·train_acc·test_acc·시간이 표로 표시되고, 끝나면 최종 결과 요약과 함께 `training_results.xlsx`(per_epoch·summary 시트)로 자동 저장됩니다. `엑셀로 다시 저장`으로 위치 지정 저장도 가능.
4. **숫자 예측**: 학습 후 "숫자 예측을 진행할까요?"를 묻고, Yes면 검은 칸에 숫자(0~9)
## 문제 해결 (Troubleshooting)

- **`build_exe.bat` 가 명령을 토막내며 실패** → 줄바꿈/인코딩 문제였음. 현재 스크립트는 CRLF·영문으로 수정됨(해결).
- **`[WinError 5] 액세스가 거부` (pip)** → 파이썬이 공용 폴더(ProgramData\Miniconda3)에 설치돼 권한 부족. 현재 스크립트는 **별도 가상환경(.venv)** 에 설치하므로 관리자 권한 없이 동작함.
- **`[WinError 1114] DLL 초기화 ... c10.dll` (torch import 실패)** → 가장 흔한 원인은 conda(base)에 pip torch를 섞어 생긴 DLL 충돌. 현재 스크립트는 깨끗한 `.venv`에서 빌드하여 이를 회피함. 그래도 안 되면 **Microsoft Visual C++ 재배포 패키지(x64)** 설치 후 재시도:
  https://aka.ms/vs/17/release/vc_redist.x64.exe
- **처음부터 다시 하고 싶을 때** → `app_exe` 폴더의 `.venv`, `build`, `dist` 폴더를 삭제하고 `build_exe.bat`을 다시 실행.

> 참고: `.venv` 방식은 conda 환경을 건드리지 않고 격리된 곳에 설치/빌드하므로,
> 권한 문제와 DLL 충돌을 동시에 피하는 가장 안전한 방법입니다.
