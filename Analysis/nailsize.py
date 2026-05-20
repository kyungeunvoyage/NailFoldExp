"""
nail_measure.py
===============
손톱 측정 반자동 GUI 도구
- 폴더에서 이미지 자동 로드
- 6개 포인트 클릭 → 자동 계산
- CSV 누적 저장

사용법:
  pip install opencv-python pillow numpy
  python nail_measure.py

조작법:
  - 마우스 클릭: 포인트 찍기
  - z: 마지막 포인트 취소 (undo)
  - r: 현재 이미지 포인트 초기화
  - s: 저장 후 다음 이미지
  - n: 저장 없이 다음 이미지 (스킵)
  - b: 이전 이미지
  - +/-: 줌인/줌아웃
  - 드래그: 이미지 패닝
  - q: 종료
"""

import cv2
import numpy as np
import csv
import os
import sys
import math
from pathlib import Path
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMG_FOLDER = SCRIPT_DIR.parent / 'Data' / 'Before'
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
OUTPUT_CSV = 'nail_measurements.csv'
WINDOW_NAME = 'Nail Measure Tool'
WINDOW_W, WINDOW_H = 1280, 800

# 포인트 색상 (BGR)
COLORS = [
    (80, 240, 200),   # 1: 손톱 왼쪽 끝  - 연두
    (80, 240, 200),   # 2: 손톱 오른쪽 끝 - 연두
    (240, 200, 80),   # 3: 손톱 상단      - 하늘
    (240, 200, 80),   # 4: 손톱 하단      - 하늘
    (90, 80, 240),    # 5: 왼쪽 피부띠    - 빨강
    (80, 160, 240),   # 6: 오른쪽 피부띠  - 주황
]

STEP_LABELS = [
    "① 손톱 왼쪽 끝 클릭 (가로 너비 기준)",
    "② 손톱 오른쪽 끝 클릭 (가로 너비 기준)",
    "③ 손톱 상단/큐티클 경계 클릭",
    "④ 손톱 하단 경계 클릭",
    "⑤ 왼쪽 피부띠 안쪽 경계 클릭",
    "⑥ 오른쪽 피부띠 안쪽 경계 클릭",
    "✅ 완료! [s] 저장 후 다음  [r] 초기화",
]

# ── 상태 ──────────────────────────────────────────────────────────
state = {
    'img_paths': [],
    'img_idx': 0,
    'orig_img': None,
    'display_img': None,
    'points': [],          # 원본 이미지 좌표 [(x,y), ...]
    'zoom': 1.0,
    'offset': [0, 0],      # 패닝 오프셋
    'drag_start': None,
    'drag_offset': None,
    'records': [],
    'saved_indices': set(),
}

# ── 이미지 로드 ───────────────────────────────────────────────────
def load_image(idx):
    path = str(state['img_paths'][idx])
    img = cv2.imread(path)
    if img is None:
        print(f"[경고] 이미지 로드 실패: {path}")
        return False
    state['orig_img'] = img
    state['points'] = []
    state['zoom'] = 1.0
    state['offset'] = [0, 0]
    # auto-fit
    ih, iw = img.shape[:2]
    scale_x = (WINDOW_W - 320) / iw
    scale_y = WINDOW_H / ih
    state['zoom'] = min(scale_x, scale_y, 1.0)
    return True

# ── 좌표 변환 ─────────────────────────────────────────────────────
def img_to_display(ix, iy):
    """원본 좌표 → 디스플레이 좌표"""
    z = state['zoom']
    ox, oy = state['offset']
    return int(ix * z + ox), int(iy * z + oy)

def display_to_img(dx, dy):
    """디스플레이 좌표 → 원본 좌표"""
    z = state['zoom']
    ox, oy = state['offset']
    return (dx - ox) / z, (dy - oy) / z

# ── 거리 계산 ─────────────────────────────────────────────────────
def dist(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def calc_measurements():
    pts = state['points']
    results = {}
    if len(pts) >= 2:
        results['nail_width_px'] = round(dist(pts[0], pts[1]), 1)
    if len(pts) >= 4:
        results['nail_height_px'] = round(dist(pts[2], pts[3]), 1)
    if len(pts) >= 5:
        results['left_skin_px'] = round(abs(pts[0][0] - pts[4][0]), 1)
    if len(pts) >= 6:
        results['right_skin_px'] = round(abs(pts[1][0] - pts[5][0]), 1)
    # 비율
    if 'nail_width_px' in results and 'nail_height_px' in results:
        w, h = results['nail_width_px'], results['nail_height_px']
        results['ratio_w_h'] = round(w/h, 4) if h > 0 else None
    if 'nail_width_px' in results and 'left_skin_px' in results:
        results['ratio_left_w'] = round(results['left_skin_px']/results['nail_width_px'], 4)
    if 'nail_width_px' in results and 'right_skin_px' in results:
        results['ratio_right_w'] = round(results['right_skin_px']/results['nail_width_px'], 4)
    return results

# ── 렌더링 ────────────────────────────────────────────────────────
def render():
    if state['orig_img'] is None:
        return

    img = state['orig_img']
    ih, iw = img.shape[:2]
    z = state['zoom']
    ox, oy = state['offset']

    # 줌/패닝 적용
    scaled_w = int(iw * z)
    scaled_h = int(ih * z)
    scaled = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # 캔버스
    canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    # 이미지 붙이기 (클리핑)
    x1 = max(0, ox); y1 = max(0, oy)
    x2 = min(WINDOW_W - 320, ox + scaled_w)
    y2 = min(WINDOW_H, oy + scaled_h)
    sx1 = max(0, -ox); sy1 = max(0, -oy)
    sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)

    if x2 > x1 and y2 > y1:
        canvas[y1:y2, x1:x2] = scaled[sy1:sy2, sx1:sx2]

    pts = state['points']

    # 측정선 그리기
    def draw_line(a, b, color, label):
        da = img_to_display(*a)
        db = img_to_display(*b)
        cv2.line(canvas, da, db, color, 1, cv2.LINE_AA)
        mx = (da[0]+db[0])//2
        my = (da[1]+db[1])//2 - 10
        d_px = dist(a, b)
        cv2.putText(canvas, f"{label}:{d_px:.0f}px", (mx-20, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if len(pts) >= 2:
        draw_line(pts[0], pts[1], COLORS[0], "W")
    if len(pts) >= 4:
        draw_line(pts[2], pts[3], COLORS[2], "H")
    if len(pts) >= 5:
        left_end = (pts[4][0], pts[0][1])
        draw_line(pts[0], left_end, COLORS[4], "L")
    if len(pts) >= 6:
        right_end = (pts[5][0], pts[1][1])
        draw_line(pts[1], right_end, COLORS[5], "R")

    # 포인트 그리기
    for i, (px, py) in enumerate(pts):
        dx, dy = img_to_display(px, py)
        color = COLORS[i]
        cv2.circle(canvas, (dx,dy), 7, color, -1)
        cv2.circle(canvas, (dx,dy), 7, (0,0,0), 1)
        cv2.putText(canvas, str(i+1), (dx-4, dy+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

    # ── 오른쪽 패널 ──────────────────────────────────────────────
    px_start = WINDOW_W - 316
    cv2.rectangle(canvas, (px_start-4, 0), (WINDOW_W, WINDOW_H), (35,35,35), -1)
    cv2.line(canvas, (px_start-4, 0), (px_start-4, WINDOW_H), (60,60,60), 1)

    def put(text, y, color=(200,200,200), scale=0.45, bold=False):
        thickness = 2 if bold else 1
        cv2.putText(canvas, text, (px_start+6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # 파일명
    fname = state['img_paths'][state['img_idx']].name
    idx_str = f"[{state['img_idx']+1}/{len(state['img_paths'])}]"
    put(idx_str, 28, (100,200,100), 0.5, True)
    put(fname[:28], 52, (180,180,180), 0.42)

    # 저장 여부
    saved = state['img_idx'] in state['saved_indices']
    status_color = (80,220,80) if saved else (80,80,220)
    status_text = "SAVED" if saved else "NOT SAVED"
    put(status_text, 75, status_color, 0.42, True)

    cv2.line(canvas, (px_start, 88), (WINDOW_W, 88), (55,55,55), 1)

    # 단계 안내
    step = len(pts)
    put("[ 측정 단계 ]", 108, (120,200,255), 0.45, True)
    for i, label in enumerate(STEP_LABELS):
        y = 130 + i * 22
        if i < step:
            color = (80, 180, 80)
            prefix = "✓ "
        elif i == step:
            color = (80, 240, 200) if i < 6 else (80,220,80)
            prefix = "▶ "
        else:
            color = (80, 80, 80)
            prefix = "  "
        short = (prefix + label)[:38]
        put(short, y, color, 0.38)

    cv2.line(canvas, (px_start, 290), (WINDOW_W, 290), (55,55,55), 1)

    # 측정값
    m = calc_measurements()
    put("[ 측정값 (px) ]", 310, (120,200,255), 0.45, True)
    rows = [
        ("손톱 가로 W", m.get('nail_width_px')),
        ("손톱 세로 H", m.get('nail_height_px')),
        ("왼쪽 피부띠 L", m.get('left_skin_px')),
        ("오른쪽 피부띠 R", m.get('right_skin_px')),
    ]
    for i, (label, val) in enumerate(rows):
        y = 332 + i * 20
        val_str = f"{val:.1f}" if val is not None else "—"
        color = (80,240,200) if val is not None else (80,80,80)
        put(f"{label:<14} {val_str}", y, color, 0.4)

    cv2.line(canvas, (px_start, 420), (WINDOW_W, 420), (55,55,55), 1)

    put("[ 비율 ]", 440, (120,200,255), 0.45, True)
    ratios = [
        ("W : H", m.get('ratio_w_h')),
        ("L / W", m.get('ratio_left_w')),
        ("R / W", m.get('ratio_right_w')),
    ]
    for i, (label, val) in enumerate(ratios):
        y = 462 + i * 20
        val_str = f"{val:.4f}" if val is not None else "—"
        color = (240,180,80) if val is not None else (80,80,80)
        put(f"{label:<10} {val_str}", y, color, 0.4)

    cv2.line(canvas, (px_start, 520), (WINDOW_W, 520), (55,55,55), 1)

    # 단축키
    put("[ 단축키 ]", 538, (120,200,255), 0.45, True)
    keys = [
        "s  - 저장 후 다음",
        "n  - 스킵 (다음)",
        "b  - 이전 이미지",
        "z  - 마지막 포인트 취소",
        "r  - 포인트 초기화",
        "+/-  줌인/줌아웃",
        "드래그  패닝",
        "q  - 종료",
    ]
    for i, k in enumerate(keys):
        put(k, 560 + i*20, (150,150,150), 0.38)

    # 누적 저장 수
    put(f"저장됨: {len(state['records'])}개", WINDOW_H-20, (100,220,100), 0.45, True)

    cv2.imshow(WINDOW_NAME, canvas)

# ── 마우스 콜백 ───────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    if x >= WINDOW_W - 316:
        return  # 패널 영역 무시

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(state['points']) < 6:
            ix, iy = display_to_img(x, y)
            ih, iw = state['orig_img'].shape[:2]
            ix = max(0, min(iw-1, ix))
            iy = max(0, min(ih-1, iy))
            state['points'].append((ix, iy))
        else:
            state['drag_start'] = (x, y)
            state['drag_offset'] = state['offset'][:]

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON and state['drag_start']:
            dx = x - state['drag_start'][0]
            dy = y - state['drag_start'][1]
            state['offset'][0] = state['drag_offset'][0] + dx
            state['offset'][1] = state['drag_offset'][1] + dy

    elif event == cv2.EVENT_LBUTTONUP:
        state['drag_start'] = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 우클릭 = 드래그 시작 (항상 가능)
        state['drag_start'] = (x, y)
        state['drag_offset'] = state['offset'][:]

    elif event == cv2.EVENT_RBUTTONUP:
        state['drag_start'] = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        factor = 1.15 if flags > 0 else 1/1.15
        # 마우스 위치 중심으로 줌
        old_z = state['zoom']
        state['zoom'] = max(0.1, min(10.0, state['zoom'] * factor))
        new_z = state['zoom']
        state['offset'][0] = int(x - (x - state['offset'][0]) * new_z / old_z)
        state['offset'][1] = int(y - (y - state['offset'][1]) * new_z / old_z)

    render()

# ── CSV 저장 ──────────────────────────────────────────────────────
def save_to_csv():
    m = calc_measurements()
    if len(state['points']) < 6:
        print("[경고] 6개 포인트를 모두 찍어야 저장됩니다.")
        return False

    fname = state['img_paths'][state['img_idx']].name
    subject_id = fname.rsplit('.', 1)[0]  # 확장자 제거

    record = {
        'subject_id': subject_id,
        'filename': fname,
        'nail_width_px': m.get('nail_width_px', ''),
        'nail_height_px': m.get('nail_height_px', ''),
        'left_skin_px': m.get('left_skin_px', ''),
        'right_skin_px': m.get('right_skin_px', ''),
        'ratio_w_h': m.get('ratio_w_h', ''),
        'ratio_left_w': m.get('ratio_left_w', ''),
        'ratio_right_w': m.get('ratio_right_w', ''),
        'pt1_x': round(state['points'][0][0], 1),
        'pt1_y': round(state['points'][0][1], 1),
        'pt2_x': round(state['points'][1][0], 1),
        'pt2_y': round(state['points'][1][1], 1),
        'pt3_x': round(state['points'][2][0], 1),
        'pt3_y': round(state['points'][2][1], 1),
        'pt4_x': round(state['points'][3][0], 1),
        'pt4_y': round(state['points'][3][1], 1),
        'pt5_x': round(state['points'][4][0], 1),
        'pt5_y': round(state['points'][4][1], 1),
        'pt6_x': round(state['points'][5][0], 1),
        'pt6_y': round(state['points'][5][1], 1),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    state['records'].append(record)
    state['saved_indices'].add(state['img_idx'])

    # CSV 파일에 즉시 기록 (안전하게)
    fieldnames = list(record.keys())
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"[저장] {subject_id} → {OUTPUT_CSV} (총 {len(state['records'])}개)")
    return True

# ── 이미지 이동 ───────────────────────────────────────────────────
def go_next():
    if state['img_idx'] < len(state['img_paths']) - 1:
        state['img_idx'] += 1
        load_image(state['img_idx'])
        render()
    else:
        print("[완료] 모든 이미지 처리 완료!")

def go_prev():
    if state['img_idx'] > 0:
        state['img_idx'] -= 1
        load_image(state['img_idx'])
        render()

# ── 메인 ─────────────────────────────────────────────────────────
def main():
    # 이미지 폴더 선택
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folder = DEFAULT_IMG_FOLDER

    # 이미지 수집
    paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    if not paths:
        print(f"[오류] '{folder}'에서 이미지를 찾을 수 없습니다.")
        print("사용법: python nailsize.py [이미지폴더경로]")
        print(f"        경로 생략시 기본 폴더: {DEFAULT_IMG_FOLDER}")
        sys.exit(1)

    print(f"[시작] {len(paths)}개 이미지 발견: {folder}")
    print(f"[출력] CSV 저장 위치: {os.path.abspath(OUTPUT_CSV)}")
    print()
    print("조작법: 클릭=포인트 / z=취소 / r=초기화 / s=저장+다음 / n=스킵 / b=이전 / q=종료")

    state['img_paths'] = paths
    state['img_idx'] = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    load_image(0)
    render()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print(f"\n[종료] 총 {len(state['records'])}개 저장됨 → {OUTPUT_CSV}")
            break

        elif key == ord('s'):
            if save_to_csv():
                go_next()

        elif key == ord('n'):
            print(f"[스킵] {state['img_paths'][state['img_idx']].name}")
            go_next()

        elif key == ord('b'):
            go_prev()

        elif key == ord('z'):
            if state['points']:
                state['points'].pop()
                render()

        elif key == ord('r'):
            state['points'] = []
            render()

        elif key in (ord('+'), ord('=')):
            state['zoom'] = min(10.0, state['zoom'] * 1.2)
            render()

        elif key == ord('-'):
            state['zoom'] = max(0.1, state['zoom'] / 1.2)
            render()

        # 창이 닫히면 종료
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()