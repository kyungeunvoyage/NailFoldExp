import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from PIL import Image

# ==========================================
# 1. 경로 설정 및 사용자 데이터 디렉토리
# ==========================================
data_dir = '/Users/kyungeunjung/NailFoldExp/Data/(ATD)CurData'
base_image_path = '/Users/kyungeunjung/NailFoldExp/GraphicalAbstract/baseImg.png' # 베이스가 될 손톱 이미지 파일명 지정

# ==========================================
# 2. Region 좌표 설정 (a, b, c, d, e, f)
# ==========================================
# 실제 사용하는 'Group 29365.jpg' 이미지의 해상도에 맞춰 각 점의 (x, y) 픽셀 좌표를 입력해야 합니다.
# Mac의 미리보기(Preview) 등에서 마우스 커서를 올리면 픽셀 좌표를 확인할 수 있습니다.
# (아래는 임의로 설정한 예시 좌표입니다)
regions = ['a', 'b', 'c', 'd', 'e', 'f']
coords = {
    'a': (120, 250), # 좌측 상단
    'b': (100, 420), # 좌측 중단
    'c': (280, 520), # 좌측 하단
    'd': (480, 520), # 우측 하단
    'e': (660, 420), # 우측 중단
    'f': (640, 250)  # 우측 상단
}

def generate_overlay_heatmap(metric_name, data_values, output_filename):
    try:
        # 베이스 이미지 로드
        img = Image.open(base_image_path)
        img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Error: {base_image_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 좌표와 데이터 분리
    x_coords = [coords[region][0] for region in regions]
    y_coords = [coords[region][1] for region in regions]
    z_values = [data_values[region] for region in regions]

    # Rbf (Radial Basis Function)를 사용한 부드러운 보간(Interpolation)
    # 손가락 모양처럼 비정형적인 포인트 배열에서 부드러운 히트맵을 만들 때 유리합니다.
    rbf = Rbf(x_coords, y_coords, z_values, function='thin_plate')

    # 이미지 전체 픽셀에 대한 그리드 생성
    xi = np.linspace(0, img_width, img_width)
    yi = np.linspace(0, img_height, img_height)
    XI, YI = np.meshgrid(xi, yi)

    # 그리드에 보간된 데이터 적용
    ZI = rbf(XI, YI)
    
    # 히트맵이 손가락 바깥으로 너무 퍼지지 않게 제한 (선택적 마스킹)
    # ZI 값이 특정 범위를 넘어가면 투명 처리하는 등의 후처리를 넣을 수 있습니다.
    
    # 시각화 설정
    fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
    
    # 1. 배경 이미지 출력
    ax.imshow(img)
    
    # 2. 히트맵 오버레이 (alpha 값으로 투명도 조절, cmap으로 색상 맵 설정)
    # cmap 추천: detection/accuracy는 보통 'RdYlGn' (Red-Yellow-Green) 혹은 'jet' 사용
    heatmap = ax.imshow(ZI, extent=(0, img_width, img_height, 0), 
                        alpha=0.5, cmap='RdYlGn', vmin=min(z_values), vmax=max(z_values))
    
    # 각 포인트 마커 표시 (검은색 점)
    ax.scatter(x_coords, y_coords, color='black', s=100, zorder=5)
    
    # 포인트 레이블(a,b,c...) 표시
    for region in regions:
        ax.text(coords[region][0]-30, coords[region][1]+10, region, 
                color='black', fontsize=16, fontweight='bold', zorder=6)

    # 축 숨기기
    ax.axis('off')
    
    # 컬러바 추가
    cbar = fig.colorbar(heatmap, ax=ax, shrink=0.7)
    cbar.set_label(f'{metric_name}', fontsize=12)

    # 결과 저장
    save_path = os.path.join(data_dir, output_filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"[{metric_name}] 히트맵이 성공적으로 저장되었습니다: {save_path}")

# ==========================================
# 3. 데이터 로드 및 실행
# ==========================================
if __name__ == "__main__":
    # TODO: 실제 환경에서는 pd.read_csv() 등을 이용하여 데이터를 로드하세요.
    # 예시: df = pd.read_csv(os.path.join(data_dir, 'accuracy_data.csv'))
    
    # 임시 테스트용 데이터 (0 ~ 100% 범위 가정)
    dummy_detection_data = {'a': 80.5, 'b': 92.0, 'c': 65.0, 'd': 70.0, 'e': 95.0, 'f': 82.0}
    dummy_discrimination_data = {'a': 75.0, 'b': 85.0, 'c': 60.0, 'd': 62.0, 'e': 88.0, 'f': 78.0}
    dummy_spatial_data = {'a': 3.5, 'b': 2.1, 'c': 5.0, 'd': 4.8, 'e': 1.5, 'f': 3.2} # 단위: mm 등

    # 히트맵 생성 실행
    print("히트맵 생성을 시작합니다...")
    generate_overlay_heatmap("Detection Accuracy (%)", dummy_detection_data, "Heatmap_Detection.png")
    generate_overlay_heatmap("Discrimination Accuracy (%)", dummy_discrimination_data, "Heatmap_Discrimination.png")
    generate_overlay_heatmap("Spatial Accuracy (mm)", dummy_spatial_data, "Heatmap_Spatial.png")