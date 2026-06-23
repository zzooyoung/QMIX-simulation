# surgery_jackal.py
import os
from isaaclab.app import AppLauncher

# 화면 없이(headless) 백그라운드에서 조용히 수술 진행
app_launcher = AppLauncher(headless=True)
sim_app = app_launcher.app

from pxr import Usd
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def perform_surgery():
    input_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal.usd"
    # 현재 폴더에 수술이 끝난 깨끗한 로봇 파일을 저장합니다.
    output_usd = os.path.abspath("jackal_clean.usd")

    print("\n[수술 시작] 원본 자칼(Jackal) 로봇을 수술대에 올립니다...")
    print(f"원본 경로: {input_usd}")
    
    stage = Usd.Stage.Open(input_usd)
    if not stage:
        print("[치명적 오류] 원본 로봇 파일을 열 수 없습니다. Nucleus 서버 연결을 확인하세요.")
        return

    prims_to_remove = []
    
    # 로봇 몸통 구석구석을 스캔하며 에러의 원인(CPU 회로)을 찾습니다.
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        prim_name = prim.GetName().lower()
        
        # ActionGraph, OmniGraph, ROS2 통신 노드, 내장 오도메트리 센서 등 적발!
        if type_name in ["OmniGraph", "ActionGraph"] or "ros" in prim_name or "odom" in prim_name:
            prims_to_remove.append(prim.GetPath())

    if not prims_to_remove:
        print("[결과] 어라? 적출할 낡은 회로가 없습니다. 이미 깨끗한 상태입니다.")
    else:
        for path in prims_to_remove:
            print(f"  -> ✂️ [적출 완료] {path}")
            stage.RemovePrim(path)

    # 수술이 끝난 깨끗한 로봇을 로컬 폴더에 저장합니다.
    stage.GetRootLayer().Export(output_usd)
    print(f"\n[수술 대성공] 총 {len(prims_to_remove)}개의 악성 CPU 회로를 영구 제거했습니다!")
    print(f"[저장 완료] 신형 로봇 파일이 생성되었습니다: {output_usd}\n")

if __name__ == "__main__":
    perform_surgery()
    sim_app.close()