# 🚀 Multi-Agent Defect Mapping using QMIX in Isaac Lab

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NVIDIA%20Isaac%20Lab-2.x-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Isaac Lab">
  <img src="https://img.shields.io/badge/NVIDIA%20Isaac%20Sim-4.5.0-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Isaac Sim">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat-square&logo=ubuntu&logoColor=white" alt="Ubuntu">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License">
</p>

본 프로젝트는 **NVIDIA Isaac Lab v2.x** 환경에서 4족 보행 로봇(**ANYmal C**) 3대를 활용하여, 극한의 비정형 결함 구조물을 협력 탐색 및 매핑하는 **QMIX 기반 멀티 에이전트 강화학습(MARL)** 프레임워크입니다. 

이 저장소는 중앙 집중형 학습 및 분산 실행(**CTDE, Centralized Training with Decentralized Execution**) 아키텍처를 시뮬레이션 환경 레벨에서 완벽히 지원하도록 설계되었습니다.

---

## 📺 Demo

> 💡 *여기에 시뮬레이션 구동 화면(GIF 또는 YouTube 링크)을 추가해 주세요.*
<p align="center">
  <img src="https://github.com/user-attachments/assets/example-demo.gif" width="80%" alt="Simulation Demo"/>
</p>

---

## 🛠️ Tech Stack

| 분류 | 기술 및 프레임워크 | 비고 |
| :--- | :--- | :--- |
| **OS / Infra** | Ubuntu 22.04 LTS, Docker, X11 Socket Forwarding | 호스트 GUI 직접 송출 지원 |
| **Simulator** | NVIDIA Isaac Sim 4.5.0 / Isaac Lab 2.x | 고성능 로봇 강화학습 코어 |
| **Hardware Accelerate**| NVIDIA RTX 5080 (Blackwell), CUDA 13.0 | 하드웨어 가속 및 렌더링 |
| **RL Framework** | PyTorch, MARL QMIX (DRQN + Mixing Network) | CTDE 패러다임 구현 |
| **Physics Engine** | PhysX (CPU Mode Optimized for Multi-Agent) | 안정적인 다중 객체 충돌 연산 |

---

## ✨ Key Features

* **QMIX 기반 CTDE 데이터 파이프라인 완벽 구현**
  * **분산 관측 ($o_t^a$):** 각 에이전트 로봇 고유의 12개 관절 상태(`joint_pos`) 및 전용 가상 LiDAR 독립 수집.
  * **글로벌 상태 ($s_t$):** 중앙 가치 믹싱 네트워크(Mixing Network) 주입을 위한 전체 에이전트 센서 통합 마스터 상태 생성.
* **프로시저럴 비정형 지형 생성 (Procedural Rough Terrain)**
  * 매 시뮬레이션마다 확률적으로 변화하는 노이즈 기반 콘크리트 및 구조물 결함 굴곡 지형 실시간 빌딩.
* **고밀도 가상 LiDAR 레이캐스터 장착 (RayCaster Grid Pattern)**
  * 로봇당 2m x 2m 영역을 0.1m 고해상도로 스캔하는 441개 레이저 광선 실시간 충돌 검사 (`Shape: [1, 441, 3]`).
* **RTX 5080 및 Blackwell 아키텍처 최적화**
  * 도커 공유 메모리 확장 기법 및 PhysX 메모리 풀 할당 제한 우회를 통한 안정적인 3D 물리 루프 보장.

---

## 🏗️ System Architecture

```text
               [ Centralized Training (학습 단계) ]
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────┐       ┌───────────────────────┐ │
│ │  Robot A Local Obs (12) │ ───>  │     DRQN Agent A      │ │ ──┐
│ └─────────────────────────┘       └───────────────────────┘ │   │
│ ┌─────────────────────────┐       ┌───────────────────────┐ │   │  [Individual Qs]
│ │  Robot B Local Obs (12) │ ───>  │     DRQN Agent B      │ │ ──┼─> [Q_a, Q_b, Q_c]
│ └─────────────────────────┘       └───────────────────────┘ │   │          │
│ ┌─────────────────────────┐       ┌───────────────────────┐ │   │          ▼
│ │  Robot C Local Obs (12) │ ───>  │     DRQN Agent C      │ │ ──┘   ┌──────────────┐
│ └─────────────────────────┘       └───────────────────────┘ │   │       │  Mixing Net  │ ──> Total Q_tot
└─────────────────────────────────────────────────────────────┘   │       └──────────────┘
                                                                  │              ▲
       ┌──────────────────────────────────────────────────────────┘              │ [Dynamic Weights]
       │ ┌────────────────────────────────────────────────────────┐              │
       └─│  Global State (s_t) : 3x LiDAR Point Cloud (1323, 3)   │ ──> Hypernetworks
         └────────────────────────────────────────────────────────┘